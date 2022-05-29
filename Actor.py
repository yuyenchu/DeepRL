import gym
import os
# import threading
import numpy as np
from numpy import random
from datetime import datetime

import models
import config as cfg
if cfg.USE_MULTIPROCESSING:
    import multiprocessing as worker
    OBJ = worker.Process
else:
    import threading as worker
    OBJ = worker.Thread
from DQNagent import DQNagent

# class Actor(threading.Thread):
class Actor(OBJ):
    # def __init__(self):
    #     # threading stuff
    #     # threading.Thread.__init__(self)
    #     super(Actor, self).__init__()

    def _initialize(self, id, gym_name, memory, save_path, memlock, netlock, weights_register,\
                seed=None, verbose=False, net_update_per_epi=100, max_buffer_length=10000, n_step=1,\
                gamma=0.99, epsilon=1.0, epsilon_min=0.2, epsilon_decay=0, random_act=0, target_reward=None,\
                max_frame_per_episode=-1, max_reward_length=100, **settings):
        # self.kill = threading.Event()
        self.kill = worker.Event()
        self.id = id
        self.verbose = verbose
        self.memlock = memlock
        self.netlock = netlock
        self.weights_register = weights_register
        # self.get_weights = get_weights
        # self.kill_all_threads = kill_all_threads
        # declaring objects
        self.env = gym.make(gym_name)
        self.in_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        settings['middle_layer'] = models.LAYERS
        self.agent = DQNagent(memory, save_path, self.in_shape, self.num_actions, n_step=n_step, gamma=gamma, message=self.message, **settings)
        # constant values
        self.n_step = n_step
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random_act = random_act
        self.target_reward = target_reward
        self.net_update_per_epi = net_update_per_epi
        self.max_buffer_length = max_buffer_length
        self.max_frame_per_episode = max_frame_per_episode
        self.max_reward_length = max_reward_length
        # non-constant values
        self.epsilon = epsilon
        self.frames = 0
        self.episodes = 0
        self.rewards = []
        self.buffer = {'state':[],'state_next':[],'action':[],'reward':[],'done':[]}
        # setting seed if value is acceptable
        if isinstance(seed, int) and seed > 0:
            self.env.seed(seed)
            np.random.seed(seed)
        self.weights_info = self._get_weights_info()
        self.update_weights()

    def _get_weights_info(self):
        w1, w2 = self.agent.get_weights()
        weights = w1+w2
        return [(w.shape, w.dtype, len(w.tobytes())) for w in weights]

    # print formatting
    def message(self, *msg):
        if self.verbose:
            # print("["+str(datetime.now())+"] Actor", self.id, "-", *msg)
            msg_out = f'[{str(datetime.now())},{os.getpid()}] Actor {self.id} - {" ".join(msg)}'
            # print(msg_out)
            os.system(f'echo "{msg_out}"')

    # getting network parameters from learner
    def update_weights(self):
        # self.netlock.acquire()
        # self.message("updating network weights")
        # self.agent.set_weights(*self.get_weights())
        # self.netlock.release()
        if len(self.weights_register.value) == 0:
            return
        weights = self.weights_register.value
        # print('actor',len(weights),len(self.weights_info))
        # assert len(weights) == len(self.weights_info)
        decoded_weights = []
        idx = 0
        for s, t, l in self.weights_info:
            w = weights[idx:idx+l]
            idx += l
            weight = np.frombuffer(w, dtype=t)
            if len(s) > 0:
                weight = weight.reshape(s)
            decoded_weights.append(weight)
        idx = len(decoded_weights)//2
        w1, w2 = decoded_weights[:idx], decoded_weights[idx:]
        self.agent.set_weights(w1, w2)

    # main loop
    def run(self):
        self._initialize(**self._kwargs)
        self.message("started")
        running_reward = 0
        while not self.kill.is_set():
            state = np.array(self.env.reset())
            epi_frame = 1
            epi_reward = 0
            # episode start
            while self.max_frame_per_episode<0 or epi_frame<=self.max_frame_per_episode:
                action = self.agent.forward(state, (self.frames<self.random_act or random.uniform()<self.epsilon))
                state_next, reward, done, _ = self.env.step(action)
                epi_reward+=reward
                ################################
                # saving to local buffer
                self.buffer['state'].append(state)
                self.buffer['action'].append(action)
                self.buffer['done'].append(done)
                if epi_frame > 1:
                    for n in range(1, min(self.n_step, epi_frame)):
                        self.buffer['reward'][-n] += (self.gamma**n)*reward
                if epi_frame > self.n_step:
                    self.buffer['state_next'][-self.n_step] = state_next
                self.buffer['state_next'].append(np.full(self.in_shape, np.nan))
                self.buffer['reward'].append(reward)

                # dump buffer to memory pool
                if len(self.buffer['done']) > self.max_buffer_length:
                    self.memlock.acquire()
                    self.message(f"saving memory at episode {self.episodes} reward = {running_reward}")
                    self.agent.save_memory( self.buffer['state'][:-self.n_step],\
                                            self.buffer['state_next'][:-self.n_step],\
                                            self.buffer['action'][:-self.n_step],\
                                            self.buffer['reward'][:-self.n_step],\
                                            self.buffer['done'][:-self.n_step])
                    self.memlock.release()
                    # cleanup buffer
                    del self.buffer['state'][:-self.n_step]
                    del self.buffer['state_next'][:-self.n_step]
                    del self.buffer['action'][:-self.n_step]
                    del self.buffer['reward'][:-self.n_step]
                    del self.buffer['done'][:-self.n_step]
                ################################
                state = np.array(state_next)
                self.frames+=1
                epi_frame+=1
                self.epsilon = max(self.epsilon_min, self.epsilon-self.epsilon_decay)
                if self.kill.is_set() or done:
                    break

            # episode end and checking current running reward
            self.episodes+=1
            self.rewards.append(epi_reward)
            if len(self.rewards) > self.max_reward_length:
                del self.rewards[:1]
            running_reward = np.mean(self.rewards)
            if self.target_reward and running_reward > self.target_reward:
                self.message(f"target reward reached with running reward {running_reward}")
                # self.kill_all_threads()
            # getting latest network parameters from learner
            if self.episodes%self.net_update_per_epi == 0:
                self.update_weights()
        self.message("stopped")