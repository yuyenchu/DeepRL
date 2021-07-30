import gym
import queue
import threading
import numpy as np
from numpy import random
from datetime import datetime

from DQNagent import DQNagent

class Actor(threading.Thread):
    def __init__(self, id, gym_name, memory, save_path, memlock, netlock, get_weights, seed=None, 
                verbose=False, net_update_per_epi=100, max_buffer_length=10000, n_step=1, gamma=0.99,\
                epsilon=1.0, epsilon_min=0.2, epsilon_decay=0, random_act=0, max_frame_per_episode=-1, **settings):
        # threading stuff
        threading.Thread.__init__(self)
        self.kill = threading.Event()
        self.id = id
        self.verbose = verbose
        self.memlock = memlock
        self.netlock = netlock
        self.get_weights = get_weights
        # declaring objects
        self.env = gym.make(gym_name)
        self.in_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.agent = DQNagent(memory, save_path, self.in_shape, self.num_actions, n_step=n_step, gamma=gamma, message=self.message, **settings)
        # constant values
        self.n_step = n_step
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random_act = random_act
        self.net_update_per_epi = net_update_per_epi
        self.max_buffer_length = max_buffer_length
        self.max_frame_per_episode = max_frame_per_episode
        # non-constant values
        self.epsilon = epsilon
        self.frames = 0
        self.episodes = 0
        self.buffer = {'state':[],'state_next':[],'action':[],'reward':[],'done':[]}
        # setting seed if value is acceptable
        if isinstance(seed, int) and seed > 0:
            self.env.seed(seed)
            np.random.seed(seed)

    # print formatting
    def message(self, *msg):
        if self.verbose:
            print("["+str(datetime.now())+"] Actor", self.id, "-", *msg)

    # getting network parameters from learner
    def update_weights(self):
        self.netlock.acquire()
        self.message("updating network weights")
        self.agent.set_weights(*self.get_weights())
        self.netlock.release()
    
    # main loop
    def run(self):
        self.message("started")
        while not self.kill.is_set():
            state = np.array(self.env.reset())
            epi_frame = 1
            # episode start
            while self.max_frame_per_episode<0 or epi_frame<=self.max_frame_per_episode:
                action = self.agent.forward(state, (self.frames<self.random_act or random.uniform()<self.epsilon))
                state_next, reward, done, _ = self.env.step(action)
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
                    self.message("saving memory at episode",self.episodes)
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

            # episode end
            self.episodes+=1

            # getting latest network parameters from learner
            if self.episodes%self.net_update_per_epi == 0:
                self.update_weights()
        self.message("stopped")