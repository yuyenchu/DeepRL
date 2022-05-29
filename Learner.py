import gym
import os
# import threading
import numpy as np
import tensorflow as tf
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

# class Learner(threading.Thread):
class Learner(OBJ):
    # def __init__(self):
    #     # threading stuff
    #     # threading.Thread.__init__(self)
    #     super(Learner, self).__init__()
    def _initialize(self, id, gym_name, memory, save_path, memlock, netlock, weights_register, update_target_per_batch=3,\
                seed=None, verbose=False, n_step=1, gamma=0.99, batch_size=1024, mini_batch_num=2, **settings):
        # self.kill = threading.Event()
        self.kill = worker.Event()
        self.id = id
        self.verbose = verbose
        self.memlock = memlock
        self.netlock = netlock
        self.weights_register = weights_register
        # declaring objects
        self.env = gym.make(gym_name)
        self.in_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.memory = memory
        settings['middle_layer'] = models.LAYERS
        self.agent = DQNagent(memory, save_path, self.in_shape, self.num_actions, n_step=n_step, gamma=gamma, message=self.message, **settings)
        # constant values
        self.batch_size = batch_size
        self.mini_batch_num = mini_batch_num
        self.update_target_per_batch = update_target_per_batch
        # non-constant values
        self.batches = 0
        # setting seed if value is acceptable
        if isinstance(seed, int) and seed > 0:
            np.random.seed(seed)
        self.put_weights()

    # save / load necessary data for rebuilding
    def save(self):
        self.agent.save()

    def load(self):
        self.agent.load()

    # print formatting
    def message(self, *msg):
        if self.verbose:
            # print("["+str(datetime.now())+"] Learner", self.id, "-", *msg)
            msg_out = f'[{str(datetime.now())},{os.getpid()}] Learner {self.id} - {" ".join(msg)}'
            # print(msg_out)
            os.system(f'echo "{msg_out}"')
    
    # wrapper for agent.get_weights()
    def get_weights(self):
        return self.agent.get_weights()

    def put_weights(self):
        w1, w2 = self.agent.get_weights()
        weights = w1+w2
        weights = b''.join([w.tobytes() for w in weights])
        self.weights_register.value = weights

    # main loop
    def run(self):
        self._initialize(**self._kwargs)
        self.message("started")
        while not self.kill.is_set():
            if len(self.memory) > self.batch_size:
                # sampling a prioritized batch of memory
                self.memlock.acquire()
                self.message(f"start batch {self.batches}")
                s,s_n,a,r,d,idx,w,nid,*_ = self.agent.sample_replay(self.batch_size)
                self.memlock.release()

                # training network
                loss = 0
                target_q = self.agent.target_q(s_n, r, d)
                mini_idx = [i*round(self.batch_size/self.mini_batch_num) for i in range(self.mini_batch_num)]+[self.batch_size]
                for i in range(1,self.mini_batch_num+1):
                    # self.netlock.acquire()
                    start, end = mini_idx[i-1], mini_idx[i]
                    loss += self.agent.train(s[start:end], a[start:end], target_q[start:end], w[start:end]).numpy() 
                    # self.netlock.release()
                    self.put_weights()
                self.message(f"training network, loss = {loss}")

                # computing the new TD error
                target_q = self.agent.target_q(s_n, r, d)
                masks = tf.one_hot(a, self.num_actions)
                q_values = self.agent.model(s)
                q_val = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                priority = q_val - target_q

                # updating memory with the new priority
                self.memlock.acquire()
                for i, p, n in zip(idx, priority, nid):
                    self.memory.update(i, p, n)
                self.message(f"end batch {self.batches}")
                self.memlock.release()

                self.batches+=1
                # updating target model weights
                if self.batches%self.update_target_per_batch == 0:
                    w1, _ = self.agent.get_weights()
                    # self.netlock.acquire()
                    self.message("updating target network")
                    self.agent.set_weights(w1)
                    self.put_weights()
                    # self.netlock.release()
        self.message("stopped")