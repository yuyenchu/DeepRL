import gym
import queue
import threading
import numpy as np
import tensorflow as tf
from numpy import random
from datetime import datetime

from DQNagent import DQNagent

class Learner(threading.Thread):
    def __init__(self, id, gym_name, memory, save_path, memlock, netlock, update_target_per_batch=30,\
                seed=None, verbose=False, n_step=1, gamma=0.99, batch_size=1024, **settings):
        # threading stuff
        threading.Thread.__init__(self)
        self.kill = threading.Event()
        self.id = id
        self.verbose = verbose
        self.memlock = memlock
        self.netlock = netlock
        # declaring objects
        self.env = gym.make(gym_name)
        self.in_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.memory = memory
        self.agent = DQNagent(memory, save_path, self.in_shape, self.num_actions, n_step=n_step, gamma=gamma, message=self.message, **settings)
        # constant values
        self.batch_size = batch_size
        self.update_target_per_batch = update_target_per_batch
        # non-constant values
        self.batches = 0
        # setting seed if value is acceptable
        if isinstance(seed, int) and seed > 0:
            np.random.seed(seed)

    # save / load necessary data for rebuilding
    def save(self):
        self.agent.save()

    def load(self):
        self.agent.load()

    # print formatting
    def message(self, *msg):
        if self.verbose:
            print("["+str(datetime.now())+"] Learner", self.id, "-", *msg)
    
    # wrapper for agent.get_weights()
    def get_weights(self):
        return self.agent.get_weights()
    
    # main loop
    def run(self):
        self.message("started")
        while not self.kill.is_set():
            if len(self.memory) > self.batch_size:
                # sampling a prioritized batch of memory
                self.memlock.acquire()
                self.message("start batch",self.batches)
                s,s_n,a,r,d,idx,w,nid,*_ = self.agent.sample_replay(self.batch_size)
                self.memlock.release()

                # training network
                target_q = self.agent.target_q(s_n, r, d)
                self.netlock.acquire()
                self.message("training network, loss =", self.agent.train(s, a, target_q, w))
                self.netlock.release()

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
                self.message("end batch",self.batches)
                self.memlock.release()

                self.batches+=1
                # updating target model weights
                if self.batches%self.update_target_per_batch == 0:
                    w1, _ = self.agent.get_weights()
                    self.netlock.acquire()
                    self.message("updating target network")
                    self.agent.set_weights(w1)
                    self.netlock.release()
        self.message("stopped")