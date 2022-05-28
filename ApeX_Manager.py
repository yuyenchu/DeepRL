import numpy as np
# import threading

from PMemory import PMemory
from Actor import Actor
from Learner import Learner

import config as cfg
if cfg.USE_MULTIPROCESSING:
    import multiprocessing as worker
    from Multi_PMemory import PMemory
else:
    import threading as worker
    from PMemory import PMemory
class Manager(object):
    def __init__(self, mem_save_path, MEM_LENGTH=10000, ACTORS=10,\
                BASIC_SETTING={}, MEMORY_SETTING={}, LEARNER_SETTING={}, ACTOR_SETTING={},\
                e=0.4, a=7):
        # setting memory pool
        self.memory = PMemory(MEM_LENGTH, mem_save_path, **MEMORY_SETTING)
        # setting thread locks
        # self.memlock = threading.Lock()
        # self.netlock = threading.Lock()
        self.memlock = worker.Lock()
        self.netlock = worker.Lock()
        BASIC_SETTING["memory"]  = self.memory
        BASIC_SETTING["memlock"] = self.memlock
        BASIC_SETTING["netlock"] = self.netlock
        # create learner
        self.learner = Learner(**BASIC_SETTING, **LEARNER_SETTING)
        self.agents = [self.learner]
        # create actors
        if isinstance(ACTOR_SETTING, list):
            # condition if setting is provided for each actor Individually 
            if len(ACTOR_SETTING) != ACTORS:
                raise ValueError("length of ACTOR_SETTING does not match the amout of actors")
            for setting in ACTOR_SETTING:
                self.agents.append(Actor(**BASIC_SETTING, **setting,\
                                        get_weights=self.learner.get_weights,\
                                        kill_all_threads=self.kill_all_threads))
        else: 
            # defult for actors sharing same setting
            for i in range(ACTORS):
                self.agents.append(Actor(i, **BASIC_SETTING, **ACTOR_SETTING,\
                                        epsilon=e**(1+i*a/(ACTORS-1)),\
                                        get_weights=self.learner.get_weights, \
                                        kill_all_threads=self.kill_all_threads))
        
    # start all agents, including both learner and actors
    def start(self):
        for a in self.agents:
            a.start()

    # save / load necessary data for rebuilding
    def save(self):
        self.learner.save()
        self.memory.save()

    def load(self):
        self.learner.load()
        self.memory.load()
        for a in self.agents[1:]:
            a.update_weights()

    # stop all agents 
    def kill_all_threads(self):
        for a in self.agents:
            a.kill.set() 
        for a in self.agents:
            a.join()
        print("-"*30,"Done","-"*30)