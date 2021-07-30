import numpy as np
import time 
import threading

from PMemory import PMemory
from Actor import Actor
from Learner import Learner

class Manager(object):
    def __init__(self, mem_save_path, MEM_LENGTH=10000, ACTORS=10,\
                BASIC_SETTING={}, LEARNER_SETTING={}, ACTOR_SETTING={}):
        self.memory = PMemory(MEM_LENGTH, mem_save_path)
        self.memlock = threading.Lock()
        self.netlock = threading.Lock()
        BASIC_SETTING["memory"]  = self.memory
        BASIC_SETTING["memlock"] = self.memlock
        BASIC_SETTING["netlock"] = self.netlock
        self.learner = Learner(**BASIC_SETTING, **LEARNER_SETTING)
        self.agents = [self.learner]
        if isinstance(ACTOR_SETTING, list):
            if len(ACTOR_SETTING) != ACTORS:
                raise ValueError("length of ACTOR_SETTING does not math the amout of actors")
            for setting in ACTOR_SETTING:
                self.agents.append(Actor(**BASIC_SETTING, **setting, get_weights=self.learner.get_weights))
        else:    
            for i in range(ACTORS):
                self.agents.append(Actor(i, **BASIC_SETTING, **ACTOR_SETTING, get_weights=self.learner.get_weights, epsilon=1-i/ACTORS))
        
    def start(self):
        for a in self.agents:
            a.start()

    def save(self):
        self.manager.save()
        self.memory.save()

    def load(self):
        self.manager.load()
        self.memory.load()
        for a in self.agents[1:]:
            a.update_weights()

    def kill_all_threads(self):
        for a in self.agents:
            a.kill.set() 
        for a in self.agents:
            a.join()
        print("-"*30,"Done","-"*30)