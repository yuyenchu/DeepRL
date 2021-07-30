from PMemory import PMemory
from DQNagent import DQNagent
from Actor import Actor
from Learner import Learner
import numpy as np
import time 
from time import sleep
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import threading
from nanoid import generate

arr = np.zeros(3,dtype='S21')
temp = generate()
print(temp)
arr[0]=temp
print(arr[0])

# N = 100000
# memlock = threading.Lock()
# netlock = threading.Lock()
# pm = PMemory(N, "./test_folder/testPM.h5")
# setting = {
#     "gym_name":"CartPole-v0",
#     "memory":pm, 
#     "save_path":"./test_folder", 
#     "memlock":memlock, 
#     "netlock":netlock, 
#     "verbose":True,
#     "n_step":1, 
#     "gamma":0.99
# }
# l = Learner("main", **setting)
# a = []
# for i in range(3):
#     a.append(Actor(i, **setting, get_weights=l.get_weights, net_update_per_epi=400, epsilon=1-i/10))
#     # print(a[i].epsilon)
# for b in a:
#     b.start()
# l.start()
# # testing killing thread
# sleep(1)
# print("sleep over")
# for b in a:
#     b.kill.set() 
# l.kill.set() 
# for b in a:
#     b.join()
# l.join()
# print("done with",len(pm),"memory")

# # testing saving / loading agent
# a = DQNagent(pm, "./test_folder")
# print(a.save_path)
# a.gamma = 0.99
# a.save()
# a.load()

# # testing sampling time with different implementations
# MAX = 10**5
# for i in range (MAX):
#     pm.add(i,[[i],[i+1]])

# np.random.seed(0)
# start = time.time()
# a = pm.sample(N)
# end = time.time()
# print("time =", end-start)

# np.random.seed(0)
# start = time.time()
# b = pm.sample(N)
# end = time.time()
# print("time =", end-start)

# # testing combination of tf sequentialmodel and functional api
# seq = keras.Sequential(
#     [
#         layers.Dense(2, activation="relu", name="seq1"),
#         layers.Dense(3, activation="relu", name="seq2"),
#         layers.Dense(4, name="seq3"),
#     ]
# )
# inputs = keras.Input(shape=(784,))
# layer1 = layers.Dense(2, activation="relu", name="layer1")(inputs)
# layer2 = seq(layer1)
# layer3 = layers.Dense(3, activation="relu", name="layer2")(layer2)
# out = layers.Dense(4, name="layer3")(layer3)

 
# model = keras.Model(inputs=inputs, outputs=out)
# model.summary()

# # testing passing class func to another class
# class c1:
#     def __init__(self, f):
#         self.f = f
#     def g(self):
#         self.f("ha")

# class c2:
#     def __init__(self):
#         self.var = 3
#         self.c = c1(self.f)
#     def f(self, x):
#         print(self.var,";",x)

# c = c2()
# c.c.g()