from PMemory import PMemory
from DQNagent import DQNagent
from Actor import Actor
import numpy as np
import time 
from time import sleep
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import threading

def g():
    return 1, 2
def f(a,b):
    print(b,";",a)

f(*g())
# N = 100000
# memlock = threading.Lock()
# netlock = threading.Lock()
# pm = PMemory(N, "./test_folder/testPM.h5")
# a = []
# for i in range(3):
#     a.append(Actor(i, "CartPole-v0",pm, "./test_folder", memlock, netlock,verbose=True))
# for b in a:
#     b.start()

# # testing killing thread
# sleep(3)
# print("sleep over")
# for b in a:
#     b.kill.set() 
# for b in a:
#     b.join()
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