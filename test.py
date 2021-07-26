from PMemory import PMemory
from DQNagent import DQNagent
import numpy as np
import time 

N = 1000
MAX = 10**5
pm = PMemory(N, "./testPM.h5")
a = DQNagent(pm, "./test_folder")
a.gamma = 0.99
a.save()
a.load()

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