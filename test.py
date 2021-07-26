from PMemory import PMemory
import numpy as np
import time 

N = 1000
MAX = 10**5
pm = PMemory(MAX, "./testPM.h5")

for i in range (MAX):
    pm.add(i,[[i],[i+1]])

np.random.seed(0)
start = time.time()
a = pm.sample(N)
end = time.time()
print("time =", end-start)

np.random.seed(0)
start = time.time()
b = pm.sample(N)
end = time.time()
print("time =", end-start)