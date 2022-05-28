import multiprocessing as mp
import threading as td
import os
import sys

MP = True
obj = mp.Process if MP else td.Thread
class a(obj):
    def __init__(self,n, i):
        super().__init__()
        self.a = mp.Array('i',n)
        self.b = []
        self.n = n
        self.i = i
    def f(self, i):
        self.a[i] = i
    def g(self, i):
        self.b.append(i)
    def k(self, i):
        self.f(i)
        self.g(i)
    def run(self):
        os.system(f'echo "process {self.i}"')
        self.n[self.i]=self.i
        
def f(obj,i):
    obj.k(i)
if __name__ == '__main__':
    # A = a(10)
    arr = mp.Array('i',10)
    p = []
    for i in range(10):
        # p.append(mp.Process(target=f, args=(A,i)))
        p.append(a(arr,i))
        p[-1].start()
    for pp in p:
        pp.join()
    # print([A.a[i] for i in range(10)])
    print([arr[i] for i in range(10)])
    # print(A.b)