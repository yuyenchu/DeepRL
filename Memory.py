import h5py
import numpy as np
from numpy import random

class Memory:
    def __init__(self, capacity, path):
        self.capacity = capacity
        self.states = []
        self.states_next = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.file_path = path
        if not self.file_path.endswith(".h5"):
            self.file_path += ".h5"
    
    def __len__(self):
        return len(self.dones)
    
    def __str__(self):
        return str((self.states, self.states_next, self.actions, self.rewards, self.dones))

    def add(self, sample, error):
        s, s_n, a, r, d = sample
        self.states += s
        self.states_next += s_n 
        self.actions += a
        self.rewards += r
        self.dones += d

        if len(d) > self.capacity:
            del self.states[:1]
            del self.states_next[:1]
            del self.actions[:1]
            del self.rewards[:1]
            del self.dones[:1]

    def sample(self, n):
        indices = random.choice(range(len(self.dones)), size=n)

        s = np.array([self.states[i] for i in indices])
        s_n = np.array([self.states_next[i] for i in indices])
        a = [self.actions[i] for i in indices]
        r = [self.rewards[i] for i in indices]
        d = tf.convert_to_tensor([float(self.dones[i]) for i in indices])

        return [s, s_n, a, r, d], [], []

    def save(self):
        h5f = h5py.File(self.file_path, 'w')
        h5f.create_dataset('tree', data=self.tree.tree)
        h5f.create_dataset('data', data=self.tree.data.tolist())
        h5f.create_dataset('n_entries', data=self.tree.n_entries)
        h5f.create_dataset('write', data=self.tree.write)
        h5f.close()

    def load(self):
        h5f = h5py.File(self.file_path,'r')
        self.tree.tree = h5f['tree'][:]
        self.tree.data = h5f['data'][:]
        self.tree.n_entries = h5f['n_entries'][()]
        self.tree.write = h5f['write'][()]
        h5f.close()