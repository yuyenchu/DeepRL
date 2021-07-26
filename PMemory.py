import h5py
import numpy as np
from numpy import random

class PMemory:
    def __init__(self, capacity, path, e = 0.01, a = 0.8, beta = 0.3, beta_increment_per_sampling = 0.0005):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.file_path = path
        if not self.file_path.endswith(".h5"):
            self.file_path += ".h5"
    
    def __len__(self):
        return len(self.tree)
    
    def __str__(self):
        return str(self.tree)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, sample, error):
        if isinstance(error, np.ndarray):
            def priority_wrap(s): 
                return self._get_priority(s)

            priority = np.vectorize(priority_wrap)(error)
            for p, s in zip(priority, sample):
                self.tree.add(p, s)

        elif isinstance(error, list):
            for e, s in zip(error, sample):
                p = self._get_priority(e)
                self.tree.add(p, s)
        else:
            p = self._get_priority(error)            
            self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        s = np.arange(self.tree.total(), step=segment)+np.random.uniform(high=segment, size=n)

        (idx, p, data) = self.tree.get(s)

        sampling_probabilities = p / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return np.dstack(data), idx, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

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

class SumTree:
    def __init__(self, capacity, tree=None, data=None, n_entries=None, write=None):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) if tree is None else tree
        self.data = np.zeros(capacity, dtype=object) if data is None else data
        self.n_entries = 0 if n_entries is None else n_entries
        self.write = 0 if write is None else write

    def __len__(self):
        return self.n_entries
    
    def __str__(self):
        p = [self.tree[i+self.capacity-1] for i in range(self.n_entries)]
        p = np.argsort(p)
        return str(self.data[p].tolist())

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        def retrieve_wrap(s): 
            return self._retrieve(0, s)

        idx = np.vectorize(retrieve_wrap)(s)
        # idx = np.array(list(map(retrieve_wrap, s)))
        # idx = np.frompyfunc(retrieve_wrap, 1, 1)(s).astype(np.uint32)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



    # def sample(self, n):
    #     batch = []
    #     idxs = []
    #     segment = self.tree.total() / n
    #     priorities = []

    #     self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

    #     for i in range(n):
    #         a = segment * i
    #         b = segment * (i + 1)

    #         s = random.uniform(a, b)
    #         (idx, p, data) = self.tree.get(s)
    #         priorities.append(p)
    #         batch.append(data)
    #         idxs.append(idx)

    #     sampling_probabilities = priorities / self.tree.total()
    #     is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
    #     is_weight /= is_weight.max()

    #     return np.dstack(batch), idxs, is_weight

    # def get(self, s: float):
    #     idx = self._retrieve(0, s)
    #     dataIdx = idx - self.capacity + 1

    #     return (idx, self.tree[idx], self.data[dataIdx])