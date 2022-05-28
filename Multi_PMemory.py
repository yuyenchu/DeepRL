from posix import environ
import h5py
import numpy as np
import gym
from  multiprocessing import Array
from ctypes import c_char_p
# from numpy import random
from nanoid import generate

class PMemory:
    def __init__(self, capacity, path, e = 0.01, a = 0.8, beta = 0.3, beta_increment_per_sampling = 0.0005, gym_name = None):
        self.tree = SumTree(capacity, env=gym_name)
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
        s = np.linspace(0, self.tree.total(), num=n, endpoint=False)+np.random.uniform(high=segment, size=n)

        idx, p, data, nano_id = self.tree.get(s)

        sampling_probabilities = p / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return np.dstack(data), idx, is_weight, nano_id

    def update(self, idx, error, nano_id=None):
        if nano_id is None or self.tree.nano_id[idx-self.capacity+1]==nano_id:
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def save(self):
        h5f = h5py.File(self.file_path, 'w')
        h5f.create_dataset('tree', data=self.tree.tree)
        s,s_n,a,r,d,*_ = np.stack(self.tree.data, axis=1)
        h5f.create_dataset('state', data=np.array(s.tolist()))
        h5f.create_dataset('state_next', data=np.array(s_n.tolist()))
        h5f.create_dataset('action', data=a.astype(np.int))
        h5f.create_dataset('reward', data=r.astype(np.float))
        h5f.create_dataset('done', data=d.astype(np.bool))
        h5f.create_dataset('nano_id', data=self.tree.nano_id)
        h5f.create_dataset('n_entries', data=self.tree.n_entries)
        h5f.create_dataset('write', data=self.tree.write)
        h5f.close()

    def load(self):
        h5f = h5py.File(self.file_path,'r')
        self.tree.tree = h5f['tree'][:]
        s = h5f['state'][:]
        s_n = h5f['state_next'][:]
        a = h5f['action'][:]
        r = h5f['reward'][:]
        d = h5f['done'][:]
        for i,(S,S_N,A,R,D) in enumerate(zip(s,s_n,a,r,d)):
            self.tree.data[i] = [S,S_N,A,R,D] 
        self.tree.nano_id = h5f['nano_id'][:]
        self.tree.n_entries = h5f['n_entries'][()]
        self.tree.write = h5f['write'][()]
        h5f.close()

class SumTree:
    def __init__(self, capacity, tree=None, data=None, nano_id=None, n_entries=None, write=None, env=None):
        e = gym.make(env)
        state_info = (e.observation_space.shape, e.observation_space.dtype)
        action_info = (e.action_space.shape, e.action_space.dtype)
        self.capacity = capacity
        self.tree = Array('i',2 * capacity - 1) if tree is None else tree
        self.data = DataStore(state_info, action_info, capacity) if data is None else data
        self.nano_id = Array(c_char_p, capacity) if nano_id is None else nano_id
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
        self.nano_id[self.write] = generate().encode()
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

        return (idx, self.tree[idx], self.data[dataIdx], self.nano_id[dataIdx].decode())

class DataStore:
    # state_info, action_info, reward_info should contain info about 
    # data shape and type for state, action of the environment
    # example: (state_shape, state_type) = state_info
    def __init__(self, state_info, action_info, capacity):
        self.capacity = capacity
        self.state_info = state_info
        self.action_info = action_info
        # self.reward_info = reward_info
        # initializing shared memory arrays
        self.state_store = self._make_array(self.state_info)
        self.state_next_store = self._make_array(self.state_info)
        self.action_store = self._make_array(self.action_info)
        # self.reward_store = self._make_array(self.reward_info)
        self.reward_store = Array(np.ctypeslib.as_ctypes_type(float), self.capacity)
        self.done_store = Array(np.ctypeslib.as_ctypes_type(bool), self.capacity)

    def _get_flat_shape(self, shape):
        flattened_shape = 1
        for s in shape:
            flattened_shape *= s
        return flattened_shape

    def _make_array(self, info):
        shape, type = info
        flat_shape = self._get_flat_shape(shape)
        ctypes_type = np.ctypeslib.as_ctypes_type(type)
        return Array(ctypes_type, flat_shape*self.capacity)
    
    def _get_np_item(self, arr, key, shape):
        n = self._get_flat_shape(shape)
        item = np.array(arr[key:key+n])
        return np.reshape(item, shape)
    
    def _put_np_item(self, arr, key, value):
        item = value.flatten()
        n = len(item)
        arr[key:key+n] = item

    def __getitem__(self, key):
        if type(key) is slice:
            raise NotImplementedError()
        items = [
            self._get_np_item(self.state_store, key, self.state_info[0]),
            self._get_np_item(self.state_next_store, key, self.state_info[0]),
            self._get_np_item(self.action_store, key, self.action_info[0]),
            # self._get_np_item(self.reward_store, key, self.reward_info[0]),
            self.reward_store[key],
            self.done_store[key],
        ]
        return np.array(items)

    # params: 
    #   - value: a non-object type numpy array to be set
    #   - key: index to store value at 
    def __setitem__(self, value, key):
        if type(key) is slice:
            raise NotImplementedError()
        s,s_n,a,r,d,*_ = value
        self._put_np_item(self.state_store, key, s)
        self._put_np_item(self.state_next_store, key, s_n)
        self._put_np_item(self.action_store, key, a)
        # self._put_np_item(self.reward_store, key, r)
        self.reward_store[key] = r
        self.done_store[key] = d
