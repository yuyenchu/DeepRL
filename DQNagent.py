import json
import numpy as np
from numpy import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
# from numba import jit

class DQNagent:
    def __init__(self, memory, save_path, in_shape, num_actions,\
                DUELING=True, DOUBLE=True, DONE_PUNISH=False, n_step=1,\
                gamma=0.99, middle_layer=None, verbose=False):
        # setting path for saving 
        self.save_path = save_path
        # DQN type 
        self.DUELING = DUELING
        self.DOUBLE = DOUBLE
        self.DONE_PUNISH = DONE_PUNISH
        # agent config
        self.num_actions = num_actions
        self.in_shape = in_shape
        self.n_step = n_step
        self.gamma = gamma
        self.verbose = verbose
        # saving / loading config
        self.variables_to_save = [
            "DUELING",
            "DOUBLE",
            "DONE_PUNISH",
            "n_step",
            "num_actions",
            "gamma",
            "verbose"
        ]
        # main components of agent
        self.memory = memory
        '''
        required functions for memory
            add(sample): store a batch of sample to memory, return nothing
            sample(n): return n number of memory in format (sample, sample_index, sample_weight)
        '''
        self.loss_function = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam()
        # middle layer MUST use sequential model (NOT functional api)
        self.model = self.build_model(middle_layer)
        self.model_target = self.build_model(middle_layer)
        # ensuring save_path is a directory
        if not self.save_path.endswith("/"):
            self.save_path += "/"

    # building model for agent
    def build_model(self, middle_layer=None):
        inputs = layers.Input(shape=self.in_shape)
        if middle_layer is None:
            # define your custom model here if not provide middle_layer
            ###################################
            layer1 = layers.Dense(32, activation="relu")(inputs)
            layer2 = layers.Dense(32, activation="relu")(layer1)
            action = layers.Dense(self.num_actions, activation="linear")(layer1)
            ###################################
        else:
            layer = middle_layer(inputs)
            action = layers.Dense(self.num_actions, activation="linear")(layer)

        if self.DUELING:
            if self.verbose:
                print("activated Dueling DQN")
            # v predict state value for dueling DQN
            v = layers.Dense(1, activation="linear")(layer2)
            out = v + (action - tf.reduce_mean(action, axis=1, keepdims=True))
            return keras.Model(inputs=inputs, outputs=out)
            
        return keras.Model(inputs=inputs, outputs=action)
    
    # helper for saving / loading class variables
    def save_variables(self):
        temp = {}
        for i in self.variables_to_save:
            exec("temp['%s'] = self.%s" % (i,i))
        with open(f'{self.save_path}agent_config.json', 'w') as f:
            json.dump(temp, f, cls=NpEncoder)
            if self.verbose:
                print(temp)

    def load_variables(self):
        with open(f'{self.save_path}agent_config.json') as f:
            data = json.load(f)
            temp = [s[:s.find("[")] for s in self.variables_to_save]
            exec_str = ""
            for i in self.variables_to_save:
                exec_str += ("self.%s=%s\n"%(i,str(data[i])))
                if self.verbose:
                    exec_str += ("print('%s =' ,self.%s)\n"%(i,i))
            exec(exec_str)

    # saving / loading necessary data for rebuilding
    def save(self):
        if self.verbose:
            print("saving at", self.save_path)
        self.model.save(f'{self.save_path}model')
        self.model_target.save(f'{self.save_path}target')
        self.save_variables()

    def load(self):
        if self.verbose:
            print("loading from", self.save_path)
        self.model = load_model(f'{self.save_path}model')
        self.model_target = load_model(f'{self.save_path}target')
        self.load_variables()

    def set_weights(self, w1, w2=None):
        self.model.set_weights(w1)
        if w2 is not None:
            self.model_target.set_weights(w2)
        else:
            self.model_target.set_weights(w1)

    def get_weights(self):
        return self.model.get_weights(), self.model_target.get_weights()

    # process and return memory sampling of given sive
    def sample_replay(self, batch_size):
        sample, idx, is_weight = self.memory.sample(batch_size)
        state_sample, state_next_sample, action_sample, rewards_sample, done_sample = sample[0]
        
        return np.stack(state_sample), np.stack(state_next_sample), action_sample.astype(np.int32), rewards_sample.astype(np.float64), np.stack(done_sample), idx, np.array([is_weight])

    # process and save the given data into memory
    def save_memory(self, state, state_next, action, returns, done):
        masks = tf.one_hot(action, self.num_actions)
        q_values = self.model(np.array(state))
        q_val = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        target_q_val = self.target_q(np.array(state_next), returns, np.array(done))
        priority = q_val - target_q_val

        for p,s,s_n,a,r,d in zip(priority, state, state_next, action, returns, done):
            self.memory.add([s,s_n,a,r,d], p)

    # forward propagation of given state with model (random action if random_act set to True)
    def forward(self, state, random_act=False):
        # Use epsilon-greedy for exploration
        if random_act:
            # Take random action
            action = np.random.choice(self.num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        return action

    # train model with given sample and target(updated_q_values)
    def train(self, state_sample, action_sample, updated_q_values, sample_weight=None):
        # create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.num_actions)

        with tf.GradientTape() as tape:
            # train the model on the states and updated Q-values
            q_values = self.model(state_sample)

            # apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action, sample_weight=sample_weight)

            # backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    # compute the target q values of given sample
    def target_q(self, state_next_sample, rewards_sample, done_sample):
        if self.DOUBLE:
            future_rewards = self.model_target(state_next_sample).numpy()
            future_actions = np.argmax(self.model(state_next_sample), axis=1)
            future_rewards = future_rewards[range(len(done_sample)),future_actions]
        else:
            future_rewards = np.argmax(self.model_target(state_next_sample), axis=1)

        updated_q_values = rewards_sample + self.gamma**self.n_step * np.nan_to_num(future_rewards)
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample*self.DONE_PUNISH

        return updated_q_values

# helper class for classifying and casting variable type for json format
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)