#DQNAgent

import json
import numpy as np
from numpy import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from PMemory import PMemory
from Memory import Memory

class DQNagent:
    def __init__(self, memory, save_path):
        # setting path for saving 
        self.save_path = save_path
        # setting type of DQN
        self.DUELING = True
        self.DOUBLE = True

        self.DONE_PUNISH = False
        self.n_step = 4
        self.num_actions = 10
        self.gamma = 0.75

        self.variables_to_save = [
            "n_step"
        ]

        self.memory = memory
        self.model = self.build_model()
        self.model_target = self.build_model()

    def build_model(self):
        
        #inputs = layers.Input(shape=in_shape)
        inputs = layers.Input(shape=10)

        layer1 = layers.Dense(32, activation="relu")(inputs)
        layer2 = layers.Dense(32, activation="relu")(layer1)
        #action = layers.Dense(num_actions, activation="linear")(layer1)
        action = layers.Dense(self.num_actions, activation="linear")(layer1)

        if self.DUELING:
            print("activated Dueling DQN")

            # v predict state value for dueling DQN
            v = layers.Dense(1, activation="linear")(layer2)
            out = v + (action - tf.reduce_mean(action, axis=1, keepdims=True))
            return keras.Model(inputs=inputs, outputs=out)
            
        return keras.Model(inputs=inputs, outputs=action)
        
    def save_variables(self):
        temp = {}
        for i in self.variables_to_save:
            exec("temp['%s'] = self.%s" % (i,i))
        with open(f'{save_path}config.json', 'w') as f:
            json.dump(temp, f, cls=NpEncoder)

    def load_variables(self):
        with open(f'{save_path}config.json') as f:
            data = json.load(f)
            temp = [s[:s.find("[")] for s in self.variables_to_save]
            for i in self.variables_to_save:
                exec_str += ("self.%s=%s\nprint('%s =' ,%s)\n"%(i,str(data[i]),i,i))
            exec(exec_str)

    def save(self):
        self.model.save(f'model')
        self.model_target.save(f'target')
        self.save_variables()

    def load(self):
        self.model = load_model(f'model')
        self.model_target = load_model(f'target')
        self.load_variables()

    def sample_replay(self, batch_size):
        sample, idx, is_weight = self.memory.sample(batch_size)
        state_sample, state_next_sample, action_sample, rewards_sample, done_sample = sample[0]
        
        return np.stack(state_sample), np.stack(state_next_sample), action_sample.astype(np.int32), rewards_sample.astype(np.float64), np.stack(done_sample), idx, is_weight

    def save_memory(self, state, state_next, action, returns, done):
        masks = tf.one_hot(action, self.num_actions)
        q_values = self.model(np.array(state))
        q_val = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        target_q_val = self.target_q(np.array(state_next), returns, np.array(done))
        priority = q_val - target_q_val

        for p,s,s_n,a,r,d in zip(priority, state,state_next, action, returns,done):
            memory.add([s,s_n,a,r,d], p)

    def forward(self, state, random_act=False):
        # Use epsilon-greedy for exploration
        if random_act:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        return action

    def train(self, state_sample, action_sample, updated_q_values):
        self.loss_function = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam()
        
        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.num_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def target_q(self, state_next_sample, rewards_sample, done_sample):
        """
        state_next_sample: np.ndarray
        rewards_sample: array
        done_sample: bool array

        Sampled data of different actors
        """
        if self.DOUBLE:
            future_actions = np.argmax(self.model(state_next_sample), axis=1)
            future_rewards = self.model_target(state_next_sample).numpy()
            # print(future_rewards)
            # arr_len_done = [int(i) for i in range(len(done_sample))]
            # print(type(arr_len_done))
            # future_actions = np.array(future_actions)

            future_rewards = future_rewards[range(len(done_sample)),future_actions]
        else:
            future_rewards = tf.reduce_max(self.model_target(state_next_sample), axis=1)

        updated_q_values = rewards_sample + self.gamma**self.n_step * np.nan_to_num(future_rewards)

        updated_q_values = updated_q_values * (1 - done_sample) - done_sample*self.DONE_PUNISH

        return updated_q_values
    
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