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
    def __init__(self, save_path):
        # setting path for saving 
        self.save_path = save_path
        # setting type of DQN
        self.DUELING = True
        self.DOUBLE = True
        self.PRIORITIZED_MEMORY = True

        
        self.DONE_PUNISH = False

        if PRIORITIZED_MEMORY:
            self.memory = PMemory()
        else:
            self.memory = Memory()
        
    def save_variables(self):
        temp = {}
        for i in variables_to_save:
            exec("temp['%s'] = self.%s" % (i,i))
        with open(f'{save_path}variables_v{VERSION}.json', 'w') as f:
            json.dump(temp, f, cls=NpEncoder)

    def load_variables(self):
        with open(f'{save_path}variables_v{VERSION}.json') as f:
            data = json.load(f)
            temp = [s[:s.find("[")] for s in variables_to_save]
            for i in variables_to_save:
                exec_str += ("self.%s=%s\nprint('%s =' ,%s)\n"%(i,str(data[i]),i,i))
            exec(exec_str)

    def save(self):
        self.model.save(f'{selfsave_path}{GAME}_v{VERSION}')
        self.model_target.save(f'{selfsave_path}{GAME}_target_v{VERSION}')
        self.save_variables()
    
    def target_q(state_next_sample, rewards_sample, done_sample):
        if DOUBLE:
            future_actions = np.argmax(self.model.predict(state_next_sample), axis=1)
            future_rewards = model_target.predict(state_next_sample)
            future_rewards = future_rewards[range(l),future_actions]
        else:
            future_rewards = tf.reduce_max(self.model_target.predict(state_next_sample), axis=1)

        updated_q_values = rewards_sample + gamma**n_step * np.nan_to_num(future_rewards)

        updated_q_values = updated_q_values * (1 - done_sample) - done_sample*DONE_PUNISH

        return updated_q_values

    def sample_replay():
        if self.PRIORITIZED_MEMORY:
            sample, idx, is_weight = self.memory.sample(batch_size)
            state_sample, state_next_sample, action_sample, rewards_sample, done_sample = sample[0]
        else:
            state_sample, state_next_sample, action_sample, rewards_sample, done_sample = self.memory.sample(batch_size)
            idx, is_weight = [], []
        
        return np.stack(state_sample), np.stack(state_next_sample), action_sample.astype(np.int32), rewards_sample.astype(np.float64), np.stack(done_sample)

    def save_memory(state, state_next, action, reward, done):
        # Save actions and states in replay buffer
        state_history.append(state)
        action_history.append(action)
        done_history.append(done)
        
        # if MULTI_STEP_RETURN:
        if timestep > 1:
            for n in range(1, min(n_step,timestep)):
                returns_history[-n] += (gamma**n)*reward
        if timestep > n_step:
            state_n_next_history[-n_step] = state_next
        state_n_next_history.append(np.full(in_shape, np.nan))
        returns_history.append(reward)
        # else:
        #     state_next_history.append(state_next)
        #     rewards_history.append(reward)

        # Limit the state and reward history
        if len(state_history) > max_memory_length:
            del state_history[:1]
            # del state_next_history[:1]
            # del rewards_history[:1]
            del action_history[:1]
            del done_history[:1]
            del state_n_next_history[:1]
            del returns_history[:1]

    def save_p_memory(state_history,state_n_next_history,action_history,returns_history,done_history):
        masks = tf.one_hot(action_history, num_actions)
        q_values = model(np.array(state_history))
        q_val = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        target_q_val = target_q(np.array(state_n_next_history), returns_history, np.array(done_history))
        priority = q_val - target_q_val

        for p,s,s_n,a,r,d in zip(priority,state_history,state_n_next_history,action_history,returns_history,done_history):
            memory.add(p,[s,s_n,a,r,d])

        state_history = []
        action_history = []
        done_history = []
        state_n_next_history = []
        returns_history = []

    def forward(state, frame_count):
        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        return action

    def train(state_sample, action_sample, updated_q_values):
        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, num_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss
    
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