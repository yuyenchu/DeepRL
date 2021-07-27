# -*- coding: utf-8 -*-
from google.colab import drive

drive.mount('/content/gdrive')

# !pip install atari-py
# !python -m atari_py.import_roms gdrive/MyDrive/ml_data/ROMS

"""# Train Configurations

###Import and Path settings
"""

# Commented out IPython magic to ensure Python compatibility.
import gym
import json
import numpy as np
from numpy import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from matplotlib import animation, rc, pyplot as plt
from ipywidgets import IntProgress, HBox, Label
from IPython.display import display, HTML
from time import sleep

# %matplotlib inline
# DQN setting
DUELING = True
DOUBLE = True
PRIORITIZED_MEMORY = True
DONE_PUNISH = False
# gym setting
GAME = "CartPole"
GAME_VER = 0
# save/load setting
LOAD = False
SAVE_CKPT = False
VERSION = "2.0a"
save_path = f"/content/gdrive/MyDrive/ml_data/{GAME}/"

"""###Non constant variables (will be changed during training)"""

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
# state_next_history = []
# rewards_history = []
done_history = []
episode_reward_history = []
state_n_next_history = []
returns_history = []
running_reward = 0
episode_count = 0
frame_count = 0

epsilon = 1.0  # Epsilon greedy parameter

"""###Constant Variables (stays same in training if not changed manually)"""

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.75  # Discount factor for past rewards
epsilon_min = 0.075  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 200

# Use the Baseline Atari environment because of Deepmind helper functions
env = gym.make(f'{GAME}-v{GAME_VER}')
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
# env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)
# num_actions = sum(env.action_space.shape)
num_actions = env.action_space.n
out_shape = env.action_space.shape
in_shape = env.observation_space.shape
# Number of frames to take random action and observe output
epsilon_random_frames = 10000
# Number of frames for exploration
epsilon_greedy_frames = 20000
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 70000
# Maximum runnung reward length
max_reward_length = 100
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 4000
# Using huber loss for stability
loss_function = keras.losses.Huber()
# steps for multi-step return
n_step = 4
n_step = max(1,n_step) # default=1 (normal loass)
# Max frame per training loop
max_frame = 60000
# Saving model checkpoints
save_after_episode = 10
# progress bar text
label_template = " reward = {:.4f} loss = {:.4f}"
# running reawrd considered solved
reward_target = 195

if PRIORITIZED_MEMORY:
    memory = PMemory(max_memory_length, f'{save_path}memory_v{VERSION}.h5')

"""###Helper functions"""

variables_to_save = ["epsilon","frame_count","episode_count",
                     "running_reward","episode_reward_history",
                     f"action_history[-{batch_size}:]",
                     f"state_history[-{batch_size}:]", 
                    #  f"state_next_history[-{batch_size}:]",
                    #  f"rewards_history[-{batch_size}:]", 
                     f"done_history[-{batch_size}:]",
                     f"state_n_next_history[-{batch_size}:]",
                     f"returns_history[-{batch_size}:]"]



def save_variables():
    temp = {}
    for i in variables_to_save:
        exec("temp['%s'] = %s" % (i,i))
    with open(f'{save_path}variables_v{VERSION}.json', 'w') as f:
        json.dump(temp, f, cls=NpEncoder)

def load_variables():
    with open(f'{save_path}variables_v{VERSION}.json') as f:
        data = json.load(f)
        temp = [s[:s.find("[")] for s in variables_to_save]
        exec_str = "global "+", ".join(temp)+"\n"
        for i in variables_to_save:
            exec_str += ("%s=%s\nprint('%s =' ,%s)\n"%(i,str(data[i]),i,i))
        exec(exec_str)

def create_q_model(dueling=False):
    
    inputs = layers.Input(shape=in_shape)

    # Convolutions on the frames on the screen
    # layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    # layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    # layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    # layer4 = layers.Flatten()(layer3)

    # layer5 = layers.Dense(512, activation="relu")(layer4)
    layer1 = layers.Dense(32, activation="relu")(inputs)
    layer2 = layers.Dense(32, activation="relu")(layer1)
    action = layers.Dense(num_actions, activation="linear")(layer1)

    if dueling:
        print("activated Dueling DQN")

        # v predict state value for dueling DQN
        v = layers.Dense(1, activation="linear")(layer2)
        out = v + (action - tf.reduce_mean(action, axis=1, keepdims=True))
        return keras.Model(inputs=inputs, outputs=out)
        
    return keras.Model(inputs=inputs, outputs=action)

def target_q(state_next_sample, rewards_sample, done_sample):
    # state_sample = np.array([state_history[i] for i in indices])
    # action_sample = [action_history[i] for i in indices]
    # done_sample = tf.convert_to_tensor(
    #     [float(done_history[i]) for i in indices]
    # )
    # if MULTI_STEP_RETURN:
        # state_next_sample = np.array([state_n_next_history[i] for i in indices])
        # rewards_sample = [returns_history[i] for i in indices]
    # else:
    #     state_next_sample = np.array([state_next_history[i] for i in indices])
    #     rewards_sample = [rewards_history[i] for i in indices]
        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
    l = len(state_next_sample)
    if DOUBLE:
        future_actions = np.argmax(model.predict(state_next_sample), axis=1)
        future_rewards = model_target.predict(state_next_sample)
        future_rewards = future_rewards[range(l),future_actions]
    else:
        future_rewards = tf.reduce_max(model_target.predict(state_next_sample), axis=1)

    # if MULTI_STEP_RETURN:
    # MULTI_STEP_RETURN = cumulated return + discount factor ** n * expected future reward
    updated_q_values = rewards_sample + gamma**n_step * np.nan_to_num(future_rewards)
    # else:
    #     # Q value = reward + discount factor * expected future reward
    #     updated_q_values = rewards_sample + gamma * future_rewards

    # If final frame set the last value to -1
    updated_q_values = updated_q_values * (1 - done_sample) - done_sample*DONE_PUNISH

    return updated_q_values

def sample_replay():
    # Get indices of samples for replay buffers
    indices = np.random.choice(range(len(done_history)), size=batch_size)

    # Using list comprehension to sample from replay buffer
    state_sample = np.array([state_history[i] for i in indices])
    state_next_sample = np.array([state_n_next_history[i] for i in indices])
    action_sample = [action_history[i] for i in indices]
    rewards_sample = [returns_history[i] for i in indices]
    done_sample = tf.convert_to_tensor(
        [float(done_history[i]) for i in indices]
    )

    return state_sample, state_next_sample, action_sample, rewards_sample, done_sample

def sample_p_replay():
    sample, idx, is_weight = memory.sample(batch_size)
    # print(sample)
    state_sample, state_next_sample, action_sample, rewards_sample, done_sample = sample[0]
    
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

"""###Preparing model"""

if LOAD:
    model = load_model(f'{save_path}{GAME}_v{VERSION}')
    model_target = load_model(f'{save_path}{GAME}_target_v{VERSION}')
    load_variables()
else:
    # The first model makes the predictions for Q-values which are used to
    # make a action.
    model = create_q_model(DUELING)

    # Build a target model for the prediction of future rewards.
    # The weights of a target model get updated every 10000 steps thus when the
    # loss between the Q-values is calculated the target Q-value is stable.
    model_target = create_q_model(DUELING)

ckpt = tf.train.Checkpoint(model)
manager = tf.train.CheckpointManager(ckpt, directory=f'{save_path}checkpoint', checkpoint_name=f'{GAME}_v{VERSION}.pt', max_to_keep=5)

target_ckpt = tf.train.Checkpoint(model_target)
target_manager = tf.train.CheckpointManager(target_ckpt, directory=f'{save_path}target_checkpoint', checkpoint_name=f'{GAME}_target_v{VERSION}.pt', max_to_keep=5)

"""# Train Loop"""

MAX_EPISODE_SHOWN = 30
inf_loop = False
max_loop = frame_count//max_frame
widgets = []

while frame_count//max_frame <= max_loop or inf_loop:  # Run until solved or reached max frames
    state = np.array(env.reset())
    episode_reward = 0
    episode_loss = 0

    # widgets for jupyter notebook
    p = IntProgress(max = max_steps_per_episode)
    ep_l = Label(value='Episode '+str(episode_count))
    l = Label(value=label_template.format(0,0))
    b = HBox([ep_l,p,l])
    widgets.append(b)
    display(b)
    # close widgets if too many has been displayed
    if len(widgets) > MAX_EPISODE_SHOWN:
        widgets[0].close()
        del widgets[:1]
    
    for timestep in range(1, max_steps_per_episode+1):
        # render the environment observation (doesn't work for jupyter notebook)
        # env.render()

        frame_count += 1
        p.value = timestep

        # Q model forwrd prorogation to produce discrete action
        action = forward(state, frame_count)

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the action in environment
        state_next, reward, done, _ = env.step(action)

        save_memory(state, state_next, action, reward, done)
        episode_reward += reward
        state = np.array(state_next)

        # Train every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and ((not PRIORITIZED_MEMORY and len(done_history) > batch_size) or (PRIORITIZED_MEMORY and len(memory) > batch_size)):
            state_sample, state_next_sample, action_sample, rewards_sample, done_sample = sample_p_replay() if PRIORITIZED_MEMORY else sample_replay()
            updated_q_values = target_q(state_next_sample, rewards_sample, done_sample)
            episode_loss += train(state_sample, action_sample, updated_q_values)

        l.value = label_template.format(episode_reward,episode_loss)

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

        if done:
            print('\tDone at',timestep,'frames / total',frame_count,'frames')
            break

    if PRIORITIZED_MEMORY:
        save_p_memory(state_history,state_n_next_history,action_history,returns_history,done_history)
    
    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > max_reward_length:
        del episode_reward_history[:1]
    
    running_reward = np.mean(episode_reward_history)
    episode_count += 1

    if episode_count % save_after_episode == 0 and SAVE_CKPT:
        print("model saved at",manager.save())
        print("target model saved at",target_manager.save())
        save_variables()

    # Condition considered task solved
    if running_reward > reward_target:  
        print("Solved at episode {}!".format(episode_count))
        break

"""# Save Model"""

model.save(f'{save_path}{GAME}_v{VERSION}')
model_target.save(f'{save_path}{GAME}_target_v{VERSION}')
save_variables()

################################################################
class D3QNgym:
    ### to be implemented ###
    def __init__(self, save_path, game, ...runtime_parameters):
        # initialize variables
        self.save_path = save_path
        self.env = gym.make(game)   # some parameters change based on game
        self.memory = PMemory(...)  # watch out for memory size
        self.agent = DQNagent(...)  # require maual setup DQNagent config in DQNagent.py
        self.runtime_parameters =...# ex: epsilon, frame_count, running_reward
        self.buffer =...            # memory buffer for running each episode
    
    def setup(self):
        # setup everything needed to start run
    
    def run(self, ...run_options):
        self.setup()
        while ...condition:
            pass
        # run and train agent to fit env

    ### implement after run is stable ###
    def save(self):
        # save necessary variables for rebuilding
    
    def load(self):
        # load and rebuild class and variables

    def render(self, ...render_options):
        # render/output an episode visualization
