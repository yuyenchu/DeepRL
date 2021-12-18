import numpy as np
import gym
from numpy import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from PMemory import PMemory
from DQNagent import DQNagent

class D3QNgym:
    ### to be implemented ###
    def __init__(self, GAME, GAME_VER, save_path, save_directory, max_steps_per_episode, update_after_actions, max_frame, saving_check_point, reward_target, capacity=1000000):
        # initialize variables
        self.save_path = save_path
        #!!!!!!!!!!!!!!!!
        self.memory = PMemory(capacity, self.save_path)  # watch out for memory size

        # Configuration paramaters for the whole setup
        self.seed = 42
        self.gamma = 0.75  # Discount factor for past rewards
        self.epsilon_min = 0.075  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.epsilon_interval = (self.epsilon_max - self.epsilon_min)  # Rate at which to reduce chance of random action being taken
        self.batch_size = 32  # Size of batch taken from replay buffer
        self.max_steps_per_episode = max_steps_per_episode

        # Use the Baseline Atari environment because of Deepmind helper functions
        self.env = gym.make(f'{GAME}-v{GAME_VER}')
        # Warp the frames, grey scale, stake four frame and scale to smaller ratio
        # env = wrap_deepmind(env, frame_stack=True, scale=True)
        self.env.seed(self.seed)
        # num_actions = sum(env.action_space.shape)
        self.num_actions = self.env.action_space.n
        self.out_shape = self.env.action_space.shape
        self.in_shape = self.env.observation_space.shape

        self.agent = DQNagent(self.memory, save_directory, in_shape=self.in_shape, num_actions=self.num_actions)  # require maual setup DQNagent config in DQNagent.py

        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 10000
        # Number of frames for exploration
        self.epsilon_greedy_frames = 20000
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        self.max_memory_length = 70000
        # Maximum runnung reward length
        self.max_reward_length = 100
        # Train the model after 4 actions
        self.update_after_actions = update_after_actions
        # How often to update the target network
        self.update_target_network = 4000
        # Using huber loss for stability
        self.loss_function = keras.losses.Huber()
        # steps for multi-step return
        self.n_step = 4
        self.n_step = max(1,self.n_step) # default=1 (normal loass)
        # Max frame per training loop
        self.max_frame = max_frame
        # Saving model checkpoints
        self.save_after_episode = saving_check_point
        # progress bar text
        self.label_template = " reward = {:.4f} loss = {:.4f}"
        # running reawrd considered solved
        self.reward_target = reward_target
    
    def setup(self):
        # setup everything needed to start run

        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        # Experience replay buffers
        self.action_history = []
        self.state_history = []
        # state_next_history = []
        # rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.state_n_next_history = []
        self.returns_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.frame_count = 0
        self.max_loop = self.frame_count//self.max_frame
        
        self.epsilon = 1.0  # Epsilon greedy parameter
    
    def run(self):
        self.setup()
        while self.frame_count//self.max_frame <= self.max_loop:
            state = np.array(self.env.reset())
            episode_reward = 0
            episode_loss = 0

            for timestep in range(1, self.max_steps_per_episode+1):
                # render the environment observation (doesn't work for jupyter notebook)
                # env.render()

                self.frame_count += 1

                # Q model forwrd prorogation to produce discrete action
                if np.random.uniform() <= self.epsilon:
                    action = self.agent.forward(state, True)
                else:
                    action = self.agent.forward(state, False)

                # Decay probability of taking random action
                self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.epsilon_min)

                # Apply the action in environment
                state_next, reward, done, _ = self.env.step(action)

                self.state_history.append(state)
                self.state_n_next_history.append(state_next)
                self.action_history.append(action)
                self.returns_history.append(reward)
                self.done_history.append(done)

                episode_reward += reward
                state = np.array(state_next)

                # Train every fourth frame and once batch size is over 32
                # Train every fourth frame and once batch size is over 32
                if self.frame_count % self.update_after_actions == 0 and len(self.memory) > self.batch_size:
                    state_sample, state_next_sample, action_sample, rewards_sample, done_sample, _, isWeight, _ = self.agent.sample_replay(self.batch_size)
                    updated_q_values = self.agent.target_q(state_next_sample, rewards_sample, done_sample)
                    episode_loss += self.agent.train(state_sample, action_sample, updated_q_values)

                if self.frame_count % self.update_target_network == 0:
                    # update the the target network with new weights
                    self.agent.set_weights(self.agent.get_weights()[0])
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, self.episode_count, self.frame_count))

                if done:
                    print('\tDone at',timestep,'frames / total',self.frame_count,'frames')
                    break
                        
            self.agent.save_memory(self.state_history, self.state_n_next_history,self.action_history,self.returns_history,self.done_history)
            
            print(sum(self.returns_history))
            print(episode_loss)

            self.state_history=[] 
            self.state_n_next_history=[]
            self.action_history=[]
            self.returns_history=[]
            self.done_history=[]

            # Update running reward to check condition for solving
            self.episode_reward_history.append(episode_reward)
            if len(self.episode_reward_history) > self.max_reward_length:
                del self.episode_reward_history[:1]
        
            running_reward = np.mean(self.episode_reward_history)
            self.episode_count += 1

            # Condition considered task solved
            if running_reward > self.reward_target:  
                print("Solved at episode {}!".format(self.episode_count))
                break
                # run and train agent to fit env

    ### implement after run is stable ###
    def save(self):
        # save necessary variables for rebuilding
        pass
    
    def load(self):
        # load and rebuild class and variables
        pass

    def render(self):
        # render/output an episode visualization
        pass