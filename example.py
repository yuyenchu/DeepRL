from time import sleep
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ApeX_Manager import Manager

ACTORS = 3
MEM_LENGTH = 1000000
MEM_SAVE_PATH = "./test_folder/testPM.h5"
actor_layers = keras.Sequential(
    [
        layers.Dense(32, activation="relu", name="layer1"),
        layers.Dense(32, activation="relu", name="layer2"),
    ]
)

BASIC_SETTING = {
    "gym_name": "CartPole-v0",
    "save_path":"./test_folder", 
    "verbose":  True,
    "seed":     42,
    "n_step":   4, 
    "gamma":    0.75,
    "middle_layer": actor_layers,
    "DONE_PUNISH":  True,
}
LEARNER_SETTING = {
    "id":   "main",
    "batch_size": 512,
    "update_target_per_batch": 4,
}
ACTOR_SETTING = [
{
    "id": "random",
    "net_update_per_epi":   200,
    "max_buffer_length":    2000,
    "target_reward":        195,
    "max_frame_per_episode":200,
},
{
    "id": "normal",
    "net_update_per_epi":   200,
    "max_buffer_length":    2000,
    "target_reward":        195,
    "epsilon_decay":        1/(10**7),
    "epsilon_min":          0.25,
    "random_act":           10000,
    "max_frame_per_episode":200,
},
{
    "id": "greedy",
    "net_update_per_epi":   200,
    "max_buffer_length":    2000,
    "target_reward":        195,
    "epsilon_decay":        1/(10**5),
    "epsilon_min":          0.1,
    "random_act":           1000,
    "max_frame_per_episode":200,
}
]

manager = Manager(MEM_SAVE_PATH, MEM_LENGTH, ACTORS, BASIC_SETTING, LEARNER_SETTING, ACTOR_SETTING)
manager.start()
# sleep(1800)
# print("time up")
# manager.kill_all_threads()