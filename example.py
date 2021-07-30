from time import sleep
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ApeX_Manager import Manager

ACTORS = 3
MEM_LENGTH = 100000
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
    "seed":     0,
    "n_step":   1, 
    "gamma":    0.99,
    "middle_layer": actor_layers
}
LEARNER_SETTING = {
    "id":   "main",
    "update_target_per_batch": 10,
}
ACTOR_SETTING = {
    "net_update_per_epi":   400,
    "target_reward":        40,
}

manager = Manager(MEM_SAVE_PATH, MEM_LENGTH, ACTORS, BASIC_SETTING, LEARNER_SETTING, ACTOR_SETTING)
manager.start()
sleep(1800)
print("time up")
manager.kill_all_threads()