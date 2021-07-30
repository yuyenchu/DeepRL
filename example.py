from time import sleep

from ApeX_Manager import Manager

ACTORS = 20
MEM_LENGTH = 100000
MEM_SAVE_PATH = "./test_folder/testPM.h5"
BASIC_SETTING = {
    "gym_name": "CartPole-v0",
    "save_path":"./test_folder", 
    "verbose":  True,
    "n_step":   1, 
    "gamma":    0.99
}
LEARNER_SETTING = {
    "id":   "main",
}
ACTOR_SETTING = {
    "net_update_per_epi": 400,
}

manager = Manager(MEM_SAVE_PATH, MEM_LENGTH, ACTORS, BASIC_SETTING, LEARNER_SETTING, ACTOR_SETTING)
manager.start()
sleep(20)
manager.kill_all_threads()