import time
import gym
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('./test_folder/CartPole')
env = gym.make('CartPole-v0')
state = env.reset()
done = False

while not done:
    print(state)
    action = model.predict(np.array([state]))
    state_next, reward, done, _ = env.step(np.argmax(action))
    state = state_next
    env.render()
    # time.sleep(0.05)

time.sleep(3)