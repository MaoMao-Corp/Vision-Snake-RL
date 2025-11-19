from stable_baselines3 import PPO
from snake_logic import SnakeEnv
import time
import os
import matplotlib.pyplot as plt

# Create the environment
env = SnakeEnv(size=5)

# Load the trained model
model = PPO.load("snake_ppo", env=env)

import matplotlib.pyplot as plt

obs = env.reset()[0]
done = False

while not done:
    state = env.game._get_state()
    # render as RGB-like image
    mat = obs.transpose(1, 2, 0)  # CHW → HWC
    plt.imshow(mat)
    plt.axis("off")
    plt.pause(0.2)
    plt.clf()
    
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

plt.show()
