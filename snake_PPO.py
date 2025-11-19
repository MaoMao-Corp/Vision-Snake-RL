from stable_baselines3 import PPO
import snake_logic as snake
import DynamicCNN as dyn

env = snake.SnakeEnv(size=5, noise=snake.NoiseModel())

policy_kwargs = dict(
    features_extractor_class=dyn.SnakePPONet,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    verbose=1,
)

model.learn(total_timesteps=100_000)
model.save("snake_ppo")
