import gym
from gym.envs.registration import register

max_episode_steps = 5000

register(
    id='RotaryInvertedPendulum-v0',
    entry_point='rotaryenv:RotaryInvertedPendulumEnv',
    max_episode_steps = max_episode_steps
)


env = gym.make('RotaryInvertedPendulum-v0')
obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
env.close()