import gym
from gym.envs.registration import register

import pandas as pd
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

max_episode_steps = 5000

train_mode = False      ## True = Training rotary inverted pendulum with PPO | False = Evaluation

save_file = "ppo_pendulum"
save_file = save_file + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

eval_file = "ppo_pendulum_dr"

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []
        self.scores = []

    def _on_step(self):
        info = self.locals
        self.rewards.append(info['rewards'])
        if self.locals['infos'][0]['TimeLimit.truncated']:
            self.scores.append(sum(self.rewards))
            self.rewards = []

        return True

    def _on_training_end(self):
        scores_df = pd.DataFrame(self.scores)
        scores_df.to_excel("scores.xlsx", index=False)
        print("Completed!")


register(
    id='RotaryInvertedPendulum-v0',
    entry_point='rotaryenv:RotaryInvertedPendulumEnv',
    max_episode_steps=max_episode_steps
)


if train_mode:
    env = make_vec_env("RotaryInvertedPendulum-v0", n_envs=10)
    model = PPO("MlpPolicy", env,
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_steps=2048, verbose=1, tensorboard_log="./tsboard/")
    
    model.learn(total_timesteps=1500000)
    model.save("./save/{save_file}".format(save_file=save_file))
else:
    env = gym.make("RotaryInvertedPendulum-v0")
    model = PPO.load("./save/{eval_file}".format(eval_file=eval_file))

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
