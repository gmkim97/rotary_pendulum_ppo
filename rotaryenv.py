import gym
from gym import utils
from gym import spaces
import mujoco_py
import numpy as np
import math
import warnings
import time

warnings.filterwarnings("ignore")

class RotaryInvertedPendulumEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': [
        'human',
        'rgb_array',
        'depth_array',
        ],
        'render_fps': 500,}

    def __init__(self):
        super(RotaryInvertedPendulumEnv, self).__init__()

        self._max_velocity_joint0 = 22.0
        self._max_velocity_joint1 = 50.0

        # Load MJCF model
        model_path = "./assets/rotary_pen.xml"
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)

        # Domain Randomization : Randomize the physical properties of Pendulum
        self.randomize_physical_properties()

        # Initialize renderer
        self.viewer = None

        # Define action & observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float64,
        )

        # Distribution of random initial states
        self._init_pos_high = math.pi
        self._init_pos_low = -math.pi
        self._init_vel_high = 3.0
        self._init_vel_low = -3.0

        # Reward function parameters
        theta1_weight = 0.0
        theta2_weight = 10.0
        dtheta1_weight = 1.0
        dtheta2_weight = 3.0

        self._desired_obs_values = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        self._obs_weights = [
            theta1_weight,
            theta1_weight,
            theta2_weight,
            theta2_weight,
            dtheta1_weight,
            dtheta2_weight,
        ]
        self._action_weight = 0.0

        

    def reset(self):
        # Random reset
        self.sim.reset()
        self.randomize_state()
        return self._get_obs()


    def step(self, action):
        self.sim.data.ctrl[:] = action
        self.sim.step()
        self.bound_velocities()
        obs = self._get_obs()
        reward = self.calculate_reward(obs, action)

        # Terminate if simulation become unstable
        notdone = np.isfinite(obs).all()
        done = not notdone

        return obs, reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.render()
        time.sleep(0.01)        ## Adjusting speed of visualization


    def close(self):
        if self.viewer is not None:
            self.viewer = None
    

    def randomize_physical_properties(self):     ## Physical parameter randomization
        pendulum_mass = np.random.uniform(low=0.008, high=0.010)    ## default: 0.009801
        pendulum_length = 0.0645   ## default: 0.0645
        arm1_mass = np.random.uniform(low=0.030, high=0.050)    ## default: 0.040466

        self.sim.model.body_mass[self.sim.model.body_name2id('arm1')] = arm1_mass
        self.sim.model.body_mass[self.sim.model.body_name2id('arm2')] = pendulum_mass
        self.sim.model.geom_size[self.sim.model.geom_name2id('arm2_geom')] = [0.005, pendulum_length, 0]
        self.sim.model.geom_pos[self.sim.model.geom_name2id('arm2_geom')] = [0.0, 0.0, pendulum_length]

        print(f"arm1_mass: {arm1_mass} | pendulum_mass: {pendulum_mass} | pendulum_length: {pendulum_length}")


    def randomize_state(self):      ## State randomization
        sim_state = self.sim.get_state()

        init_qpos = sim_state.qpos.ravel().copy()
        init_qvel = sim_state.qvel.ravel().copy()

        sim_state.qpos[:] = init_qpos + np.random.uniform(
            size=self.model.nq, low=self._init_pos_low, high=self._init_pos_high
        )
        sim_state.qpos[:][1] -= math.pi
        sim_state.qvel[:] = init_qvel + np.random.uniform(
            size=self.model.nv, low=self._init_vel_low, high=self._init_vel_high
        )


        self.sim.set_state(sim_state)
        self.sim.forward()


    def _get_obs(self):
        # sin and cos instead of limit to get rid of discontinuities
        # scale angular velocities so that they won't dominate
        obs = np.array(
            [
                np.sin(self.sim.data.qpos[0]),
                np.cos(self.sim.data.qpos[0]),
                np.sin(self.sim.data.qpos[1]),
                np.cos(self.sim.data.qpos[1]),
                self.sim.data.qvel[0] / self._max_velocity_joint0,
                self.sim.data.qvel[1] / self._max_velocity_joint1,
            ]
        )
        return obs
    

    def calculate_reward(self, obs: np.array, a: np.array):
        observation_reward = np.sum(
            [
                -weight * np.power((desired_value - observation_value), 2)
                for (observation_value, desired_value, weight) in zip(
                    obs, self._desired_obs_values, self._obs_weights
                )
            ]
        )

        action_reward = -self._action_weight * np.power(a[0], 2)

        return observation_reward + action_reward


    # def _is_done(self, obs):
    #     return np.abs(obs[1]) > np.pi / 2


    def bound_velocities(self):
        # Bound velocities to set ranges - it isn't possible to define it in xml model
        self.sim.data.qvel[0] = np.clip(
            self.sim.data.qvel[0], -self._max_velocity_joint0, self._max_velocity_joint0
        )
        self.sim.data.qvel[1] = np.clip(
            self.sim.data.qvel[1], -self._max_velocity_joint1, self._max_velocity_joint1
        )