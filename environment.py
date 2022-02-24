import numpy as np
from typing import Optional, Union, Tuple
from unityagents import UnityEnvironment
import gym
from gym.core import ObsType, ActType
from gym import spaces


class Environment(gym.Env):

    def __init__(self, unityenv_path):
        super(Environment, self).__init__()
        self.unity_env = UnityEnvironment(file_name=unityenv_path)
        self.brain_name = self.unity_env.brain_names[0]
        env_info = self.unity_env.reset(train_mode=True)[self.brain_name]
        brain = self.unity_env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        observation = env_info.vector_observations[0]
        self.observation_size = observation.shape
        self.action_space = spaces.Discrete(n=self.action_size)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_size, dtype=np.float32)
        print(f'observation size: {self.observation_size} / action size: {self.action_size}')
        print(f'observation space: {self.observation_space} / action space: {self.action_space}')

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        env_info = self.unity_env.step(action)[self.brain_name]
        obs = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = bool(env_info.local_done[0])
        return obs, reward, done, {'env_info': env_info}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> Union[ObsType, Tuple[ObsType, dict]]:
        env_info = self.unity_env.reset(train_mode=options['train_mode'])[self.brain_name]
        observation = env_info.vector_observations[0]
        return observation

    def render(self, mode="human"):
        pass
