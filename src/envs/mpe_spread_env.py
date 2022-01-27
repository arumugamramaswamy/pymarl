from envs.multiagentenv import MultiAgentEnv
from pettingzoo.mpe import simple_spread_v2
import supersuit as ss
import numpy as np

class SimpleSpreadEnv(MultiAgentEnv):

    def __init__(self, N=3, episode_limit=25, **kwargs) -> None:
        super().__init__()
        env = simple_spread_v2.parallel_env(N=N, max_cycles=episode_limit)
        self.env = ss.pettingzoo_env_to_vec_env_v0(env)
        self.observation = self.env.reset()
        self.n_actions = 5
        self.N = N
        self.episode_limit = episode_limit

    def step(self, actions):
        """ Returns reward, terminated, info """
        obs, rewards, dones, info = self.env.step(actions)
        if all(dones):
            # extract terminal obs from info
            self.observation = [x["terminal_observation"] for x in info]
            terminated = True
            
        else:
            self.observation = obs
            terminated = False
            
        return rewards.sum(), terminated, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.observation

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.observation[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.get_obs_agent(0).shape[0]

    def get_state(self):
        return np.concatenate(self.get_obs())

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.get_state().shape[0]

    def get_avail_actions(self):

        return [[1 for _ in range(self.n_actions)] for _ in range(self.N)]
    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1 for _ in range(self.n_actions)]

    def reset(self):
        """ Returns initial observations and states"""
        self.observation = self.env.reset()
        return self.get_obs(), self.get_state()

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_agents": self.N,
                    "episode_limit": self.episode_limit,
                    "n_actions": self.n_actions,
                    }
        return env_info

    def get_stats(self):
        return {}

    def close(self):
        pass

    def render(self, mode='human'):
        self.env.render(mode=mode)
