from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert
from copy import deepcopy
class MMDP(MultiAgentEnv):
    def __init__(self,n_agents=3,episode_limit=10,**kwargs) -> None:
        self.n_agents=n_agents
        self.episode_limit=episode_limit
        self.n_actions = 2
        self.reset()

    def reset(self):
        """ Returns initial observations and states"""
        self.pos = [1]*self.n_agents
        self.t=0
    
    def step(self, actions):
        """ Returns reward, terminated, info """
        self.t+=1
        actions_int = [int(a) for a in actions]
        rewards = [0,0,0]
        if self.t>=self.episode_limit:
            done=True
        else:
            done=False
        # print('before:',self.pos,actions_int)
        if self.pos == [0]*self.n_agents:
            return 0,done,{'episode_limit':True}

        
        if sum(actions_int) == 0:
            reward = 1
        elif sum(actions_int)<=self.n_agents//2:
            self.pos = [0]*self.n_agents
            reward = 0
        else:
            reward=0
        
        # if done and sum(actions_int) == 0 and sum(self.pos)==0:
        #     reward=1
        return reward,done,{'episode_limit':True}
    
    def get_obs(self):
        """ Returns all agent observations in a list """
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return deepcopy([self.pos[agent_id]])

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return 1

    def get_state(self):
        return deepcopy(self.pos)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents

    def get_avail_actions(self):
        return [[1]*self.n_actions]*self.n_agents

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1]*self.n_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def set_state(self,pos):
        self.pos=deepcopy(pos)

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass
    def get_stats(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info