from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert
from copy import deepcopy
class Consensus(MultiAgentEnv):
    def __init__(self,n_agents=3,**kwargs) -> None:
        self.n_agents=n_agents
        self.episode_limit=1
        self.n_actions = 3#stay,left,right
        self.pos = list(np.random.randint(0,3,size=self.n_agents))

    def reset(self):
        """ Returns initial observations and states"""
        self.pos = list(np.random.randint(0,3,size=self.n_agents))
    
    def step(self, actions):
        """ Returns reward, terminated, info """
        actions_int = [int(a) for a in actions]
        rewards = [0,0,0]
        # print('before:',self.pos,actions_int)
        for i in range(self.n_agents):
            if actions_int[i]==1:
                if self.pos[i]>0:
                    self.pos[i]-=1
            elif actions_int[i]==2:
                if self.pos[i]<2:
                    self.pos[i]+=1
            rewards[self.pos[i]]+=1

        # print(rewards,self.pos)
        if rewards[1]>self.n_agents-self.n_agents:
            reward = rewards[1]/self.n_agents
        else:
            reward = 0.1*(rewards[0]+rewards[2])/self.n_agents


        return reward,True,{'episode_limit':True}
    
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