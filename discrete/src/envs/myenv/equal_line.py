from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert
from copy import deepcopy
class EqualLine(MultiAgentEnv):
    def __init__(self,map_name='equal_line',n_agents=3,line_length=10,vision_range=3,episode_limit=10,**kwargs) -> None:
        self.line_length=max(10,n_agents*2)
        self.n_agents=n_agents
        self.vision_range=vision_range
        self.episode_limit=episode_limit
        self.action_book = [0,-0.01,-0.05,-0.1,-0.5,-1,0.01,0.05,0.1,0.5,1]#[0.1*i for i in range(-10,11)]#[0,0.01,0.1,1,-0.01,-0.1,-1]
        self.n_actions = len(self.action_book)
        self.reset()

    def reset(self):
        """ Returns initial observations and states"""
        self.t = 0
        self.last_min_dis = 0
        self.pos = np.random.random(size=self.n_agents)*2
    
    def step(self, actions):
        """ Returns reward, terminated, info """
        self.t+=1
        actions_int = [int(a) for a in actions]
        move = []
        for i in range(self.n_agents):
            self.pos[i] +=self.action_book[actions_int[i]]
            self.pos[i] = min(self.pos[i],self.line_length)
            self.pos[i] = max(self.pos[i],0)
        
        if self.t>=self.episode_limit:
            done = True
        else:
            done = False

        reward = self._cal_reward()

        return reward,done,{}
    def _cal_reward(self):
        max_dis = 0
        min_dis = self.line_length
        for i in range(self.n_agents):
            dis_to_i = np.abs(np.delete(self.pos,i)-self.pos[i])
            max_dis = max(max_dis,np.max(dis_to_i))
            min_dis = min(min_dis,np.min(dis_to_i))
        
        reward = (min_dis-self.last_min_dis)/(self.line_length/(self.n_agents-1))
        self.last_min_dis = min_dis
        # if reward==0:
        #     print(min_dis,self.pos,self.t)
        return reward*10
        

    def get_obs(self):
        """ Returns all agent observations in a list """
        return [self.get_obs_agent(0)]*self.n_agents

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return deepcopy(self.pos/self.line_length)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.pos.shape[0]

    def get_state(self):
        return deepcopy(self.pos/self.line_length)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.pos.shape[0]

    def get_avail_actions(self):
        return [[1]*self.n_actions]*self.n_agents

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1]*self.n_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    

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