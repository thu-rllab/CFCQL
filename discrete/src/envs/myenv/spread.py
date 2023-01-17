from envs.multiagentenv import MultiAgentEnv
import torch as th
import numpy as np
import random
import pygame
from utils.dict2namedtuple import convert
from copy import deepcopy
class Spread(MultiAgentEnv):
    def __init__(self,map_name='spread',n_agents=3,n_landmarks=3,line_length=10,vision_range=3,episode_limit=10,**kwargs) -> None:
        self.line_length=line_length
        self.n_agents=n_agents
        self.n_landmarks=n_landmarks
        self.vision_range=vision_range
        self.episode_limit=episode_limit
        self.action_book = [0.1*i for i in range(-10,11)]
        self.n_actions = len(self.action_book)
        self.agent_size = 0.05
        self.reset()

    def reset(self):
        """ Returns initial observations and states"""
        self.t = 0
        self.last_min_dis = 0
        self.pos = np.random.random(size=self.n_agents)*self.line_length
        self.landmarks = np.random.random(size=self.n_landmarks)*self.line_length
        self.last_pos = deepcopy(self.pos) 
        self.landmarks_occupied = [0]*self.n_landmarks
    
    def step(self, actions):
        """ Returns reward, terminated, info """
        self.t+=1
        actions_int = [int(a) for a in actions]
        move = []
        self.last_pos = deepcopy(self.pos) 
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
        new_landmarks_occupied = [0]*self.n_landmarks
        for i in range(self.n_landmarks):
            for j in range(self.n_agents):
                if np.abs(self.pos[j]-self.landmarks[i])<=self.agent_size:
                    new_landmarks_occupied[i]=1
                    break
        collision = self._is_collision()
        reward = 10*(sum(new_landmarks_occupied)-sum(self.landmarks_occupied))-collision
        self.landmarks_occupied = new_landmarks_occupied
        return reward
    
    def _is_collision(self):
        collision = 0
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if j==i:
                    continue
                # if (self.last_pos[i]-self.last_pos[j])*(self.pos[i]-self.pos[j])<0:
                if np.abs(self.pos[i]-self.pos[j])<self.agent_size:
                    collision+=1
        return collision

        

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