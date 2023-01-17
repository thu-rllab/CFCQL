import time
import numpy as np
import torch as th
from gym import spaces
from inspect import getargspec
import sys
import gym
import ic3net_envs
from ..multiagentenv import MultiAgentEnv
class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)
class GymWrapper(MultiAgentEnv):
    '''
    for multi-agent
    '''
    def __init__(self, env_name, **kwargs):
        args = obj(kwargs)
        self.args = args
        self.args.nfriendly = self.args.nagents
        # print(args)
        if env_name == 'predator_prey':
            env = gym.make('PredatorPrey-v0')
        elif env_name == 'traffic_junction':
            env = gym.make('TrafficJunction-v0')
        else:
            raise RuntimeError("wrong env name")
        # env = gym.make(env_name)
        if args.display:
            env.init_curses()
        env.multi_agent_init(args)
        # env = GymWrapper(env)

        self.env = env
        self.episode_limit = self.args.max_steps
        self.epochs = 0
        self.n_agents = self.args.nagents

    #######multiagentenv##############33
    def step(self, _actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(_actions):
            actions = _actions.cpu().numpy()
        else:
            actions = _actions
        self.time_step += 1
        obs, rewards, done, infos = self.env.step(actions.tolist())
        obs = self._flatten_obs(obs)
        self.obs = np.array(obs)
        
        info = {"success":0}

        if self.time_step >= self.episode_limit:
            done = True
            # info["episode_limit"] = True
            if hasattr(self.env, 'stat'):
                self.env.stat.pop('steps_taken', None)
                info["success"] = self.env.stat["success"]
            
        # if infos['is_completed'].sum() == self.n_agents:
        #     info["battle_won"] = True
        # self.render()
        # time.sleep(5)
        # self.end_render()

        return sum(rewards), done, info
    
    def get_obs(self):
        """Returns all agent observations in a list."""
        return self.obs.reshape(self.n_agents, -1)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.obs[agent_id].reshape(-1)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.observation_dim

    def get_global_state(self):
        return self.obs.flatten()

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.dim_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""

        return self.dim_actions

    def reset(self):
        """Returns initial observations and states."""
        self.epochs += 1
        self.time_step = 0
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            obs = self.env.reset(self.epochs)
        else:
            obs = self.env.reset()
        obs = self._flatten_obs(obs)
        self.obs = np.array(obs)
        # print(self.obs,self.obs.shape)

        return self.get_obs(), self.get_global_state()

    
    def render(self):
        # pass
        self.env.render()
        time.sleep(0.5)
    def end_render(self):
        self.env.exit_render()

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def get_stats(self):
        # pass
        if hasattr(self.env, 'stat'):
            self.env.stat.pop('steps_taken', None)
            return self.env.stat
        else:
            return dict()

    def save_replay(self):
        """Save a replay."""
        pass

    ###################################

    @property
    def observation_dim(self):
        '''
        for multi-agent, this is the obs per agent
        '''

        # tuple space
        if hasattr(self.env.observation_space, 'spaces'):
            total_obs_dim = 0
            for space in self.env.observation_space.spaces:
                if hasattr(self.env.action_space, 'shape'):
                    total_obs_dim += int(np.prod(space.shape))
                else: # Discrete
                    total_obs_dim += 1
            return total_obs_dim
        else:
            return int(np.prod(self.env.observation_space.shape))

    @property
    def num_actions(self):
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return int(self.env.action_space.nvec[0])
        elif hasattr(self.env.action_space, 'n'):
            # Discrete
            return self.env.action_space.n

    @property
    def dim_actions(self):
        # for multi-agent, this is the number of action per agent
        if hasattr(self.env.action_space, 'nvec'):
            # MultiDiscrete
            return self.env.action_space.shape[0]
            # return len(self.env.action_space.shape)
        elif hasattr(self.env.action_space, 'n'):
            # Discrete => only 1 action takes place at a time.
            return 1

    @property
    def action_space(self):
        return self.env.action_space

    def _reset(self, epoch):
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            obs = self.env.reset(epoch)
        else:
            obs = self.env.reset()

        obs = self._flatten_obs(obs)
        return obs


    def end_display(self):
        self.env.exit_render()

    def _step(self, action):
        # TODO: Modify all environments to take list of action
        # instead of doing this
        if self.dim_actions == 1:
            action = action[0]
        obs, r, done, info = self.env.step(action)
        obs = self._flatten_obs(obs)
        return (obs, r, done, info)

    def reward_terminal(self):
        if hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return np.zeros(1)

    def _flatten_obs(self, obs):
        if isinstance(obs, tuple):
            _obs=[]
            for agent in obs: #list/tuple of observations.
                ag_obs = []
                for obs_kind in agent:
                    ag_obs.append(np.array(obs_kind).flatten())
                _obs.append(np.concatenate(ag_obs))
            obs = np.stack(_obs)
        # print(obs,self.observation_dim)
        obs = obs.reshape(1, -1, self.observation_dim)
        obs = th.from_numpy(obs).double()
        return obs

