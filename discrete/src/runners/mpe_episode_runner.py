from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = 1#self.args.batch_size_run
        assert self.batch_size == 1

        if 'mpe_env' in self.args.env:
            self.pan_mpe_env=True if 'pan' in self.args.env else False
            # if self.pan_mpe_env:
            #     import panmultiagent.scenarios as mpe_scenarios
            # else:
            import multiagent.scenarios as mpe_scenarios
            scenario = mpe_scenarios.load(self.args.scenario_name + ".py").Scenario()
            world = scenario.make_world()
            
            self.args.env_args['world'] = world
            self.args.env_args['episode_limit'] = self.args.episode_limit

            self.args.env_args['reset_callback'] = scenario.reset_world
            self.args.env_args['reward_callback'] = scenario.reward
            self.args.env_args['observation_callback'] = scenario.observation
            self.args.env_args['post_step_callback'] = scenario.post_step if hasattr(scenario, 'post_step') else None
            self.args.env_args['done_callback'] = scenario.check_done if hasattr(scenario, 'check_done') else None
            self.args.env_args['info_callback'] = scenario.benchmark_data

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, opponents=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

        if opponents is not None:
            self.opponents = opponents
    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            obs = self.env.get_obs()
            if self.pan_mpe_env:
                if self.env.world.scenario_name in ['simple_tag', 'simple_world']:
                    obs_in_pre_transition_data = [obs[:self.env.num_cooperating_agents]]
                else:
                    obs_in_pre_transition_data = [obs[self.env.num_opponent_agents:]]
            else:
                obs_in_pre_transition_data = [obs]
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": obs_in_pre_transition_data
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            
            if self.pan_mpe_env:
                if self.env.world.scenario_name in ['simple_tag', 'simple_world']:
                    opponent_obs = obs[self.env.num_cooperating_agents:]
                elif self.env.world.scenario_name in ['simple_adversary', 'simple_crypto']:
                    opponent_obs = obs[:self.env.num_opponent_agents]
                opponent_obs = th.FloatTensor(np.expand_dims(opponent_obs, 1)).cuda()
            
                opponent_actions = th.cat([
                    self.opponents[i].step(opponent_obs[i], explore=False) for i in range(self.env.num_opponent_agents)
                ])
                opponent_actions = th.argmax(opponent_actions, 1) # convert one-hot to int
                opponent_actions = th.tensor(th.unsqueeze(opponent_actions, 0)).cuda()
            
                all_actions = th.cat((actions, opponent_actions), 1)

                reward, terminated, env_info = self.env.step(all_actions[0])
                if self.env.world.scenario_name in ['simple_tag', 'simple_world']:
                    reward, terminated = reward[0], terminated[0]
                else:
                    reward, terminated = reward[-1], terminated[-1]
            else:
                reward, terminated, env_info = self.env.step(actions[0])
                reward = reward[0]
                terminated = terminated[0]

            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        obs = self.env.get_obs()
        if self.pan_mpe_env:
            if self.env.world.scenario_name in ['simple_tag', 'simple_world']:
                obs_in_last_data = [obs[:self.env.num_cooperating_agents]]
            else:
                obs_in_last_data = [obs[self.env.num_opponent_agents:]]
        else:
            obs_in_last_data = [obs]
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": obs_in_last_data
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        if self.t_env > self.args.t_max:
            log_prefix = "final_" + log_prefix 
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        t_env = min(self.t_env,self.args.t_max)
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                # print(k,v)
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], t_env)
            else:
                self.logger.log_stat(prefix + k , v, t_env)
        stats.clear()
