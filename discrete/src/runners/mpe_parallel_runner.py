from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
# import panmultiagent.scenarios as panmpe_scenarios
# import multiagent.scenarios as mpe_scenarios
# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        self.pan_mpe_env=True if 'pan' in self.args.env else False
        # if self.pan_mpe_env:
        #     import panmultiagent.scenarios as mpe_scenarios
        # else:
        if 'mpe' in self.args.env:
            import multiagent.scenarios as mpe_scenarios
        for i, worker_conn in enumerate(self.worker_conns):
            if 'mpe_env' in self.args.env:
                scenario = mpe_scenarios.load(self.args.scenario_name + ".py").Scenario()
                world = scenario.make_world()
                
                self.args.env_args['world'] = world
                self.args.env_args['episode_limit'] = self.args.episode_limit

                self.args.env_args['reset_callback'] = scenario.reset_world
                self.args.env_args['reward_callback'] = scenario.reward
                self.args.env_args['observation_callback'] = scenario.observation
                self.args.env_args['post_step_callback'] = scenario.post_step if hasattr(scenario, 'post_step') else None
                self.args.env_args['info_callback'] = scenario.benchmark_data
            ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        # self.pan_mpe_env=True if 'pan' in self.args.env else False
        if self.pan_mpe_env:
            self.is_mpe = True
            self.parent_conns[0].send(("get_mpe_need",None))
            mpe_data = self.parent_conns[0].recv()
            self.num_cooperating_agents = mpe_data['num_cooperating_agents']
            self.num_opponent_agents = mpe_data['num_opponent_agents']

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac, opponents=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

        if opponents is not None:
            self.opponents = opponents

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        self.allobs = []
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            self.allobs.append(data["obs"])
            if self.pan_mpe_env:
                if self.args.scenario_name in ['simple_tag', 'simple_world']:
                    obs_in_pre_transition_data = self.allobs[-1][:self.num_cooperating_agents]
                else:
                    obs_in_pre_transition_data = self.allobs[-1][self.num_opponent_agents:]
            else:
                obs_in_pre_transition_data = self.allobs[-1]
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(obs_in_pre_transition_data)

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
        

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            
            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        if self.pan_mpe_env:
                            obs = self.allobs[idx]
                            if self.args.scenario_name in ['simple_tag', 'simple_world']:
                                opponent_obs = obs[self.num_cooperating_agents:]
                            elif self.args.scenario_name in ['simple_adversary', 'simple_crypto']:
                                opponent_obs = obs[:self.num_opponent_agents]
                            opponent_obs = th.FloatTensor(np.expand_dims(opponent_obs, 1)).cuda()
                        
                            opponent_actions = th.cat([
                                self.opponents[i].step(opponent_obs[i], explore=False) for i in range(self.num_opponent_agents)
                            ])
                            opponent_actions = th.argmax(opponent_actions, 1) # convert one-hot to int
                            opponent_actions = th.tensor(th.unsqueeze(opponent_actions, 0)).cuda()
                        
                            cpu_action = th.cat((actions[idx].unsqueeze(0), opponent_actions), 1).to('cpu')
                        else:
                            cpu_action = [cpu_actions[action_idx]]

                        parent_conn.send(("step", cpu_action[0]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            self.allobs = []
            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                self.allobs.append(-1)
                if not terminated[idx]:
                    data = parent_conn.recv()

                    obs = data["obs"]
                    self.allobs[-1]=obs
                    if self.pan_mpe_env:
                        if self.args.scenario_name in ['simple_tag', 'simple_world']:
                            obs_in_pre_transition_data = obs[:self.num_cooperating_agents]
                            data['reward'], data['terminated'] = data['reward'][0], data['terminated'][0]
                        else:
                            obs_in_pre_transition_data = obs[self.num_opponent_agents:]
                            data['reward'], data['terminated'] = data['reward'][-1], data['terminated'][-1]
                    else:
                        obs_in_pre_transition_data = obs
                        data['reward'], data['terminated'] = data['reward'][0], data['terminated'][0]

                    pre_transition_data["obs"].append(obs_in_pre_transition_data)
                    
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        if self.t_env > self.args.t_max:
            log_prefix = "final_" + log_prefix 
        infos = [cur_stats] + final_env_infos

        # if test_mode:
        #     print(infos)
        #     print(set.union(*[set(d) for d in infos]))
        #     for d in infos:
        #         print(d.get('success',0))
        #     print(sum(d.get('success', 0) for d in infos))


        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            # print(cur_stats)
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


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "get_mpe_need":
            remote.send({"num_cooperating_agents":env.num_cooperating_agents,
            "num_opponent_agents":env.num_opponent_agents})
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

