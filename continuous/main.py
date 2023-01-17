import argparse
import torch
import time
import os, sys, tempfile
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from utils.tb_log import configure_tb, log_and_print
from utils.buffer import ReplayBuffer
from algorithms.maddpg import MATD3
from algorithms.maddpg_cc import MATD3CC
from tensorboard_logger import log_value
import datetime
import random
import copy
import shutil
from utils.vae_class import VAE


from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import h5py

try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('MujocoMulti not installed')

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def eval_policy(agent, env_name, seed, eval_episodes, discrete_action, env_args=None):
    if env_name in ['HalfCheetah-v2']:
        env = MujocoMulti(env_args=env_args)
        env.seed(seed + 100)

        all_episodes_rewards = []
        for ep_i in range(eval_episodes):
            agent.prep_rollouts(device='cpu')

            env.reset()
            done = False
            episode_reward = 0.
            # episode_num=0
            # r30 = 0
            while not done:
                obs = env.get_obs()
                torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
                torch_agent_actions = agent.step(torch_obs, explore=False)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                actions = [ac.squeeze(0) for ac in agent_actions]
                
                reward, done, info = env.step(actions)
                # if episode_num<30:
                #     r30+=reward
                # episode_num += 1    
                episode_reward += reward
            # print(r30)
            all_episodes_rewards.append(episode_reward)
        
        mean_episode_reward = np.mean(np.array(all_episodes_rewards))
        return mean_episode_reward
    else:
        avg_predator_return = 0.
    
        env = make_parallel_env(env_name, 1, seed + 100, discrete_action)

        for ep_i in range(0, eval_episodes):
            obs = env.reset()
            agent.prep_rollouts(device='cpu')

            for et_i in range(config.episode_length):
                obs_len = agent.nagents
                if env_name in ['simple_tag', 'simple_world'] and "CC" in agent.__class__.__name__:
                    obs_len += agent.num_preys
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(obs_len)]
                torch_agent_actions = agent.step(torch_obs, explore=False)
                agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                next_obs, rewards, dones, infos = env.step(actions)
                
                if env_name in ['simple_tag', 'simple_world']:
                    avg_predator_return += rewards[0][0]
                else:
                    avg_agent_reward = np.mean(rewards[0])
                    avg_predator_return += avg_agent_reward

                obs = next_obs

        avg_predator_return /= eval_episodes
        return avg_predator_return


def offline_train(config):
    unique_token = "{}__{}__{}".format(config.data_type, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"), config.seed)
    if config.omar:
        unique_token="omar_"+unique_token
    if config.cf_cql:
        unique_token="cfcql_"+unique_token
    elif config.cql:
        unique_token="cql_"+unique_token
    if config.central_critic:
        unique_token="CC_"+unique_token
    if not config.no_log:
        outdir = os.path.join(config.dir,config.env_id, unique_token)
        os.makedirs(outdir)
        # outdir = prepare_output_dir(config.dir + '/' + config.env_id, argv=sys.argv)
        print('\033[1;32mOutput files are saved in {} \033[1;0m'.format(outdir))
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(config.n_training_threads)

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        if config.env_id == 'simple_spread':
            config.lr=0.01
        elif config.env_id == 'simple_world':
            config.steps_per_update=20

        env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.discrete_action)
        env_args, env_info = None, None
    else:
        env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
        env = MujocoMulti(env_args=env_args)
        env.seed(config.seed)

        env_info = env.get_env_info()

        config.batch_size = 256
        config.hidden_dim = 256
        if not config.set_lr:
            config.lr = 0.0003
        config.tau = 0.005
        config.gamma = 0.99

        config.omar_iters = 2
        if not config.set_omar_num_samples:
            config.omar_num_samples = 50
        config.omar_num_elites = 5 
    if not config.no_log:
        configure_tb(outdir)
    kwargs={}
    if config.set_alpha:
        kwargs.update({"cql_alpha":config.cql_alpha})
    if config.set_omar_coe:
        kwargs.update({"omar_coe":config.omar_coe})
    if config.central_critic:
        ma_agent = MATD3CC.init_from_env(
            env, config.env_id, config.data_type,
            tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim,
            cql=config.cql, lse_temp=config.lse_temp, batch_size=config.batch_size, num_sampled_actions=config.num_sampled_actions,
            omar=config.omar, omar_iters=config.omar_iters, omar_mu=config.omar_mu, omar_sigma=config.omar_sigma, omar_num_samples=config.omar_num_samples, omar_num_elites=config.omar_num_elites, 
            env_info=env_info, logging_interval=config.logging_interval, no_log=config.no_log,cf_cql=config.cf_cql, soft_q = not(config.no_soft_q),
            action_noise_scale=config.action_noise_scale, cf_omar=not(config.no_cf_omar), cf_tau=config.cf_tau, no_action_reg=config.no_action_reg, 
            bc_tau=config.bc_tau, sample_action_class=config.sample_action_class, beta_action_noise=config.beta_action_noise,
            cf_pol=not(config.no_cf_pol), cf_target=config.cf_target, **kwargs
        )
    else:
        ma_agent = MATD3.init_from_env(
            env, config.env_id, config.data_type,
            tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim,
            cql=config.cql, lse_temp=config.lse_temp, batch_size=config.batch_size, num_sampled_actions=config.num_sampled_actions,
            omar=config.omar, omar_iters=config.omar_iters, omar_mu=config.omar_mu, omar_sigma=config.omar_sigma, omar_num_samples=config.omar_num_samples, omar_num_elites=config.omar_num_elites, 
            env_info=env_info, logging_interval=config.logging_interval, no_log=config.no_log,cf_cql=config.cf_cql, soft_q = not(config.no_soft_q), **kwargs
        )

    if config.env_id in ['simple_tag', 'simple_world']:
        pretrained_model_dir = './datasets/{}/pretrained_adv_model.pt'.format(config.env_id)
        ma_agent.load_pretrained_preys(pretrained_model_dir)

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [obsp.shape[0] for obsp in env.observation_space],
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space],
        )
    else:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space],
        )
    replay_buffer.load_batch_data(config.dataset_dir, rew_scale = config.rew_scale)
    if np.isinf(replay_buffer.ave_reward):
        replay_buffer.ave_reward = replay_buffer.sum_reward / (replay_buffer.filled_i/config.episode_length)
    print('Average_reward:', replay_buffer.ave_reward)

    #load vae weight
    if config.cf_weight:
        ma_agent.cf_weight=True
        sample = replay_buffer.sample(config.batch_size, to_gpu=True)
        if config.env_id in ['HalfCheetah-v2']:
            states, obs, acs, rews, next_states, next_obs, dones, next_acs = sample
            train_states = states[0]
        else:
            obs, acs, rews, next_obs, dones, next_acs = sample
            train_states = torch.cat([ob.unsqueeze(1) for ob in obs], dim=1).reshape(obs[0].shape[0], obs[0].shape[1]*len(obs)) #bs, ds  
        train_actions = torch.cat(acs, dim=1)
        state_dim = train_states.shape[1]
        action_dim = train_actions.shape[1]
        max_action = env.action_space[0].high[0]
        device='cuda'
        vae = VAE(state_dim, action_dim, action_dim*2, max_action, hidden_dim=config.vae_hidden_dim).to(device)
        load_path = os.path.join(config.vae_model_dir, config.env_id)
        all_dirs = os.listdir(load_path)
        dd = ""
        for d in all_dirs:
            if d.startswith(config.data_type+"_") and d.endswith(str(config.dataset_num)):
                dd=d
                break
        if not len(dd):
            print("No weight of vae finded!")
            exit(1)
        model_path = os.path.join(config.vae_model_dir, config.env_id, dd, "model.pt")
        print("Load from", model_path)
        vae.load_state_dict(torch.load(model_path))
        vae.eval()
        ma_agent.vae = vae

    for t in range(config.num_steps + 1):
        if t % config.eval_interval == 0 or t == config.num_steps:
            eval_return = eval_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action, env_args=env_args)
            if not config.no_log:
                log_and_print('eval_return', eval_return, t)
                log_and_print('normed_eval_return', eval_return/replay_buffer.ave_reward, t)
                

        if (t % config.steps_per_update) < config.n_rollout_threads:
            ma_agent.prep_training(device='gpu') if config.use_gpu else ma_agent.prep_training(device='cpu')

            for u_i in range(config.n_rollout_threads):
                if config.central_critic:
                    sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                    ma_agent.update(sample, t)
                    for j in range(config.ca_ratio-1):
                        sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
                        ma_agent.update(sample, t+j+1, only_critic=True)
                    
                else:
                    nagents = ma_agent.nagents if config.env_id in ['simple_spread', 'HalfCheetah-v2'] else ma_agent.num_predators

                    for a_i in range(nagents):
                        sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)

                        ma_agent.update(sample, a_i, t)

                ma_agent.update_all_targets()
    try:
        env.close()
    except:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Name of directory to store model/training contents", type=str, default='results')

    parser.add_argument("--env_id", help="Name of environment", type=str, default='simple_spread')
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset number")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=1, type=int)
    parser.add_argument("--discrete_action", action='store_true', default=False)
    parser.add_argument("--use_gpu", default=1, type=int)

    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--set_lr", action='store_true')
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--rew_scale", default=1.0, type=float)
    

    parser.add_argument('--gaussian_noise_std', default=0.1, type=float)

    parser.add_argument("--data_type", default='medium', type=str)
    parser.add_argument('--dataset_dir', default='./datasets', type=str)

    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--eval_interval', default=1000, type=int)
    parser.add_argument('--num_steps', default=int(1e5), type=int)

    parser.add_argument("--cql", action='store_true')
    parser.add_argument("--set_alpha", action='store_true')
    parser.add_argument('--cql_alpha', default=1.0, type=float)
    parser.add_argument("--lse_temp", default=1.0, type=float)
    parser.add_argument('--num_sampled_actions', default=10, type=int)
    
    parser.add_argument('--sample_action_class', default="1-1-1", type=str, help="random,cur_a,next_a;1 use cf;0 not use")
    parser.add_argument('--beta_action_noise', default=0.00001, type=float)
    
     
    parser.add_argument('--cql_sample_noise_level', default=0.2, type=float)

    parser.add_argument("--omar", action='store_true')
    parser.add_argument('--omar_coe', default=1.0, type=float) 
    parser.add_argument('--set_omar_coe', action='store_true')
    parser.add_argument('--omar_iters', default=3, type=int)
    parser.add_argument('--omar_mu', default=0., type=float)
    parser.add_argument('--omar_sigma', default=2.0, type=float)
    parser.add_argument("--set_omar_num_samples", action='store_true')
    parser.add_argument('--omar_num_samples', default=10, type=int)
    parser.add_argument('--omar_num_elites', default=10, type=int)

    parser.add_argument("--logging_interval", default=100, type=int)
    parser.add_argument("--central_critic", action='store_true')
    parser.add_argument("--no_log", action='store_true')
    parser.add_argument("--cf_cql", action='store_true')
    parser.add_argument("--no_soft_q", action='store_true')
    parser.add_argument("--ca_ratio", default=5, type=int)
    parser.add_argument("--action_noise_scale", default=0.05, type=float)
    parser.add_argument("--no_cf_omar", action='store_true')
    parser.add_argument("--no_cf_pol", action='store_true')
    parser.add_argument("--cf_target", action='store_true')
    
    

    parser.add_argument("--cf_weight", action='store_true')
    parser.add_argument("--no_action_reg", action='store_true')
    parser.add_argument("--cf_tau",default=1.0, type=float)
    parser.add_argument("--bc_tau",default=0.0, type=float)
    parser.add_argument("--vae_model_dir", default="./results/vae", type=str)
    parser.add_argument("--vae_hidden_dim", default=750, type=int)
    
    config = parser.parse_args()

    if config.env_id in ['simple_spread', 'simple_tag']:
        config.num_steps = 200000
    elif config.env_id == 'HalfCheetah-v2':
        config.num_steps = int(1e6)
        config.steps_per_update = 10
        config.eval_interval = 5000
        config.episode_length=1000
        config.gamma=0.99
        
    config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)
        
    offline_train(config)