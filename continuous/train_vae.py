import argparse
import torch
import time
import torch.nn.functional as F
import os, sys, tempfile
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from utils.tb_log import configure_tb, log_and_print
from utils.buffer import ReplayBuffer
from tensorboard_logger import log_value
import datetime
import random
from tqdm import tqdm
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
import h5py
from utils.vae_class import VAE

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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    # dataset
    parser.add_argument('--env_id', type=str, default='simple_spread')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--dataset_dir', default='./datasets', type=str)
    parser.add_argument('--version', type=str, default='v2')
    # model
    parser.add_argument('--model', default='VAE', type=str)
    parser.add_argument('--hidden_dim', type=int, default=750)
    parser.add_argument('--beta', type=float, default=0.5)
    # train
    parser.add_argument('--num_iters', type=int, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--scheduler', default=False, action='store_true')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--no_max_action', default=False, action='store_true')
    parser.add_argument('--clip_to_eps', default=False, action='store_true')
    parser.add_argument('--eps', default=1e-4, type=float)
    parser.add_argument('--latent_dim', default=None, type=int, help="default: action_dim * 2")
    parser.add_argument('--no_normalize', default=False, action='store_true', help="do not normalize states")
    parser.add_argument("--data_type", default='medium', type=str)
    parser.add_argument("--n_training_threads", default=1, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--discrete_action", action='store_true', default=False)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--no_log", action='store_true')
    parser.add_argument("--logging_interval", default=20, type=int)

    parser.add_argument('--eval_data', default=0.0, type=float, help="proportion of data used for validation, e.g. 0.05")
    # work dir
    parser.add_argument("--dir", help="Name of directory to store model/training contents", type=str, default='results/vae')

    parser.add_argument('--notes', default=None, type=str)

    config = parser.parse_args()
    config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.seed)
    unique_token = "{}__{}__{}".format(config.data_type, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f"), config.seed)
    outdir = os.path.join(config.dir, config.env_id, unique_token)
    device='cuda'
    if not config.no_log:
        os.makedirs(outdir)
        configure_tb(outdir)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.set_num_threads(config.n_training_threads)

    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed, config.discrete_action)
        env_args, env_info = None, None
        if config.env_id in ['simple_spread']:
            ac_space = env.action_space
            ob_space = env.observation_space
        else:
            ac_space = [x for i,x in enumerate(env.action_space) if env.agent_types[i] == 'adversary']
            ob_space = [x for i,x in enumerate(env.observation_space) if env.agent_types[i] == 'adversary']
        n_agents = len(ac_space)
        state_dim = sum([ob_space[i].shape[0] for i in range(len(ob_space))])
        action_dim = sum([ac_space[i].shape[0] for i in range(len(ob_space))])
        max_action = float(ac_space[0].high[0])
        replay_buffer = ReplayBuffer(
            config.buffer_length, n_agents,
            [obsp.shape[0] for obsp in ob_space],
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in ac_space],
        )
    else:
        env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x3', "agent_obsk": 0,}
        env = MujocoMulti(env_args=env_args)
        env.seed(config.seed)

        env_info = env.get_env_info()
        n_agents = env_info["n_agents"]
        state_dim = env_info["state_shape"]
        action_dim = env_info["n_actions"]*env_info["n_agents"]
        max_action = env.action_space[0].high[0]
        replay_buffer = ReplayBuffer(
            config.buffer_length, n_agents,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space],
        )
    replay_buffer.load_batch_data(config.dataset_dir)
    latent_dim = action_dim * 2
    if config.latent_dim is not None:
        latent_dim = config.latent_dim
    if config.model == 'VAE':
        vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=config.hidden_dim).to(device)
    else:
        raise NotImplementedError
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.gamma)
    for step in tqdm(range(config.num_iters + 1), desc='train'):
        sample = replay_buffer.sample(config.batch_size, to_gpu=True)
        if config.env_id in ['HalfCheetah-v2']:
            states, obs, acs, rews, next_states, next_obs, dones, next_acs = sample
            train_states = states[0]
        else:
            obs, acs, rews, next_obs, dones, next_acs = sample
            train_states = torch.cat([ob.unsqueeze(1) for ob in obs], dim=1).reshape(obs[0].shape[0], obs[0].shape[1]*len(obs)) #bs, ds  
        train_actions = torch.cat(acs, dim=1)
        # Variational Auto-Encoder Training
        recon, mean, std = vae(train_states, train_actions)

        recon_loss = F.mse_loss(recon, train_actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + config.beta * KL_loss
        
        if step % config.logging_interval == 0:
            dic = {}
            dic.update({'recon_loss':recon_loss.item()})
            dic.update({'KL_loss':KL_loss.item()})
            dic.update({'vae_loss':vae_loss.item()})
            log_and_print(list(dic.keys()), list(dic.values()), step, multi=True)
        optimizer.zero_grad()
        vae_loss.backward()
        optimizer.step()
    torch.save(vae.state_dict(), os.path.join(outdir, "model.pt"))
    print('Model saved to', os.path.join(outdir, "model.pt"))