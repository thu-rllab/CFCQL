import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from tqdm import tqdm
import os

from utils.vae import VAE
import time
from coolname import generate_slug
import utils
import json
from log import Logger
import d4rl
from utils import get_lr


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
# dataset
parser.add_argument('--env', type=str, default='hopper')
parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
parser.add_argument('--version', type=str, default='v2')
# model
parser.add_argument('--model', default='VAE', type=str)
parser.add_argument('--hidden_dim', type=int, default=750)
parser.add_argument('--beta', type=float, default=0.5)
# train
parser.add_argument('--num_iters', type=int, default=int(1e5))
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--scheduler', default=False, action='store_true')
parser.add_argument('--gamma', default=0.95, type=float)
parser.add_argument('--no_max_action', default=False, action='store_true')
parser.add_argument('--clip_to_eps', default=False, action='store_true')
parser.add_argument('--eps', default=1e-4, type=float)
parser.add_argument('--latent_dim', default=None, type=int, help="default: action_dim * 2")
parser.add_argument('--no_normalize', default=False, action='store_true', help="do not normalize states")

parser.add_argument('--eval_data', default=0.0, type=float, help="proportion of data used for validation, e.g. 0.05")
# work dir
parser.add_argument('--work_dir', type=str, default='train_vae')
parser.add_argument('--notes', default=None, type=str)

args = parser.parse_args()

# make directory
base_dir = 'runs'
utils.make_dir(base_dir)
base_dir = os.path.join(base_dir, args.work_dir)
utils.make_dir(base_dir)
args.work_dir = os.path.join(base_dir, args.env + '_' + args.dataset)
utils.make_dir(args.work_dir)

ts = time.gmtime()
ts = time.strftime("%m-%d-%H:%M", ts)
exp_name = str(args.env) + '-' + str(args.dataset) + '-' + ts + '-bs'  \
    + str(args.batch_size) + '-s' + str(args.seed) + '-b' + str(args.beta) + \
    '-h' + str(args.hidden_dim) + '-lr' + str(args.lr) + '-wd' + str(args.weight_decay)
exp_name += '-' + generate_slug(2)
if args.notes is not None:
    exp_name = args.notes + '_' + exp_name
args.work_dir = args.work_dir + '/' + exp_name
utils.make_dir(args.work_dir)

args.model_dir = os.path.join(args.work_dir, 'model')
utils.make_dir(args.model_dir)

with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)

utils.snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')
logger = Logger(args.work_dir, use_tb=True)

utils.set_seed_everywhere(args.seed)

device = 'cuda'

# load data
env_name = f"{args.env}-{args.dataset}-{args.version}"
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
if args.no_max_action:
    max_action = None
print(state_dim, action_dim, max_action)
latent_dim = action_dim * 2
if args.latent_dim is not None:
    latent_dim = args.latent_dim

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
if not args.no_normalize:
    mean, std = replay_buffer.normalize_states()
else:
    print("No normalize")
if args.clip_to_eps:
    replay_buffer.clip_to_eps(args.eps)
states = replay_buffer.state
actions = replay_buffer.action

if args.eval_data:
    eval_size = int(states.shape[0] * args.eval_data)
    eval_idx = np.random.choice(states.shape[0], eval_size, replace=False)
    train_idx = np.setdiff1d(np.arange(states.shape[0]), eval_idx)
    eval_states = states[eval_idx]
    eval_actions = actions[eval_idx]
    states = states[train_idx]
    actions = actions[train_idx]
else:
    eval_states = None
    eval_actions = None

# train
if args.model == 'VAE':
    vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=args.hidden_dim).to(device)
else:
    raise NotImplementedError
optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)

total_size = states.shape[0]
batch_size = args.batch_size

for step in tqdm(range(args.num_iters + 1), desc='train'):
    idx = np.random.choice(total_size, batch_size)
    train_states = torch.from_numpy(states[idx]).to(device)
    train_actions = torch.from_numpy(actions[idx]).to(device)

    # Variational Auto-Encoder Training
    recon, mean, std = vae(train_states, train_actions)

    recon_loss = F.mse_loss(recon, train_actions)
    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    vae_loss = recon_loss + args.beta * KL_loss

    logger.log('train/recon_loss', recon_loss, step=step)
    logger.log('train/KL_loss', KL_loss, step=step)
    logger.log('train/vae_loss', vae_loss, step=step)

    optimizer.zero_grad()
    vae_loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        logger.dump(step)
        torch.save(vae.state_dict(), '%s/vae_model_%s_%s_b%s_%s.pt' %
                   (args.model_dir, args.env, args.dataset, str(args.beta), step))

        if eval_states is not None and eval_actions is not None:
            vae.eval()
            with torch.no_grad():
                eval_states_tensor = torch.from_numpy(eval_states).to(device)
                eval_actions_tensor = torch.from_numpy(eval_actions).to(device)

                # Variational Auto-Encoder Evaluation
                recon, mean, std = vae(eval_states_tensor, eval_actions_tensor)

                recon_loss = F.mse_loss(recon, eval_actions_tensor)
                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss = recon_loss + args.beta * KL_loss

                logger.log('eval/recon_loss', recon_loss, step=step)
                logger.log('eval/KL_loss', KL_loss, step=step)
                logger.log('eval/vae_loss', vae_loss, step=step)
            vae.train()

    if args.scheduler and (step + 1) % 10000 == 0:
        logger.log('train/lr', get_lr(optimizer), step=step)
        scheduler.step()

logger._sw.close()
