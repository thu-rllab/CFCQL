import copy
from torch.distributions import Categorical
from components.episode_buffer import EpisodeBatch
from modules.critics.doubleqv import DoubleQNetwork,ValueNetwork
import torch as th
from utils.rl_utils import build_td_lambda_targets
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

def asymmetric_l2_loss(u, tau):
    return th.abs(tau - (u < 0).float()) * u**2

def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)

EXP_ADV_MAX = 100.

class IQLLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.q = DoubleQNetwork(scheme, args)
        self.value = ValueNetwork(scheme, args)
        self.target_q = copy.deepcopy(self.q)


        self.agent_params = list(mac.parameters())
        self.q_params = list(self.q.parameters())
        self.value_params = list(self.value.parameters())

        self.agent_optimiser = Adam(params=self.agent_params,  lr=args.lr)
        self.q_optimiser = Adam(params=self.q_params,  lr=args.lr)
        self.value_optimiser = Adam(params=self.value_params,  lr=args.lr)

        self.policy_lr_schedule = CosineAnnealingLR(self.agent_optimiser, args.t_max)

        self.tau = args.tau
        self.beta = args.beta
        self.gamma = args.gamma
        self.alpha = args.alpha

    def train(self, batch: EpisodeBatch, t_env: int, log):
        actions = batch["actions"][:, :-1].long()
        max_t = actions.shape[1]+1
        bs = actions.shape[0]
        terminated = batch["terminated"][:, :-1].float()#bs,t,1
        avail_actions = batch["avail_actions"][:, :-1]
        rewards = batch["reward"][:, :-1]#bs,t,1
        mask = batch["filled"][:, :-1].float() 

        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])#bs,t,1

        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1,1,self.n_agents,1)

        v_inputs = self.value._build_inputs(batch, bs, max_t)
        v_out = self.value(v_inputs)#bs,t+1,na,1
        v = v_out[:,:-1]
        next_v = v_out[:,1:].detach()

        with th.no_grad():
            inputs = self.target_q._build_inputs(batch, bs, max_t)
            target_q_all = self.target_q.forward(inputs).detach()[:, :-1]#bs,t,na,ad
            target_q = th.gather(target_q_all,dim=-1,index=actions)#bs,t,na,1

        
        adv = target_q - v
        v_loss = (asymmetric_l2_loss(adv, self.tau)*mask).sum()/mask.sum()
        self.value_optimiser.zero_grad(set_to_none=True)
        v_loss.backward()
        v_grad_norm = th.nn.utils.clip_grad_norm_(self.value_params, self.args.grad_norm_clip)
        self.value_optimiser.step()

        # Update Q function
        targets = rewards.unsqueeze(-1) + (1. - terminated.unsqueeze(-1).float()) * self.gamma * next_v.detach()
        q_inputs = self.q._build_inputs(batch, bs, max_t)
        qs = self.q.both(q_inputs)
        q_loss = sum((((th.gather(q,dim=-1,index=actions) - targets)*mask)**2).sum()/ mask.sum() for q in qs) / len(qs) 
        self.q_optimiser.zero_grad(set_to_none=True)
        q_loss.backward()
        q_grad_norm = th.nn.utils.clip_grad_norm_(self.q_params, self.args.grad_norm_clip)
        self.q_optimiser.step()

         # Update target Q network
        update_exponential_moving_average(self.target_q, self.q, self.alpha)

        # Update policy
        exp_adv = th.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)#bs,t,na,1

        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(max_t - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        pi_taken = th.gather(mac_out, dim=-1, index=actions)#bs,t,na,1
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)

        
        policy_loss = -((exp_adv * log_pi_taken)*mask).sum()/mask.sum()
        self.agent_optimiser.zero_grad(set_to_none=True)
        policy_loss.backward()
        policy_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()
        self.policy_lr_schedule.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("value_loss", v_loss.item(), t_env)
            self.logger.log_stat("value_grad_norm", v_grad_norm.item(), t_env)
            self.logger.log_stat("q_loss", q_loss.item(), t_env)
            self.logger.log_stat("q_grad_norm", q_grad_norm.item(), t_env)
            self.logger.log_stat("policy_loss", policy_loss.item(), t_env)
            self.logger.log_stat("policy_grad_norm", policy_grad_norm.item(), t_env)
            self.logger.log_stat("v_mean", (v*mask).sum()/mask.sum(), t_env)
            self.logger.log_stat("q_taken_mean", ((th.gather(qs[0],dim=-1,index=actions))*mask).sum()/mask.sum(), t_env)
            self.logger.log_stat("target_mean", (target_q*mask).sum()/mask.sum(), t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
        self.q.cuda()
        self.value.cuda()
        self.target_q.cuda()



