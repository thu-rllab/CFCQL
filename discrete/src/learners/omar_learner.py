import copy
from torch.distributions import Categorical
from torch.optim.rmsprop import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.critics.offpg import OffPGCritic
import torch as th
from utils.rl_utils import build_td_lambda_targets, build_target_q
from torch.optim import Adam
from modules.mixers.qmix import QMixer
from utils.th_utils import get_parameters_num
import numpy as np

class OMARLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.critic = OffPGCritic(scheme, args)
        # self.mixer = QMixer(args)
        self.target_critic = copy.deepcopy(self.critic)
        # self.target_mixer = copy.deepcopy(self.mixer)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        # self.mixer_params = list(self.mixer.parameters())
        # self.params = self.agent_params + self.critic_params
        self.c_params = self.critic_params# + self.mixer_params

        # self.agent_optimiser =  Adam(params=self.agent_params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        self.agent_optimiser =  RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser =  Adam(params=self.critic_params, lr=args.critic_lr, weight_decay=getattr(args, "weight_decay", 0))
        # self.mixer_optimiser =  RMSprop(params=self.mixer_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        print('Mixer Size: ')
        print(get_parameters_num(list(self.c_params)))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        log = self.train_critic(batch, t_env=t_env,episode_num=episode_num)
        # Get the relevant quantities
        
        actions = batch["actions"][:, :-1]
        max_t = actions.shape[1]+1
        bs = actions.shape[0]
        n_agents = actions.shape[2]
        
        terminated = batch["terminated"][:, :-1].float()
        avail_actions = batch["avail_actions"][:, :-1]
        action_dim = avail_actions.shape[-1]
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents).view(-1)#bs*ts*na
        states = batch["state"][:, :-1]

        #build q
        inputs = self.critic._build_inputs(batch, bs, max_t)
        q_vals = self.critic.forward(inputs).detach()[:, :-1]#bs,ts,na,ad

        mac_out = []
        self.mac.init_hidden(bs)
        for t in range(max_t - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time#bs,ts,na,ad

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        # mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        # mac_out[avail_actions == 0] = 0
        if th.any(th.isnan(mac_out)):
            print('there is nan!!!!!!!!!!!!')

        # Calculated baseline
        
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)#bs,ts,na
        pi = mac_out.view(-1, self.n_actions)
        baseline = th.sum(mac_out * q_vals, dim=-1).view(-1).detach()#bs*ts*na

        # Calculate policy grad with mask
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(-1)#bs*ts*na
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        # coe = self.mixer.k(states).view(-1)

        advantages = (q_taken.view(-1) - baseline)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        coma_loss = - ((advantages.detach() * log_pi_taken) * mask).sum() / mask.sum()
        
        loss = coma_loss*(1-self.args.omar_coe)

        #############omar####################
        self.omar_mu = th.cuda.FloatTensor(bs,max_t-1,n_agents, 1).zero_() + action_dim/2
        self.omar_sigma = th.cuda.FloatTensor(bs,max_t-1,n_agents, 1).zero_() + action_dim/2
        repeat_avail_action = th.repeat_interleave(avail_actions.unsqueeze(-2),repeats=self.args.omar_num_samples,dim=-2)#bs,ts,na,nsample,ad
        for iter_idx in range(self.args.omar_iters):
            dist = th.distributions.Normal(self.omar_mu, self.omar_sigma)

            cem_sampled_acs = dist.sample((self.args.omar_num_samples,)).permute(1,2,3,0,4).clamp(0, action_dim-1)
            cem_sampled_acs = th.div(cem_sampled_acs+0.5,1,rounding_mode='trunc').long()#discretize
            #bs,ts,na,nsample,1
            cem_sampled_avail = th.gather(repeat_avail_action,dim=-1,index=cem_sampled_acs)#bs,ts,na,nsample,1

            repeat_q_vals = th.repeat_interleave(q_vals.unsqueeze(-2),repeats=self.args.omar_num_samples,dim=-2)#bs,ts,na,nsample,ad
            all_pred_qvals = th.gather(repeat_q_vals,dim=-1,index=cem_sampled_acs)#bs,ts,na,nsample,1
            all_pred_qvals[cem_sampled_avail==0]=-1e10
            if th.min(th.max(q_vals, -1, keepdim=True)[0]).item()<-1e9:
                continue

            updated_mu = self.compute_softmax_acs(all_pred_qvals, cem_sampled_acs)
            self.omar_mu = updated_mu#bs,ts,na,1

            updated_sigma = th.sqrt(th.mean(((cem_sampled_acs - updated_mu.unsqueeze(-2))*cem_sampled_avail) ** 2, -2))
            self.omar_sigma = updated_sigma+0.01#bs,ts,na,1

        top_qvals, top_inds = th.topk(all_pred_qvals, 1, dim=-2)#bs,ts,na,1,1
        top_acs = th.gather(cem_sampled_acs, -2, top_inds)#bs,ts,na,1,1
        curr_pol_actions = mac_out.argmax(-1,keepdim=True)#bs,ts,na,1

        cem_qvals = top_qvals.squeeze(-1)#bs,ts,na,1
        pol_qvals = th.gather(q_vals, dim=3, index=curr_pol_actions)#bs,ts,na,1
        cem_acs = top_acs.squeeze(-1)#bs,ts,na,1
        pol_acs = curr_pol_actions#bs,ts,na,1

        candidate_qvals = th.cat([pol_qvals, cem_qvals], -1)#bs,ts,na,2
        candidate_acs = th.cat([pol_acs, cem_acs], -1)#bs,ts,na,2

        max_qvals, max_inds = th.max(candidate_qvals, -1, keepdim=True)#bs,ts,na,1

        max_acs = th.gather(candidate_acs, -1, max_inds)#bs,ts,na,1
        one_hot_max_acs = th.nn.functional.one_hot(max_acs,num_classes=action_dim).double()#bs,ts,na,ad

        omar_loss = th.nn.functional.cross_entropy(mac_out.view(-1,action_dim),one_hot_max_acs.view(-1,action_dim).detach())
        loss += self.args.omar_coe*omar_loss


        ####################################

        p_before = copy.deepcopy(self.agent_params)

        # Optimise agents
        self.agent_optimiser.zero_grad()
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        if th.any(th.isnan(grad_norm)):
            print('nan')
        self.agent_optimiser.step()

        #compute parameters sum for debugging
        p_sum = 0.
        for p in self.agent_params:
            p_sum += p.data.abs().sum().item() / 100.0
        if np.isnan(p_sum):
            print('nan')


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for key in log.keys():
                self.logger.log_stat(key, sum(log[key])/len(log[key]), t_env)
            #self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            # self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("coma_loss", coma_loss.item(), t_env)
            self.logger.log_stat("omar_loss", omar_loss.item(), t_env)
            self.logger.log_stat("agent_lr", self.args.lr, t_env)
            # self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env
        if th.any(th.isnan(omar_loss)) or th.any(th.isnan(pi)):
            print('there is nan!!!!!!!!!!!!')

    def train_critic(self, on_batch, t_env=None,episode_num=None):
        rewards = on_batch["reward"][:, :-1]
        actions = on_batch["actions"][:, :]
        max_t = actions.shape[1]
        bs = actions.shape[0]
        n_agents = actions.shape[2]
        terminated = on_batch["terminated"][:, :-1].float()
        mask = on_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = on_batch["avail_actions"][:]
        states = on_batch["state"]

        #build_target_q
        target_inputs = self.target_critic._build_inputs(on_batch, bs, max_t)
        target_q_vals = self.target_critic.forward(target_inputs).detach()
        target_q_vals_taken = th.gather(target_q_vals, dim=-1, index=actions)#bs,ts,na,1
        repeat_r = th.repeat_interleave(rewards.unsqueeze(-2),repeats=n_agents,dim=-2)#bs,ts,na,1
        repeat_terminated = th.repeat_interleave(terminated.unsqueeze(-2),repeats=n_agents,dim=-2)#bs,ts,na,1
        repeat_mask = th.repeat_interleave(mask.unsqueeze(-2),repeats=n_agents,dim=-2)#bs,ts,na,1
        #targets_taken = self.target_mixer(th.gather(target_q_vals, dim=3, index=actions).squeeze(3), states)
        target_q = build_td_lambda_targets(repeat_r, repeat_terminated, repeat_mask, target_q_vals_taken, self.n_agents, self.args.gamma, self.args.td_lambda).detach()

        inputs = self.critic._build_inputs(on_batch, bs, max_t)

        #train critic
        log={}
        for t in range(max_t - 1):
            mask_t = repeat_mask[:, t:t+1]
            if mask_t.sum() < 0.5:
                continue
            q_vals = self.critic.forward(inputs[:, t:t+1])#bs,1,na,ad
            q_vals_taken = th.gather(q_vals, -1, index=actions[:, t:t+1])
            target_q_t = target_q[:, t:t+1].detach()
            q_err = (q_vals_taken - target_q_t) * mask_t
            critic_loss = (q_err ** 2).sum() / mask_t.sum()
            negative_sampling = th.logsumexp(q_vals,dim=-1,keepdim=True)#bs,1,na,1
            # negative_sampling = mac_out.max(dim=-1)[0].mean()
            dataset_expec = q_vals_taken#bs,1,na
            cql_loss = self.args.cql_alpha * ((negative_sampling-dataset_expec)* mask_t).sum()/mask_t.sum()
            critic_loss += cql_loss
            self.agent_optimiser.zero_grad()
            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            if (t == 0):
                log["critic_loss"]=[]
                log["critic_grad_norm"]=[]
                mask_elems = mask_t.sum().item()
                log["td_error_abs"]=[]
                log["target_mean"]=[]
                log["q_taken_mean"]=[]


            log["critic_loss"].append(critic_loss.item())
            log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            log["td_error_abs"].append((q_err.abs().sum().item() / mask_elems))
            log["target_mean"].append((target_q_t * mask_t).sum().item() / mask_elems)
            log["q_taken_mean"].append((q_vals_taken * mask_t).sum().item() / mask_elems)

        # if t_env - self.log_stats_t >= self.args.learner_log_interval:
        #     for key in log.keys():
        #         self.logger.log_stat(key, sum(log[key])/len(log[key]), t_env)
        #     self.log_stats_t = t_env
        #update target network
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        return log
    def compute_softmax_acs(self, q_vals, acs):
        max_q_vals = th.max(q_vals, -1, keepdim=True)[0]#bs,ts,na,1,1
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = th.exp(norm_q_vals)#bs,ts,na,nsample,1
        a_mult_e = acs * e_beta_normQ#bs,ts,na,nsample,1
        numerators = a_mult_e
        denominators = e_beta_normQ

        sum_numerators = th.sum(numerators, -2)
        sum_denominators = th.sum(denominators, -2)

        softmax_acs = sum_numerators / sum_denominators#bs,ts,na,1

        return softmax_acs

    def build_exp_q(self, target_q_vals, mac_out, states):
        target_exp_q_vals = th.sum(target_q_vals * mac_out, dim=3)
        # target_exp_q_vals = self.target_mixer.forward(target_exp_q_vals, states)
        return target_exp_q_vals

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        # self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        # self.mixer.cuda()
        self.target_critic.cuda()
        # self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        # th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        # th.save(self.mixer_optimiser.state_dict(), "{}/mixer_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
       # self.target_critic.load_state_dict(self.critic.agent.state_dict())
        # self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
        # self.mixer_optimiser.load_state_dict(th.load("{}/mixer_opt.th".format(path), map_location=lambda storage, loc: storage))