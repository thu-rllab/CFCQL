import torch
import torch.nn.functional as F
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients
from utils.agents import TD3MultiAgent, DDPGAgent
import itertools
import numpy as np
from utils.tb_log import log_and_print
from utils.noise import action_noise
import random

MSELoss = torch.nn.MSELoss()

class MATD3CC(object):
    def __init__(
        self, 
        agent_init_params, 
        alg_types, 
        adv_init_params=None,
        gamma=0.95, 
        tau=0.01, 
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        gaussian_noise_std=None, 
        agent_max_actions=None, 
        cql=False, cql_alpha=None, lse_temp=1.0, num_sampled_actions=None, cql_sample_noise_level=0.2,
        omar=None, omar_coe=None,
        omar_mu=None, omar_sigma=None, omar_num_samples=None, omar_num_elites=None, omar_iters=None, batch_size=None, 
        env_id=None,
        **kwargs
    ):
        self.env_id = env_id
        self.is_mamujoco = True if self.env_id == 'HalfCheetah-v2' else False

        assert (ma == agent_max_actions[0] for ma in agent_max_actions)
        self.max_action = agent_max_actions[0]
        self.min_action = -self.max_action

        self.nagents = len(alg_types)
        self.alg_types = alg_types

        self.agent = TD3MultiAgent(
            lr=lr, 
            discrete_action=discrete_action, 
            hidden_dim=hidden_dim, 
            gaussian_noise_std=gaussian_noise_std, 
            **agent_init_params
        )

        if self.env_id in ['simple_tag', 'simple_world']:
            self.num_predators = len(self.agent.policys)
            self.num_preys = len(adv_init_params)

            self.preys = [DDPGAgent(lr=lr, discrete_action=discrete_action, hidden_dim=hidden_dim, **params) for params in adv_init_params]

        self.niter = 0

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action

        self.pol_dev, self.trgt_pol_dev, self.critic_dev, self.trgt_critic_dev = 'cpu', 'cpu', 'cpu', 'cpu' 

        self.omar = omar
        if self.omar:
            self.omar_coe = omar_coe

            self.omar_iters = omar_iters
            self.omar_num_samples = omar_num_samples
            self.init_omar_mu, self.init_omar_sigma = omar_mu, omar_sigma
            self.omar_mu = torch.cuda.FloatTensor(batch_size, self.agent_init_params['num_out_pol'][0]).zero_() + self.init_omar_mu
            self.omar_sigma = torch.cuda.FloatTensor(batch_size, self.agent_init_params['num_out_pol'][0]).zero_() + self.init_omar_sigma
            self.omar_num_elites = omar_num_elites

        self.cql = cql
        if self.cql:
            self.cql_alpha = cql_alpha
            self.cql_sample_noise_level = cql_sample_noise_level
            self.lse_temp = lse_temp
            self.num_sampled_actions = num_sampled_actions
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.cf_weight=False #delayed to vae loader
        self.sample_action_class =[int(a) for a in self.sample_action_class.split("-")]

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        if self.env_id in ['simple_world', 'simple_tag']:
            prey_obs = [obs for i,obs in zip(range(len(observations)), observations) if i >= self.num_predators]
            observations = [obs for i,obs in zip(range(len(observations)), observations) if i < self.num_predators]    
        res = self.agent.step(observations)
        if self.env_id in ['simple_world', 'simple_tag']:
            for i, obs in zip(range(self.num_preys), prey_obs):
                res.append(self.preys[i].step(obs, explore=False))
        return res

    def calc_gaussian_pdf(self, samples, mu=0):
        pdfs = 1 / (self.cql_sample_noise_level * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * self.cql_sample_noise_level**2) )
        pdf = torch.prod(pdfs, dim=-1)
        return pdf

    def get_policy_actions(self, states, networks, ns=None):
        if ns:
            num_sampled_actions = ns
        else:
            num_sampled_actions = self.num_sampled_actions
        noisy_actions, random_noises_log_pis = [], []
        for i in range(self.nagents):
            action = networks[i](states[i])

            formatted_action = action.unsqueeze(1).repeat(1, num_sampled_actions, 1).view(action.shape[0] * num_sampled_actions, action.shape[1]) #bs*ns, ac

            random_noises = torch.FloatTensor(formatted_action.shape[0], formatted_action.shape[1])

            random_noises = random_noises.normal_() * self.cql_sample_noise_level
            random_noises_pi = self.calc_gaussian_pdf(random_noises).view(action.shape[0], num_sampled_actions, 1).cuda()
            random_noises_log_pi = torch.log(random_noises_pi)
            random_noises = random_noises.cuda()

            noisy_action = (formatted_action + random_noises).clamp(-self.max_action, self.max_action)
            noisy_actions.append(noisy_action)
            random_noises_log_pis.append(random_noises_log_pi)

        return noisy_actions, random_noises_log_pis #[na(list), bs*ns, ac]|[na(list), bs, ns, 1]

    def compute_softmax_acs(self, q_vals, acs):
        max_q_vals = torch.max(q_vals, 1, keepdim=True)[0]
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(norm_q_vals)
        a_mult_e = acs * e_beta_normQ
        numerators = a_mult_e
        denominators = e_beta_normQ

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        softmax_acs = sum_numerators / sum_denominators

        return softmax_acs
    def get_cf_weight(self, states, obs):
        with torch.no_grad():
            actions = []
            for i in range(self.nagents):
                action = self.agent.policys[i](obs[i])
                actions.append(action)
            log_mus = self.vae.importance_sampling_estimator(states, torch.cat(actions, dim=1), 0.5, self.nagents, num_samples=20)
            log_pi = actions[0].shape[1] * np.log(1 / (self.cql_sample_noise_level * np.sqrt(2 * np.pi)))
            log_ratio = torch.cat([log_pi - log_mu.unsqueeze(1) for log_mu in log_mus],dim=-1) * self.cf_tau
            log_ratio = F.softmin(log_ratio,dim=1)
        return log_ratio.permute(1,0).unsqueeze(2)


    def update(self, sample, t, parallel=False, only_critic=False):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next observations, and episode end masks) 
                    sampled randomly from the replay buffer. Each is a list with entries corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch): If passed in, important quantities will be logged
        """
        dic = {} # dictionary for print and log
        if self.is_mamujoco:
            states, obs, acs, rews, next_states, next_obs, dones, next_acs = sample
            states = states[0]
            next_states = next_states[0]
        else:
            obs, acs, rews, next_obs, dones, next_acs = sample
            states = torch.cat([ob.unsqueeze(1) for ob in obs], dim=1).reshape(obs[0].shape[0], obs[0].shape[1]*len(obs)) #bs, ds  
            next_states = torch.cat([ob.unsqueeze(1) for ob in next_obs], dim=1).reshape(obs[0].shape[0], obs[0].shape[1]*len(obs)) #bs, ds  
            

        self.agent.critic_optimizer.zero_grad()
        trgt_obs_acs = [next_states]
        for i in range(self.nagents):
            if self.action_noise_scale > 0:
                trgt_obs_acs.append(action_noise(self.agent.target_policys[i](next_obs[i]), sigma=self.action_noise_scale))
            else:
                trgt_obs_acs.append(self.agent.target_policys[i](next_obs[i]))
        if self.cf_target:
            all_trgt_vf_in = []
            for i in range(self.nagents):
                toa_i = [next_states] + next_acs
                toa_i[i+1] = trgt_obs_acs[i+1]
                all_trgt_vf_in.append(torch.cat(toa_i, dim=1))
            all_trgt_vf_in = torch.cat([toa.unsqueeze(0) for toa in all_trgt_vf_in], dim=0) #na, bs, s+acda
            next_q_value1, next_q_value2 = self.agent.target_critic(all_trgt_vf_in.reshape(self.nagents*next_acs[0].shape[0], -1)) 
            next_q_value1 = next_q_value1.reshape(self.nagents,next_acs[0].shape[0],1)
            next_q_value2 = next_q_value2.reshape(self.nagents,next_acs[0].shape[0],1)
            next_q_value1 = torch.mean(next_q_value1, dim=0)
            next_q_value2 = torch.mean(next_q_value2, dim=0)
            next_q_value = torch.min(next_q_value1, next_q_value2)
        else:
            trgt_vf_in = torch.cat(trgt_obs_acs, dim=1)
            
            next_q_value1, next_q_value2 = self.agent.target_critic(trgt_vf_in) 
            next_q_value = torch.min(next_q_value1, next_q_value2)

        target_value = rews[0].view(-1, 1) + self.gamma * next_q_value * (1 - dones[0].view(-1, 1)) #bs,1

        vf_in = torch.cat([states]+acs, dim=1)
        
        actual_value1, actual_value2 = self.agent.critic(vf_in) 

        vf_loss = MSELoss(actual_value1, target_value.detach()) + MSELoss(actual_value2, target_value.detach())
        
        if self.cql:
            if self.is_mamujoco:
                if self.cf_cql:
                    bs,  acd = states.shape[0], acs[0].shape[1]
                    formatted_states = states.unsqueeze(1).unsqueeze(0).repeat(self.nagents, 1, self.num_sampled_actions, 1).view(self.nagents*bs*self.num_sampled_actions, states.shape[1]) #na*bs*ns, state_dim
                    random_actions = torch.cat([(torch.FloatTensor(acs[i].shape[0] * self.num_sampled_actions, acs[i].shape[1]).uniform_(-1, 1)).cuda().unsqueeze(0) for i in range(self.nagents)], dim=0)#[na, bs*ns, ac]
                    cf_random_actions = torch.cat([action_noise(a,sigma=self.beta_action_noise, noise_type='uniform').unsqueeze(0) for a in acs], dim=0).unsqueeze(2).unsqueeze(0).repeat(self.nagents, 1, 1, self.num_sampled_actions, 1).view(self.nagents, self.nagents, bs*self.num_sampled_actions, acs[0].shape[1]) #na, na, bs*ns, action_dim
                    curr_actions, curr_action_log_pi = self.get_policy_actions(obs, self.agent.policys) #[na(list), bs*ns, ac]|[na(list), bs, ns, 1]    
                    new_curr_actions, new_curr_action_log_pi = self.get_policy_actions(next_obs, self.agent.policys)
                    cf_curr_actions = torch.cat([action_noise(a,sigma=self.beta_action_noise, noise_type='uniform').unsqueeze(0) for a in acs], dim=0).unsqueeze(2).unsqueeze(0).repeat(self.nagents, 1, 1, self.num_sampled_actions, 1).view(self.nagents, self.nagents, bs*self.num_sampled_actions, acs[0].shape[1]) #na, na, bs*ns, action_dim
                    cf_new_curr_actions = torch.cat([action_noise(a,sigma=self.beta_action_noise, noise_type='uniform').unsqueeze(0) for a in next_acs], dim=0).unsqueeze(2).unsqueeze(0).repeat(self.nagents, 1, 1, self.num_sampled_actions, 1).view(self.nagents, self.nagents, bs*self.num_sampled_actions, next_acs[0].shape[1]) #na, na, bs*ns, action_dim
                    for i in range(self.nagents):
                        cf_random_actions[i, i] = random_actions[i] #[na, na, bs*ns, ac]
                        cf_curr_actions[i, i] = curr_actions[i]
                        cf_new_curr_actions[i, i] = new_curr_actions[i]
                    cf_random_action_log_pi = np.log(0.5 ** random_actions.shape[-1]) 
                    cf_curr_action_log_pi = torch.cat([c.unsqueeze(0) for c in curr_action_log_pi], dim=0) #[na, bs, ns, 1]
                    cf_new_curr_action_log_pi = torch.cat([c.unsqueeze(0) for c in new_curr_action_log_pi], dim=0) #[na, bs, ns, 1]
                    random_vf_in = torch.cat([formatted_states, cf_random_actions.permute(0,2,1,3).reshape(self.nagents*bs*self.num_sampled_actions, acd*self.nagents)], dim=1)
                    curr_vf_in = torch.cat([formatted_states, cf_curr_actions.permute(0,2,1,3).reshape(self.nagents*bs*self.num_sampled_actions, acd*self.nagents)], dim=1)
                    new_curr_vf_in = torch.cat([formatted_states, cf_new_curr_actions.permute(0,2,1,3).reshape(self.nagents*bs*self.num_sampled_actions, acd*self.nagents)], dim=1)
                    
                    random_Q1, random_Q2 = self.agent.critic(random_vf_in) #na, bs*ns
                    curr_Q1, curr_Q2 = self.agent.critic(curr_vf_in)
                    new_curr_Q1, new_curr_Q2 = self.agent.critic(new_curr_vf_in)

                    random_Q1, random_Q2 = random_Q1.view(self.nagents, bs, self.num_sampled_actions, 1), random_Q2.view(self.nagents, bs, self.num_sampled_actions, 1)
                    curr_Q1, curr_Q2 = curr_Q1.view(self.nagents, bs, self.num_sampled_actions, 1), curr_Q2.view(self.nagents, bs, self.num_sampled_actions, 1)
                    new_curr_Q1, new_curr_Q2 = new_curr_Q1.view(self.nagents, bs, self.num_sampled_actions, 1), new_curr_Q2.view(self.nagents, bs, self.num_sampled_actions, 1)
                    cat_q1 = torch.zeros([self.nagents, bs, 0, 1]).to(random_Q1.device)
                    cat_q2 = torch.zeros([self.nagents, bs, 0, 1]).to(random_Q1.device)
                    Q_list =[[random_Q1, random_Q2], [curr_Q1, curr_Q2], [new_curr_Q1, new_curr_Q2]]
                    pi_list = [cf_random_action_log_pi, cf_new_curr_action_log_pi.detach(), cf_curr_action_log_pi.detach()]
                    for i in range(3):
                        if self.sample_action_class[i] == 1:
                            cat_q1 = torch.cat([cat_q1, Q_list[i][0]-self.soft_q*pi_list[i]], dim=2)
                            cat_q2 = torch.cat([cat_q2, Q_list[i][1]-self.soft_q*pi_list[i]], dim=2)
                    policy_qvals1 = torch.logsumexp(cat_q1 / self.lse_temp, dim=2) * self.lse_temp
                    policy_qvals2 = torch.logsumexp(cat_q2 / self.lse_temp, dim=2) * self.lse_temp #na, bs, 1
                    if self.cf_weight:
                        weight = self.get_cf_weight(states, obs)
                    else:
                        weight = torch.tensor([1/self.nagents]*self.nagents).to(policy_qvals1.device).view(self.nagents, 1,1)
                    policy_qvals1 = (policy_qvals1*weight).sum(0) #bs, 1
                    policy_qvals2 = (policy_qvals2*weight).sum(0) #bs, 1
                    if t % self.logging_interval == 0 and not self.no_log:
                        dic.update({"cf_random_action_log_pi":cf_random_action_log_pi.mean().item()})
                        dic.update({"cf_new_curr_action_log_pi":cf_new_curr_action_log_pi.mean().item()})
                        dic.update({"cf_curr_action_log_pi":cf_curr_action_log_pi.mean().item()})
                else:
                    formatted_obs = states.unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(-1, states.shape[1])
                    random_actions = [(torch.FloatTensor(acs[i].shape[0] * self.num_sampled_actions, acs[i].shape[1]).uniform_(-1, 1)).cuda() for i in range(self.nagents)]#[na(list), bs*ns, ac]
                    random_action_log_pi = np.log(0.5 ** sum([random_actions[i].shape[-1] for i in range(self.nagents)]))
                    curr_actions, curr_action_log_pi = self.get_policy_actions(obs, self.agent.policys)
                    curr_action_log_pi = sum(curr_action_log_pi)
                    new_curr_actions, new_curr_action_log_pi = self.get_policy_actions(next_obs, self.agent.policys)
                    new_curr_action_log_pi = sum(new_curr_action_log_pi)

                    random_vf_in = torch.cat([formatted_obs]+random_actions, dim=1)
                    curr_vf_in = torch.cat([formatted_obs]+curr_actions, dim=1)
                    new_curr_vf_in = torch.cat([formatted_obs]+ new_curr_actions, dim=1)

                    random_Q1, random_Q2 = self.agent.critic(random_vf_in)
                    curr_Q1, curr_Q2 = self.agent.critic(curr_vf_in)
                    new_curr_Q1, new_curr_Q2 = self.agent.critic(new_curr_vf_in)

                    random_Q1, random_Q2 = random_Q1.view(states.shape[0], self.num_sampled_actions, 1), random_Q2.view(states.shape[0], self.num_sampled_actions, 1)
                    curr_Q1, curr_Q2 = curr_Q1.view(states.shape[0], self.num_sampled_actions, 1), curr_Q2.view(states.shape[0], self.num_sampled_actions, 1)
                    new_curr_Q1, new_curr_Q2 = new_curr_Q1.view(states.shape[0], self.num_sampled_actions, 1), new_curr_Q2.view(states.shape[0], self.num_sampled_actions, 1)
                    if self.soft_q:
                        cat_q1 = torch.cat([random_Q1 - random_action_log_pi, new_curr_Q1 - new_curr_action_log_pi, curr_Q1 - curr_action_log_pi], 1)
                        cat_q2 = torch.cat([random_Q2 - random_action_log_pi, new_curr_Q2 - new_curr_action_log_pi, curr_Q2 - curr_action_log_pi], 1)
                    else:
                        cat_q1 = torch.cat([random_Q1, new_curr_Q1, curr_Q1], 1)
                        cat_q2 = torch.cat([random_Q2, new_curr_Q2, curr_Q2], 1)
                    policy_qvals1 = torch.logsumexp(cat_q1 / self.lse_temp, dim=1) * self.lse_temp
                    policy_qvals2 = torch.logsumexp(cat_q2 / self.lse_temp, dim=1) * self.lse_temp
                        
            else:
                if self.cf_cql:
                    bs,  acd = states.shape[0], acs[0].shape[1]

                    formatted_states = states.unsqueeze(1).unsqueeze(0).repeat(self.nagents, 1, self.num_sampled_actions, 1).view(self.nagents*bs*self.num_sampled_actions, states.shape[1]) #na*bs*ns, state_dim
                    random_actions = torch.cat([(torch.FloatTensor(acs[i].shape[0] * self.num_sampled_actions, acs[i].shape[1]).uniform_(-1, 1)).cuda().unsqueeze(0) for i in range(self.nagents)], dim=0)#[na, bs*ns, ac]
                    cf_random_actions = torch.cat([a.unsqueeze(0) for a in acs], dim=0).unsqueeze(2).unsqueeze(0).repeat(self.nagents, 1, 1, self.num_sampled_actions, 1).view(self.nagents, self.nagents, bs*self.num_sampled_actions, acs[0].shape[1]) #na, na, bs*ns, action_dim
                    for i in range(self.nagents):
                        cf_random_actions[i, i] = random_actions[i] #[na, na, bs*ns, ac]
                    cf_random_action_log_pi = np.log(0.5 ** random_actions[i].shape[-1]) 
                    random_vf_in = torch.cat([formatted_states, cf_random_actions.permute(0,2,1,3).reshape(self.nagents*bs*self.num_sampled_actions, acd*self.nagents)], dim=1)
                    random_Q1, random_Q2 = self.agent.critic(random_vf_in) #na, bs*ns
                    random_Q1, random_Q2 = random_Q1.view(self.nagents, bs, self.num_sampled_actions, 1), random_Q2.view(self.nagents, bs, self.num_sampled_actions, 1)
                    if self.soft_q:
                        random_Q1 = random_Q1 - cf_random_action_log_pi
                        random_Q2 = random_Q2 - cf_random_action_log_pi
                    policy_qvals1 = torch.logsumexp(random_Q1 / self.lse_temp, dim=2) * self.lse_temp
                    policy_qvals2 = torch.logsumexp(random_Q2 / self.lse_temp, dim=2) * self.lse_temp #na, bs, 1
                    if self.cf_weight:
                        weight = self.get_cf_weight(states, obs)
                    else:
                        weight = torch.tensor([1/self.nagents]*self.nagents).to(policy_qvals1.device).view(self.nagents, 1,1)
                    policy_qvals1 = (policy_qvals1*weight).sum(0) #bs, 1
                    policy_qvals2 = (policy_qvals2*weight).sum(0) #bs, 1
                    if t % self.logging_interval == 0 and not self.no_log:
                        dic.update({"cf_random_action_log_pi":cf_random_action_log_pi.mean().item()})
                else:
                    formatted_obs = states.unsqueeze(1).repeat(1, self.num_sampled_actions, 1).view(-1, states.shape[1])
                    random_actions = [(torch.FloatTensor(acs[i].shape[0] * self.num_sampled_actions, acs[i].shape[1]).uniform_(-1, 1)).cuda() for i in range(self.nagents)]#[na(list), bs*ns, ac]
                    random_action_log_pi = np.log(0.5 ** sum([random_actions[i].shape[-1] for i in range(self.nagents)]))
                    random_vf_in = torch.cat([formatted_obs]+random_actions, dim=1)
                    random_Q1, random_Q2 = self.agent.critic(random_vf_in)
                    random_Q1, random_Q2 = random_Q1.view(states.shape[0], self.num_sampled_actions, 1), random_Q2.view(states.shape[0], self.num_sampled_actions, 1)
                    if self.soft_q:
                        random_Q1 = random_Q1 - random_action_log_pi
                        random_Q2 = random_Q2 - random_action_log_pi
                    policy_qvals1 = torch.logsumexp(random_Q1 / self.lse_temp, dim=1) * self.lse_temp
                    policy_qvals2 = torch.logsumexp(random_Q2 / self.lse_temp, dim=1) * self.lse_temp

            dataset_q_vals1 = actual_value1
            dataset_q_vals2 = actual_value2

            cql_term1 = (policy_qvals1 - dataset_q_vals1).mean()
            cql_term2 = (policy_qvals2 - dataset_q_vals2).mean()
            
            cql_term = cql_term1 + cql_term2
            vf_loss += self.cql_alpha * cql_term
            if t % self.logging_interval == 0 and not self.no_log:
                dic.update({"policy_qvals":(policy_qvals1+policy_qvals2).mean().item()/2})
                dic.update({"dataset_q_vals":(dataset_q_vals1+dataset_q_vals2).mean().item()/2})
                if self.is_mamujoco:
                    dic.update({"raw_policy_qvals":(cat_q1+cat_q2).mean().item()/2})
                else:
                    dic.update({"raw_policy_qvals":(random_Q1+random_Q2).mean().item()/2})
        vf_loss.backward()
        if parallel:
            average_gradients(self.agent.critic)
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
        self.agent.critic_optimizer.step()
        if only_critic:
            if t % self.logging_interval == 0 and not self.no_log:
                dic.update({"vf_loss":vf_loss.item()})
                if self.cql:
                    dic.update({"cql_term":(self.cql_alpha * cql_term).item()})
                log_and_print(list(dic.keys()), list(dic.values()), t, multi=True)
            return 
        self.agent.critic_optimizer.zero_grad()

        curr_pol_vf_ins = [states]
        curr_pol_outs = []
        for i in range(self.nagents):
            self.agent.policy_optimizers[i].zero_grad()
            curr_pol_out = self.agent.policys[i](obs[i])
            curr_pol_vf_ins.append(curr_pol_out)
            curr_pol_outs.append(curr_pol_out)
        if self.cf_omar:
            all_cur_pol_out = torch.cat([c.unsqueeze(0) for c in curr_pol_outs], dim=0) #[na, bs, acd]
        else:
            all_cur_pol_out = torch.cat(curr_pol_outs, dim=1).unsqueeze(0).repeat(self.nagents, 1, 1) #na, bs, acda
        if self.cf_pol:
            all_vf_in = []
            for i in range(self.nagents):
                cf_action = [a for a in acs]
                cf_action[i] = curr_pol_outs[i]
                single_vf_in = torch.cat([states]+cf_action, dim=1)
                all_vf_in.append(single_vf_in)
            all_vf_in = torch.cat([a.unsqueeze(0) for a in all_vf_in], dim=0) #na, bs, s+acda
            vf_in = all_vf_in.reshape(self.nagents*acs[0].shape[0], -1)
        else:
            vf_in = torch.cat(curr_pol_vf_ins, dim=1)

        if self.omar:
            if self.cf_pol:
                pred_qvals = self.agent.critic.Q1(vf_in)
                pred_qvals = pred_qvals.reshape(self.nagents, acs[0].shape[0], 1)
            else:
                pred_qvals = self.agent.critic.Q1(vf_in) #[bs, 1]
                pred_qvals = pred_qvals.unsqueeze(0).repeat(self.nagents, 1, 1)#[na, bs, 1]
            if self.is_mamujoco:
                na, bs, acd, std = self.nagents, acs[0].shape[0], acs[0].shape[1], states.shape[1]
                if not self.cf_omar:
                    acd = sum([aci.shape[1] for aci in acs])
                self.omar_mu = torch.cuda.FloatTensor(na, bs, acd).zero_() + self.init_omar_mu
                self.omar_sigma = torch.cuda.FloatTensor(na, bs, acd).zero_() + self.init_omar_sigma
                formatted_states = states.unsqueeze(1).unsqueeze(0).repeat(na, 1, self.omar_num_samples, 1).view(na*bs*self.omar_num_samples, std) #na*bs*omar_sampels, std
                last_top_k_qvals, last_elite_acs = None, None
                for iter_idx in range(self.omar_iters):
                    dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma+1e-10)
                    if self.cf_omar:
                        cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1,2,0,3).clamp(-self.max_action, self.max_action) #na, bs, omar_samples, acd
                        ori_actions = torch.cat([a.unsqueeze(0) for a in acs], dim=0).unsqueeze(2).unsqueeze(0).repeat(self.nagents, 1, 1, self.omar_num_samples, 1) #na, na, bs, omar_samples, acd
                        for i in range(na):
                            ori_actions[i,i]=cem_sampled_acs[i]
                        formatted_cem_sampled_acs = ori_actions.permute(0,2,3,1,4).reshape(na*bs*self.omar_num_samples, na*acd)
                    else:
                        cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1,2,0,3) #na, bs, omar_samples, acd
                        formatted_cem_sampled_acs = cem_sampled_acs.reshape(na*bs*self.omar_num_samples, acd)
                    vf_in = torch.cat((formatted_states, formatted_cem_sampled_acs), dim=1)
                    all_pred_qvals = self.agent.critic.Q1(vf_in).view(na, bs, self.omar_num_samples, 1)

                    if iter_idx > 0:
                        all_pred_qvals = torch.cat((all_pred_qvals, last_top_k_qvals), dim=2)
                        cem_sampled_acs = torch.cat((cem_sampled_acs, last_elite_acs), dim=2)

                    top_k_qvals, top_k_inds = torch.topk(all_pred_qvals, self.omar_num_elites, dim=2) #na, bs, omar_num_elites, 1
                    elite_ac_inds = top_k_inds.repeat(1, 1, 1, acd) #na, bs, omar_num_elites, acd
                    elite_acs = torch.gather(cem_sampled_acs, 2, elite_ac_inds) #na, bs, omar_num_elites, acd

                    last_top_k_qvals, last_elite_acs = top_k_qvals, elite_acs

                    updated_mu = torch.mean(elite_acs, dim=2)
                    updated_sigma = torch.std(elite_acs, dim=2)

                    self.omar_mu = updated_mu
                    self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals, 1, dim=2) #na, bs, 1, 1
                top_ac_inds = top_inds.repeat(1, 1, 1, acd)
                top_acs = torch.gather(cem_sampled_acs, 2, top_ac_inds) #na, bs, 1, acd

                cem_qvals = top_qvals
                pol_qvals = pred_qvals.unsqueeze(2) #na,bs, 1, 1
                cem_acs = top_acs
                pol_acs =  all_cur_pol_out.unsqueeze(2)#na, bs, 1, acd

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 2) #na, bs, 2, 1
                candidate_acs = torch.cat([pol_acs, cem_acs], 2) #na, bs, 2, acd

                max_qvals, max_inds = torch.max(candidate_qvals, 2, keepdim=True) #na, bs, 1, 1
                max_ac_inds = max_inds.repeat(1, 1, 1, acd) #na, bs, 1, acd

                max_acs = torch.gather(candidate_acs, 2, max_ac_inds).squeeze(2) #na, bs, acd
            else:
                na, bs, acd, std = self.nagents, acs[0].shape[0], acs[0].shape[1], states.shape[1]
                if not self.cf_omar:
                    acd = sum([aci.shape[1] for aci in acs])
                self.omar_mu = torch.cuda.FloatTensor(na, bs, acd).zero_() + self.init_omar_mu
                self.omar_sigma = torch.cuda.FloatTensor(na, bs, acd).zero_() + self.init_omar_sigma
                formatted_states = states.unsqueeze(1).unsqueeze(0).repeat(na, 1, self.omar_num_samples, 1).view(na*bs*self.omar_num_samples, std) #na*bs*omar_sampels, std
                for iter_idx in range(self.omar_iters):
                    dist = torch.distributions.Normal(self.omar_mu, self.omar_sigma+1e-10)
                    if self.cf_omar:
                        cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1,2,0,3).clamp(-self.max_action, self.max_action) #na, bs, omar_samples, acd
                        ori_actions = torch.cat([a.unsqueeze(0) for a in acs], dim=0).unsqueeze(2).unsqueeze(0).repeat(self.nagents, 1, 1, self.omar_num_samples, 1) #na, na, bs, omar_samples, acd
                        for i in range(na):
                            ori_actions[i,i]=cem_sampled_acs[i]
                        formatted_cem_sampled_acs = ori_actions.permute(0,2,3,1,4).reshape(na*bs*self.omar_num_samples, na*acd)
                    else:
                        cem_sampled_acs = dist.sample((self.omar_num_samples,)).detach().permute(1,2,0,3) #na, bs, omar_samples, acd
                        formatted_cem_sampled_acs = cem_sampled_acs.reshape(na*bs*self.omar_num_samples, acd)
                    vf_in = torch.cat((formatted_states, formatted_cem_sampled_acs), dim=1)
                    all_pred_qvals = self.agent.critic.Q1(vf_in)
                    all_pred_qvals = all_pred_qvals.view(na*bs, self.omar_num_samples, 1)

                    updated_mu = self.compute_softmax_acs(all_pred_qvals, cem_sampled_acs.view(na*bs, self.omar_num_samples, acd))
                    self.omar_mu = updated_mu.reshape(na, bs, acd)

                    updated_sigma = torch.sqrt(torch.mean((cem_sampled_acs - self.omar_mu.unsqueeze(2)) ** 2, 2))
                    self.omar_sigma = updated_sigma

                top_qvals, top_inds = torch.topk(all_pred_qvals.view(na, bs, self.omar_num_samples, 1), 1, dim=2) #na, bs, 1, 1
                top_ac_inds = top_inds.repeat(1, 1, 1, acd)
                top_acs = torch.gather(cem_sampled_acs, 2, top_ac_inds) #na, bs, 1, acd

                cem_qvals = top_qvals
                pol_qvals = pred_qvals.unsqueeze(2) #na,bs, 1, 1
                cem_acs = top_acs
                pol_acs =  all_cur_pol_out.unsqueeze(2)#na, bs, 1, acd

                candidate_qvals = torch.cat([pol_qvals, cem_qvals], 2) #na, bs, 2, 1
                candidate_acs = torch.cat([pol_acs, cem_acs], 2) #na, bs, 2, acd

                max_qvals, max_inds = torch.max(candidate_qvals, 2, keepdim=True) #na, bs, 1, 1
                max_ac_inds = max_inds.repeat(1, 1, 1, acd) #na, bs, 1, acd

                max_acs = torch.gather(candidate_acs, 2, max_ac_inds).squeeze(2) #na, bs, acd
 
                        
            mimic_acs = max_acs.detach() #na, bs, acd
            
            mimic_term = F.mse_loss(all_cur_pol_out, mimic_acs)

            pol_loss = self.omar_coe * mimic_term - (1 - self.omar_coe) * pred_qvals.mean()
        else:
            pol_loss = -self.agent.critic.Q1(vf_in).mean()
        if not self.no_action_reg:
            pol_loss += sum((curr_pol_out ** 2).mean() * 1e-3 for curr_pol_out in curr_pol_outs)
        bc_loss = ((torch.cat(curr_pol_outs, dim=1) - torch.cat(acs, dim=1))**2).mean()
        pol_bc_loss = self.bc_tau * bc_loss + (1-self.bc_tau) * pol_loss
        pol_bc_loss.backward()
        for i in range(self.nagents):
            if parallel:
                average_gradients(self.agent.policys[i])
            torch.nn.utils.clip_grad_norm_(self.agent.policys[i].parameters(), 0.5)
            self.agent.policy_optimizers[i].step()
        if t % self.logging_interval == 0 and not self.no_log:
            dic.update({"bc_loss":self.bc_tau * bc_loss.item()})
            dic.update({"pol_loss":(1-self.bc_tau) * pol_loss.item()})
            dic.update({"vf_loss":vf_loss.item()})
            if self.cql:
                dic.update({"cql_term":(self.cql_alpha * cql_term).item()})
            if self.omar:
                dic.update({"mimic_term":(self.omar_coe * mimic_term).item()})
                dic.update({"pred_qvals":((1 - self.omar_coe) * pred_qvals.mean()).item()})
            log_and_print(list(dic.keys()), list(dic.values()), t, multi=True)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been performed for each agent)
        """
        soft_update(self.agent.target_critic, self.agent.critic, self.tau)
        for i in range(self.nagents):
            soft_update(self.agent.target_policys[i], self.agent.policys[i], self.tau) 
        self.niter += 1

    def prep_training(self, device='gpu'):
        for i in range(self.nagents):
            self.agent.policys[i].train()
            self.agent.target_policys[i].train()
        self.agent.critic.train()
        self.agent.target_critic.train()

        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for i in range(self.nagents):
                self.agent.policys[i] = fn(self.agent.policys[i])
                self.agent.target_policys[i] = fn(self.agent.target_policys[i])

            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.policy = fn(p.policy)

            self.pol_dev = device
        if not self.critic_dev == device:
            self.agent.critic = fn(self.agent.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device: 
            for i in range(self.nagents):
                self.agent.target_policys[i] = fn(self.agent.target_policys[i])
            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.target_policy = fn(p.target_policy)

            self.trgt_pol_dev = device 
        if not self.trgt_critic_dev == device:
            self.agent.target_critic = fn(self.agent.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for p in self.agent.policys:
            p.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for i in range(self.nagents):
                self.agent.policys[i] = fn(self.agent.policys[i])
            if self.env_id in ['simple_tag', 'simple_world']:
                for p in self.preys:
                    p.policy = fn(p.policy)
            self.pol_dev = device

    @classmethod
    def init_from_env(cls, env, env_id, data_type, env_info=None, agent_alg="td3", adversary_alg="ddpg", gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, 
                    cql=False, batch_size=None, lse_temp=None, num_sampled_actions=None, gaussian_noise_std=None, omar=None, omar_mu=None, omar_sigma=None, 
                    omar_num_samples=None, omar_num_elites=None, omar_iters=None, **kwargs):
        """
        Instantiate instance of this class from multi-agent environment
        """
        if env_id in ['simple_tag', 'simple_world']:
            alg_types = [agent_alg for atype in env.agent_types if atype == 'adversary']
        elif env_id in ['simple_spread']:
            alg_types = [agent_alg for atype in env.agent_types]
        elif env_id in ['HalfCheetah-v2']:
            alg_types = [agent_alg for _ in range(env_info['n_agents'])]

        agent_init_params = {}
        all_n_actions = []
        all_n_obs = []
        agent_max_actions = []
        adv_init_params = []

        if env_id == 'HalfCheetah-v2':
            for agent_idx, algtype in zip(range(len(alg_types)), alg_types):
                acsp = env_info['action_spaces'][agent_idx]
                num_out_pol = acsp.shape[0]
                agent_max_actions.append(acsp.high[0])
                all_n_actions.append(acsp.shape[0])
            agent_init_params={'num_in_pol': env_info['obs_shape'], 'num_out_pol': all_n_actions, 'num_in_critic': env_info['state_shape']+sum(all_n_actions), 'num_agent':env_info['n_agents']}
                
        else:
            if env_id in ['simple_tag', 'simple_world']:
                predator_num = len(alg_types)
                prey_num = len([agent_alg for atype in env.agent_types if atype == 'agent'])
                env_action_space = [env.action_space[i] for i in range(len(env.action_space)) if env.agent_types[i] == 'adversary']
                env_obs_space = [env.observation_space[i] for i in range(len(env.observation_space)) if env.agent_types[i] == 'adversary']
                env_action_space_prey = [env.action_space[i] for i in range(len(env.action_space)) if env.agent_types[i] == 'agent']
                env_obs_space_prey = [env.observation_space[i] for i in range(len(env.observation_space)) if env.agent_types[i] == 'agent']
            else:
                env_action_space, env_obs_space = env.action_space, env.observation_space
            for acsp, obsp in zip(env_action_space, env_obs_space):
                all_n_obs.append(obsp.shape[0])
                all_n_actions.append(acsp.shape[0])
                agent_max_actions.append(acsp.high[0])
            for i in range(1, len(all_n_actions)):
                assert (all_n_actions[i] == all_n_actions[0])
            agent_init_params={'num_in_pol': env.observation_space[0].shape[0], 'num_out_pol': all_n_actions, 'num_in_critic': sum(all_n_obs)+sum(all_n_actions), 'num_agent':len(alg_types)}
            if env_id in ['simple_tag', 'simple_world']:
                for acsp, obsp in zip(env_action_space_prey, env_obs_space_prey):
                    num_in_pol = obsp.shape[0]
                    num_out_pol = acsp.shape[0]
                    num_in_critic = num_in_pol + num_out_pol
                    adv_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})

        env_config_map = {
            'simple_spread': {
                'random': {'omar_coe': 1.0, 'cql_alpha': 0.5},
                'medium-replay': {'omar_coe': 1.0, 'cql_alpha': 1.0},
                'medium': {'omar_coe': 1.0, 'cql_alpha': 5.0},
                'expert': {'omar_coe': 1.0, 'cql_alpha': 2.5},
            },
            'simple_tag': {
                'random': {'omar_coe': 0.9, 'cql_alpha': 0.5},
                'medium-replay': {'omar_coe': 0.9, 'cql_alpha': 0.5},
                'medium': {'omar_coe': 0.7, 'cql_alpha': 0.6},
                'expert': {'omar_coe': 0.9, 'cql_alpha': 2.0},
            },
            'simple_world': {
                'random': {'omar_coe': 1.0, 'cql_alpha': 0.5},
                'medium-replay': {'omar_coe': 0.7, 'cql_alpha': 0.5},
                'medium': {'omar_coe': 0.4, 'cql_alpha': 0.32},
                'expert': {'omar_coe': 0.8, 'cql_alpha': 0.3},
            },
            'HalfCheetah-v2': {
                'random': {'omar_coe': 1.0, 'cql_alpha': 0.1},
                'medium-replay': {'omar_coe': 0.9, 'cql_alpha': 0.4},
                'medium': {'omar_coe': 0.7, 'cql_alpha': 1.0},
                'expert': {'omar_coe': 0.5, 'cql_alpha': 1.7},
            }
        }
        omar_coe = env_config_map[env_id][data_type]['omar_coe']
        cql_alpha = env_config_map[env_id][data_type]['cql_alpha']
        if kwargs.get('cql_alpha', 0) > 0:
            cql_alpha = kwargs['cql_alpha']
        if kwargs.get('omar_coe', 0) > 0:
            omar_coe = kwargs['omar_coe']
        # cql = True if omar else cql
        print("cql_alpha", cql_alpha)
        init_dict = {
            'env_id': env_id,
            'gamma': gamma, 
            'tau': tau, 
            'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'adv_init_params': adv_init_params, 
            'discrete_action': False,
            'cql': cql, 'cql_alpha': cql_alpha, 'lse_temp': lse_temp, 'num_sampled_actions': num_sampled_actions,
            'batch_size': batch_size,
            'gaussian_noise_std': gaussian_noise_std,
            'agent_max_actions': agent_max_actions,
            'omar': omar, 'omar_coe': omar_coe,
            'omar_iters': omar_iters, 'omar_mu': omar_mu, 'omar_sigma': omar_sigma, 'omar_num_samples': omar_num_samples, 'omar_num_elites': omar_num_elites,
        }
        init_dict.update(kwargs)
        
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance

    def load_pretrained_preys(self, filename):
        if not torch.cuda.is_available():
            save_dict = torch.load(filename, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(filename)

        if self.env_id in ['simple_tag', 'simple_world']:
            prey_params = save_dict['agent_params'][self.num_predators:]

        for i, params in zip(range(self.num_preys), prey_params):
            self.preys[i].load_params_without_optims(params)

        for p in self.preys:
            p.policy.eval()
            p.target_policy.eval()

