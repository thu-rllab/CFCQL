import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
import torch.nn.functional as F

class CQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
        
        # train behaviour
        self.need_train_behaviour = False
        if getattr(self.args, 'moderate_lambda', False):
            self.behaviour_train_steps = 0 
            self.behaviour_log_stats_t = 0
            self.need_train_behaviour = True
            self.last_min_loss = 1e6
            self.epoch_since_last_min_loss = 0
            from controllers import REGISTRY as mac_REGISTRY
            args.mask_before_softmax=True
            args.agent = 'n_rnn'
            self.mini_epochs= 1
            args.agent_output_type = 'pi_logits'
            args.action_selector="multinomial"
            args.epsilon_start= .0
            args.epsilon_finish= .0
            args.epsilon_anneal_time= 500000
            self.behaviour_mac = mac_REGISTRY['basic_mac'](scheme, None, args)
            self.behaviour_params = list(self.behaviour_mac.parameters())
            self.behaviour_optimiser = Adam(params=self.behaviour_params, lr=args.lr)

    def save_behaviour_model(self,path):
        self.behaviour_mac.save_models(path)
        th.save(self.behaviour_optimiser.state_dict(), "{}/opt.th".format(path))
    def load_behaviour_model(self,path):
        self.behaviour_mac.load_models(path)
        self.behaviour_optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    def train_behaviour(self,batch: EpisodeBatch):
        # Get the relevant quantities
        self.behaviour_train_steps+=1
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        
        mask_agent = mask.unsqueeze(2).repeat(1, 1, actions.shape[2], 1)

        for _ in range(self.mini_epochs):
            # Actor
            pi = []
            self.behaviour_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.behaviour_mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = th.gather(pi, dim=3, index=actions)
            log_pi_taken = th.log(pi_taken)
            loss = -(log_pi_taken* mask_agent).sum() / mask_agent.sum()

            # Optimise agents
            self.behaviour_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.behaviour_params, self.args.grad_norm_clip)
            self.behaviour_optimiser.step()
            


        if self.behaviour_train_steps - self.behaviour_log_stats_t >= 20:
            self.logger.log_stat("bc_loss", loss.item(), self.behaviour_train_steps)
            self.behaviour_log_stats_t = self.behaviour_train_steps
            self.logger.console_logger.info("Behaviour model training loss: {}, training steps: {}".format(loss.item(), self.behaviour_train_steps))
        if loss.item()<self.last_min_loss:
            self.last_min_loss=loss.item()
            self.epoch_since_last_min_loss=0
        else:
            self.epoch_since_last_min_loss +=1
        behaviour_train_done = self.epoch_since_last_min_loss > 20
        if behaviour_train_done:
            self.epoch_since_last_min_loss=0
            self.logger.log_stat("bc_loss", loss.item(), self.behaviour_train_steps)

        return behaviour_train_done,loss.item()
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time bs,ts,na,action

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            if self.args.use_sarsa:
                target_max_qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                self.args.cql_alpha=0.0
                self.args.global_cql_alpha=0.0
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qtotals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qtotals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight
        
        ##add cql loss
        n_agents = mac_out.shape[2]
        n_actions = mac_out.shape[3]
        if self.args.raw_cql:#1121change to avail
            sample_actions_num = self.args.raw_sample_actions
            bs = actions.shape[0]
            ts = actions.shape[1]
            sample_enough =False
            repeat_avail_actions = th.repeat_interleave(avail_actions[:,:-1].unsqueeze(0),repeats=sample_actions_num,dim=0)#san,bs,ts,na,ad
            # while not sample_enough:
            total_random_actions = th.randint(low=0,high=n_actions,size=(sample_actions_num,bs,ts,n_agents,1)).to(self.args.device)#san,bs,ts,na,1
            chosen_if_avail = th.gather(repeat_avail_actions, dim=-1, index=total_random_actions).min(-2)[0]#san,bs,ts,1
            repeat_mac_out = th.repeat_interleave(mac_out[:,:-1].unsqueeze(0),repeats=sample_actions_num,dim=0)#san,bs,ts,na,ad
            random_chosen_action_qvals = th.gather(repeat_mac_out, dim=-1, index=total_random_actions).squeeze(-1)#san,bs,ts,na
            repeat_state = th.repeat_interleave(batch["state"][:, :-1].unsqueeze(0),repeats=sample_actions_num,dim=0)#san,bs,ts,sd
            random_chosen_action_qvals = random_chosen_action_qvals.view(bs*sample_actions_num,ts,-1)
            repeat_state = repeat_state.view((bs*sample_actions_num,ts,-1))
            random_chosen_action_qtotal = self.mixer(random_chosen_action_qvals,repeat_state).view(sample_actions_num,bs,ts,1)#san,bs,ts,1
            negative_sampling = th.logsumexp(random_chosen_action_qtotal*chosen_if_avail,dim=0)#bs,ts,1
            dataset_expec = chosen_action_qtotals#bs,ts,1
            cql_loss = self.args.global_cql_alpha * ((negative_sampling-dataset_expec)* mask).sum()/mask.sum()
        
        else:
            total_random_actions = actions
            lambda_mask = th.ones_like(actions).squeeze(-1)/n_agents
            if self.need_train_behaviour:
                # mu_prob = th.exp(mac_out[:,:-1])
                beta_prob = []
                self.behaviour_mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length-1):
                    agent_outs = self.behaviour_mac.forward(batch, t=t)
                    beta_prob.append(agent_outs)
                beta_prob = th.stack(beta_prob, dim=1)
                ratio = []
            negative_sampling=[]
            for ii in range(n_agents):
                noexp_negative_sampling = []
                for jj in range(n_actions):
                    random_actions = th.concat([total_random_actions[:,:,:ii],th.ones_like(actions[:,:,0:1]).to(self.args.device)*jj,total_random_actions[:,:,ii+1:]],dim=2)
                    random_chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=random_actions).squeeze(-1)
                    random_chosen_action_qtotal = self.mixer(random_chosen_action_qvals,batch["state"][:, :-1])#bs,ts,1
                    random_chosen_action_qtotal[avail_actions[:,:-1,ii,jj]==0]=-1e10
                    noexp_negative_sampling.append(random_chosen_action_qtotal)
                noexp_negative_sampling = th.concat(noexp_negative_sampling,dim=-1)
                if self.need_train_behaviour:
                    mu_prob = th.nn.functional.softmax(noexp_negative_sampling,dim=-1)#bs,ts,ad
                    assert beta_prob[:,:,ii].shape == mu_prob.shape
                    ratio.append((th.nn.functional.kl_div(th.log(beta_prob[:,:,ii]+0.00001),mu_prob+0.00001,reduction='none')*avail_actions[:,:-1,ii]).sum(-1,keepdim=True))#bs,ts,1
                negative_sampling.append(th.logsumexp(noexp_negative_sampling,dim=-1).unsqueeze(-1))#bs,ts,1(list(na))
            if self.need_train_behaviour:
                ratio = th.concat(ratio,dim=-1)#bs,ts,na
                if self.args.softmax_temp==100:
                    lambda_mask = th.nn.functional.one_hot(th.argmax(ratio,dim=-1),num_classes=n_agents).detach()
                else:
                    lambda_mask = th.nn.functional.softmax(ratio*self.args.softmax_temp,dim=-1).detach() #1126 softkl
                # lambda_mask = th.nn.functional.one_hot(th.argmin(ratio,dim=-1),num_classes=n_agents).detach()#bs,ts,na
            negative_sampling = th.concat(negative_sampling,dim=-1)#bs,ts,na
            negative_sampling = (negative_sampling*lambda_mask).sum(-1,keepdim=True)#bs,ts,1

            # negative_sampling = th.logsumexp(noexp_negative_sampling,dim=0).mean()
            dataset_expec = chosen_action_qtotals
            cql_loss = self.args.global_cql_alpha * ((negative_sampling-dataset_expec)* mask).sum()/mask.sum()



        L_td = masked_td_error.sum() / mask.sum()
        loss = cql_loss + L_td

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        
        if th.any(th.isnan(cql_loss)) or th.any(th.isnan(grad_norm)):
            print('there is nan!!!!!!!!!!!!')

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # self.logger.log_stat("q_taken_mean", (chosen_action_qtotals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qtotals * mask).sum().item()/(mask_elems), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems), t_env)
            self.logger.log_stat("q_local_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("cql_loss", (((negative_sampling-dataset_expec)* mask).sum().item()/mask_elems), t_env)
            self.logger.log_stat("negtive_sampling_mean", ((negative_sampling* mask).sum().item()/mask_elems), t_env)
            if self.need_train_behaviour:
                self.logger.log_stat("lambda_ratio",((ratio*mask).sum().item())/mask_elems,t_env)
                self.logger.log_stat("lambda_mask_max",((lambda_mask.max(-1,keepdim=True)[0]*mask).sum().item())/mask_elems,t_env)
                self.logger.log_stat("lambda_mask_min",((lambda_mask.min(-1,keepdim=True)[0]*mask).sum().item())/mask_elems,t_env)
                self.logger.log_stat("lambda_mask_mean",((lambda_mask*mask).sum().item())/mask_elems,t_env)
            self.log_stats_t = t_env
            

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.need_train_behaviour:
            self.behaviour_mac.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def cal_Dcql(self, batch: EpisodeBatch, t_env: int):
        # Get the relevant quantities
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:,:-1]
        mask_elems = mask.sum().item()
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length-1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)# Concat over time bs,ts,na,action
        mac_out[avail_actions == 0] = -1e10
        pi_prob = th.nn.functional.softmax(mac_out,dim=-1)
        pi_prob[avail_actions == 0] = 0 
        pi_prob_2 = pi_prob**2

        if self.need_train_behaviour:
            # mu_prob = th.exp(mac_out[:,:-1])
            beta_prob = []
            self.behaviour_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.behaviour_mac.forward(batch, t=t)
                beta_prob.append(agent_outs)
            beta_prob = th.stack(beta_prob, dim=1)
            beta_prob[avail_actions == 0] = 1e-10
        if self.args.raw_cql:
            Dcql_s=th.prod(th.sum(pi_prob_2/beta_prob*avail_actions,dim=-1),dim=-1,keepdim=True)-1
            Dcql = ((Dcql_s*mask).sum().item())/mask_elems
        else:
            Dcql_s=th.mean(th.sum(pi_prob_2/beta_prob*avail_actions,dim=-1),dim=-1,keepdim=True)-1
            Dcql = ((Dcql_s*mask).sum().item())/mask_elems
        self.logger.log_stat("Dcql", Dcql, t_env)
        print('Dcql:',Dcql)

