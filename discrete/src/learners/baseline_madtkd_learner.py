import copy
from components.episode_buffer import EpisodeBatch
from controllers.n_controller import NMAC
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_gae_targets
import torch as th
from torch.optim import Adam
from utils.value_norm import ValueNorm
import os 
from os.path import dirname, abspath
from controllers.dt_controller import DTMAC

class MADTKDLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger
        self.teacher = args.teacher

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        # a trick to reuse mac
       

        self.optimiser = self.mac.optimiser()
        self.last_lr = args.lr
        self.train_times = 0

        if not self.args.teacher:
            dummy_args = copy.deepcopy(args)
            dummy_args.teacher = True
            self.teacher_mac = DTMAC(scheme,None,dummy_args)
            self.teacher_optimiser = self.teacher_mac.optimiser()
            if 'map_name' in args.env_args.keys():
                teacher_checkpoint_path = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets", args.env_args['map_name']+"_"+args.h5file_suffix+"_teacher_model")
            else:
                teacher_checkpoint_path = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "offline_datasets", args.env+"_"+args.h5file_suffix+"_teacher_model")
        
            if os.path.exists(teacher_checkpoint_path):
                # behaviour_save_path = args.behaviour_checkpoint_path
                logger.console_logger.info("Loading teacher model from {}".format(teacher_checkpoint_path))
                self.teacher_mac.load_models(teacher_checkpoint_path)
                self.teacher_mac.cuda()
            else:
                logger.console_logger.info("Need train teacher model first!!!!!!!")
                raise NotImplementedError()
    
    def train_teacher(self,batch: EpisodeBatch, t_env: int, episode_num: int):
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        actions = batch["actions"][:, :-1]
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)#bs,ts,na,1
        pi=[]
        for t in range(batch.max_seq_length-1):
            agent_outs,_ = self.mac.forward(batch, t=t)
            pi.append(agent_outs)
        pi = th.stack(pi, dim=1)
        pi[avail_actions == 0] = 1e-10
        pi_taken = th.gather(pi, dim=3, index=actions)
        log_pi_taken = th.log(pi_taken)
        loss = -(log_pi_taken* mask_agent).sum() / mask_agent.sum()

        # Optimise agents
        self.optimiser.zero_grad()
        loss.backward()
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_agent.sum().item()
            self.logger.log_stat("teacher_train_loss", loss.item(), t_env)
            # self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("lr", self.last_lr, t_env)
            self.logger.log_stat("pi_taken", (pi_taken* mask_agent).sum() / mask_agent.sum(), t_env)
            self.log_stats_t = t_env
        return loss.item()
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.train_times +=1
        if self.teacher:
            loss = self.train_teacher(batch, t_env, episode_num)
            return loss
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)

        # Actor
        pi = []
        mapping_feature = []
        for t in range(batch.max_seq_length-1):
            agent_outs,mf = self.mac.forward(batch, t=t)
            pi.append(agent_outs)
            mapping_feature.append(mf)
        pi = th.stack(pi, dim=1)  # Concat over time
        pi[avail_actions == 0] = 1e-10 #bs,ts,na,ad
        mapping_feature = th.stack(mapping_feature, dim=1) #bs,ts,na,nembd
        pi_taken = th.gather(pi, dim=3, index=actions)
        log_pi_taken = th.log(pi_taken)
        # loss = -(log_pi_taken* mask_agent).sum() / mask_agent.sum()

        teacher_pi = []
        teacher_mapping_feature = []
        for t in range(batch.max_seq_length-1):
            teacher_agent_outs,teacher_mf = self.teacher_mac.forward(batch, t=t)
            teacher_pi.append(teacher_agent_outs)
            teacher_mapping_feature.append(teacher_mf)
        teacher_pi = th.stack(teacher_pi, dim=1)  # Concat over time
        teacher_pi[avail_actions == 0] = 1e-10 #bs,ts,na,ad
        teacher_mapping_feature = th.stack(teacher_mapping_feature, dim=1) #bs,ts,na,nembd

        loss = -(log_pi_taken* mask_agent).sum() / mask_agent.sum()

        loss_kl = ((th.nn.functional.kl_div(th.log(pi),teacher_pi,reduction='none')*avail_actions).sum(-1,keepdim=True)* mask_agent).sum() / mask_agent.sum()
        loss += self.args.alpha * loss_kl
        loss_rel = 0
        for ii in range(self.args.n_agents):
            for jj in range(self.args.n_agents):
                if ii == jj:continue
                loss_rel += (th.nn.functional.huber_loss(th.cosine_similarity(teacher_mapping_feature[:,:,ii].detach(),teacher_mapping_feature[:,:,jj].detach(),dim=-1),
                th.cosine_similarity(mapping_feature[:,:,ii],mapping_feature[:,:,jj],dim=-1),reduction='none').unsqueeze(-1)* mask).sum() / mask.sum()
        loss += self.args.beta * loss_rel

        # Optimise agents
        self.optimiser.zero_grad()
        loss.backward()
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if self.train_times//self.args.M_update_e==0:
            mapping_feature = mapping_feature.detach()
            teacher_loss_rel = 0
            for ii in range(self.args.n_agents):
                for jj in range(self.args.n_agents):
                    if ii == jj:continue
                    teacher_loss_rel += (th.nn.functional.huber_loss(1/th.cosine_similarity(teacher_mapping_feature[:,:,ii],teacher_mapping_feature[:,:,jj],dim=-1),
                    1/th.cosine_similarity(mapping_feature[:,:,ii],mapping_feature[:,:,jj],dim=-1),reduction='none').unsqueeze(-1)* mask).sum() / mask.sum()
            teacher_loss_rel = teacher_loss_rel*self.args.beta
            self.teacher_optimiser.zero_grad()
            teacher_loss_rel.backward()
            # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.teacher_optimiser.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_agent.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("lr", self.last_lr, t_env)
            self.logger.log_stat("loss_kl", loss_kl, t_env)
            self.logger.log_stat("loss_rel", loss_rel, t_env)
            self.log_stats_t = t_env
        return loss.item()


    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/agent_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
