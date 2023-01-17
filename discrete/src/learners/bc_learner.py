import copy
from components.episode_buffer import EpisodeBatch
from controllers.n_controller import NMAC
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_gae_targets
import torch as th
from torch.optim import Adam
from utils.value_norm import ValueNorm

class BCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        # a trick to reuse mac
        dummy_args = copy.deepcopy(args)
        dummy_args.n_actions = 1
        # self.critic = NMAC(scheme, None, dummy_args)
        self.params = list(mac.parameters()) #+ list(self.critic.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.last_lr = args.lr
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        
        # old_probs = batch["probs"][:, :-1]
        # old_probs[avail_actions == 0] = 1e-10
        # old_logprob = th.log(th.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        # PPO Loss
        for _ in range(self.args.mini_epochs):
            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = th.gather(pi, dim=3, index=actions)
            log_pi_taken = th.log(pi_taken)
            loss = -(log_pi_taken* mask_agent).sum() / mask_agent.sum()

            # Optimise agents
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_agent.sum().item()
            self.logger.log_stat("bc_loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("lr", self.last_lr, t_env)
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
