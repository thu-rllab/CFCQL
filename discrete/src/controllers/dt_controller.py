from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class DTMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self._get_input_shape(scheme)
        self._build_agents()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs,_ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        gpt_state,gpt_action,gpt_reward,gpt_timestep = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if test_mode:
            self.agent.eval()
        agent_outs, mapping_feature = self.agent(gpt_state,gpt_action,rtgs=gpt_reward,timesteps=gpt_timestep)
        # agent_outs = agent_outs[:,-1]
        # mapping_feature = mapping_feature[:,-1]

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1),mapping_feature.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        pass

    def optimiser(self):
        return self.agent.configure_optimizers(self.args, self.args.lr)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        th.nn.DataParallel(self.agent).cuda()
        # self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self):
        if self.args.teacher:
            self.agent = agent_REGISTRY['teacher_'+self.args.agent](self.args)
        else:
            self.agent = agent_REGISTRY[self.args.agent](self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        na = self.args.n_agents
        obs = batch['obs']
        reward = batch['reward']
        actions = batch['actions']
        context_length = self.args.context_length
        if not self.args.teacher:
            bs = batch.batch_size * self.n_agents
            obs = obs.transpose(1,2).reshape(bs,-1,self.args.gpt_state_shape)
            reward = th.repeat_interleave(reward.unsqueeze(0),repeats=self.n_agents,dim=0).reshape(bs,-1,1)
            actions = actions.transpose(1,2).reshape(bs,-1,1)
            gpt_state = th.zeros((bs,context_length,self.args.gpt_state_shape)).to(self.args.device)
            gpt_action = th.zeros((bs,context_length,1)).to(self.args.device)

        else:
            gpt_state = th.zeros((bs,context_length,na,self.args.gpt_state_shape)).to(self.args.device)
            gpt_action = th.zeros((bs,context_length,na,1)).to(self.args.device)
        # gpt_rtgs = th.zeros((bs,context_length,1)).to(self.args.device)
        gpt_timestep = th.zeros((bs,context_length,1)).to(self.args.device)
        true_timestep = th.stack(list(th.range(max(0,t-context_length+1),t,device=self.args.device))).unsqueeze(-1)

        gpt_state[:,-min(context_length,t+1):] = obs[:,max(0,t-context_length+1):t+1]
        # gpt_reward[:,-min(context_length,t+1):] = th.reshape(reward[:,max(0,t-context_length+1):t+1],(bs,min(context_length,t+1),-1))
        gpt_action[:,-min(context_length,t+1):-1] = actions[:,max(0,t-context_length+1):t]
        gpt_action[:,-1]=0
        gpt_timestep[:,-min(context_length,t+1):] = true_timestep
        return gpt_state,gpt_action,self._compute_reward_to_go(reward,t),gpt_timestep.type(th.int64)

    def _get_input_shape(self, scheme):
        self.args.gpt_state_shape = scheme["obs"]["vshape"]  #reward
        self.args.gpt_action_shape = scheme["actions_onehot"]["vshape"][0]
        # if self.args.teacher:
        #     self.args.gpt_state_shape = scheme["obs"]["vshape"]*self.n_agents
        #     self.args.gpt_action_shape *= self.n_agents
    
    def _compute_reward_to_go(self,reward,t):
        bs,ts,_ = reward.shape
        context_length = self.args.context_length
        gpt_rtgs = th.zeros((bs,context_length,1)).to(self.args.device)
        rtgs_t = reward[:,-1]
        for ii in range(ts-2,max(t-context_length,-1),-1):
            rtgs_t = self.args.gamma*rtgs_t +reward[:,ii]
            if ii <=t:
                gpt_rtgs[:,ii-t-1]=rtgs_t
        gpt_rtgs[:,-1]=0
        return gpt_rtgs

        



