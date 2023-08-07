
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class DoubleQNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, scheme,args, hidden_dim=256, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(DoubleQNetwork, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = self._get_input_shape(scheme)
        self.action_dim=scheme["actions_onehot"]["vshape"][0]

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.action_dim)

        self.fc4 = nn.Linear(self.state_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, self.action_dim)

        self.nonlin = nonlin

    def both(self,X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        bs,t,na,_ = X.shape
        X = X.reshape(-1,self.state_dim)
        norm_in_X = X

        h1 = self.nonlin(self.fc1(norm_in_X))
        h2 = self.nonlin(self.fc2(h1))
        a1 = self.fc3(h2)
        out = a1.reshape(bs,t,na,self.action_dim)

        h1_2 = self.nonlin(self.fc4(norm_in_X))
        h2_2 = self.nonlin(self.fc5(h1_2))
        a2 = self.fc3(h2_2)
        out_2 = a2.reshape(bs,t,na,self.action_dim)

        return out, out_2

    def forward(self, X):
        return th.min(*self.both(X))
        
    
    def _build_inputs(self, batch, bs, max_t):
        inputs = []
        # state, obs, action
        inputs.append(batch["state"][:].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs.append(batch["obs"][:])
        inputs.append(th.eye(self.n_agents, device=self.args.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]

        input_shape += self.n_agents
        return input_shape
    
class ValueNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, scheme,args, hidden_dim=256, nonlin=F.relu):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(ValueNetwork, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_dim = self._get_input_shape(scheme)
        self.action_dim=scheme["actions_onehot"]["vshape"][0]

        self.fc1 = nn.Linear(self.state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)


        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        bs,t,na,_ = X.shape
        X = X.reshape(-1,self.state_dim)
        norm_in_X = X

        h1 = self.nonlin(self.fc1(norm_in_X))
        h2 = self.nonlin(self.fc2(h1))
        a1 = self.fc3(h2)
        out = a1.reshape(bs,t,na,1)

        return out
    
    def _build_inputs(self, batch, bs, max_t):
        inputs = []
        # state, obs, action
        inputs.append(batch["state"][:].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        inputs.append(batch["obs"][:])
        inputs.append(th.eye(self.n_agents, device=self.args.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs
    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observation
        input_shape += scheme["obs"]["vshape"]

        input_shape += self.n_agents
        return input_shape