import numpy as np
import torch


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale

def action_noise(x, mu=0.0, sigma=0.05, upper=1.0, lower=-1.0, noise_type='normal'):
    mu_ = torch.full(x.shape, mu, device=x.device)
    sigma_ = torch.full(x.shape, sigma, device=x.device)
    if noise_type == 'normal':
        dis = torch.distributions.Normal(mu_, sigma_)
    elif noise_type == 'uniform':
        dis = torch.distributions.Uniform(mu_-sigma_, mu_+sigma_)
    else:
        raise NotImplementedError
    noise = dis.sample()
    return torch.clamp(x+noise, lower, upper)


