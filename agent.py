import torch
from hyper.core import hyperActor
import torch.nn as nn
import numpy as np
import os
from torch.distributions.normal import Normal




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, device, 
                 hyper = False, 
                 meta_batch_size = 8, 
                 arch_conditional_critic = False, 
                 dual_critic = False, 
                 std_mode = False, 
                 multi_gpu = False, 
                 architecture_sampling = 'biased'
                 ):
        super().__init__()
        self.hyper = hyper
        self.arch_conditional_critic = arch_conditional_critic
        self.dual_critic = dual_critic
        self.std_mode = std_mode

        if self.hyper:
            self.actor_mean = hyperActor(np.prod(envs.single_action_space.shape), np.array(envs.single_observation_space.shape).prod(), np.array([4,8,16,32,64,128,256]), \
                                         meta_batch_size = meta_batch_size, device=device, multi_gpu=multi_gpu, architecture_sampling_mode=architecture_sampling, std_mode = self.std_mode)
            # self.actor_mean.change_graph()
            self.actor_logstd = self.actor_mean.log_std

            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + arch_conditional_critic*self.actor_mean.arch_max_len, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )            

            if self.dual_critic:
                assert self.arch_conditional_critic, "Dual critic requires arch_conditional_critic to be True"

                self.critic2 = nn.Sequential(
                    layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 1), std=1.0),
                )
        
        else:
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256)),
                nn.ReLU(),                
                layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )            

    def get_value(self, x, architectures = None):
        if self.hyper and self.arch_conditional_critic:
            if self.dual_critic:
                return (self.critic(torch.cat([x, architectures if architectures is not None else self.actor_mean.arch_per_state_dim], dim = 1)), self.critic2(x))
            else:
                return self.critic(torch.cat([x, architectures if architectures is not None else self.actor_mean.arch_per_state_dim], dim = 1))
        else:
            return self.critic(x)

    def get_action(self, x, action = None):
        if self.hyper:
            action_mean, action_logstd = self.actor_mean(x)
        else:
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)        

    def get_action_and_value(self, x, action=None):
        if self.hyper:
            action_mean, action_logstd = self.actor_mean(x)
            if self.arch_conditional_critic:
                value = self.critic(torch.cat([x, self.actor_mean.arch_per_state_dim], dim = 1))

                if self.dual_critic:
                    value2 = self.critic2(x)
                    value = (value, value2)
            else:
                value = self.critic(x)
        else:
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            value = self.critic(x)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def get_mean_action(self,x):
        if self.hyper:
            action_mean, _ = self.actor_mean(x, track = False)
        else:
            action_mean = self.actor_mean(x)
        
        return action_mean

    def save_model(self, save_dir, epoch):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if epoch == -1:
            path = os.path.join(save_dir, 'model.pt')
        else:
            path = os.path.join(save_dir, 'model_epoch_{}.pt'.format(epoch))
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")


    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def to(self, device):
        super().to(device)
        self.device = device
        if self.hyper:
            self.actor_mean.list_of_shape_inds = self.actor_mean.list_of_shape_inds.to(device)
            self.actor_mean.device = device
            self.actor_mean.ghn.default_edges = self.actor_mean.ghn.default_edges.to(device)
            self.actor_mean.ghn.device = device
            self.actor_mean.ghn.default_node_feat = self.actor_mean.ghn.default_node_feat.to(device)
        return self