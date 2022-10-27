import torch
import torch.nn as nn
from torch.distributions import Normal
from hyper.core import hyperActor
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class HyperActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(HyperActorCritic, self).__init__()


        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.actor = hyperActor(num_outputs, num_inputs, np.array([4,8,16,32,64,128,256,512]), meta_batch_size = 8, device=device)

        self.actor = hyperActor(num_outputs, num_inputs, np.array([256]), meta_batch_size = 8, device=device)

        self.actor.change_graph()
        self.switch_counter = 0

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.min_log_std = -6
        self.max_log_std = 0


    def forward(self, x):  
        value = self.critic(x)
        mu, _ = self.actor(x)
        # std = log_std.exp()
        # std = self.log_std.exp().expand_as(mu)
        std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std).exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value