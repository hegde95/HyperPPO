import torch.nn as nn
from typing import List
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
import torch

class MlpEncoder(Encoder):
    def __init__(self, cfg, inp_dim):
        super().__init__(cfg)

        self.mlp_head = nn.Sequential(
            nn.Linear(inp_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.encoder_out_size = 256

    def forward(self, obs):
        x = self.mlp_head(obs)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size
    
class MTEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        out_size = 0
        self.encoders['task'] = nn.Sequential(
            nn.Embedding(10, 10),
            MlpEncoder(cfg, 10)
        )
        out_size += self.encoders['task'][1].get_out_size()

        if 'state' in self.obs_keys:
            self.encoders['state'] = MlpEncoder(cfg, obs_space['state'].shape[0])
        else:
            self.encoders['state'] = MlpEncoder(cfg, obs_space['obs'].shape[0]-1)
        out_size += self.encoders['state'].get_out_size()

        self.fc_out = nn.Sequential(
            nn.Linear(out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

        self.encoder_out_size = 4

    def forward(self, obs_dict):
        if len(self.obs_keys) == 1:
            key = self.obs_keys[0]
            return self.encoders[key](obs_dict[key])

        encodings = []
        for key in self.obs_keys:
            if key == 'task':
                x = self.encoders[key](obs_dict[key].to(torch.long)).squeeze(1)
            else:    
                x = self.encoders[key](obs_dict[key])
            encodings.append(x)

        enc = torch.cat(encodings, 1)
        out = self.fc_out(enc)
        return out

    def get_out_size(self) -> int:
        return self.encoder_out_size