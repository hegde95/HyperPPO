import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import Normal
from torch.distributions.normal import Normal

from hyper.ghn_modules import MLP_GHN, MlpNetwork
import numpy as np
from itertools import product as cartesian_product
import random
from itertools import product
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather

class hyperActor(nn.Module):

    def __init__(self, 
                act_dim, 
                obs_dim, 
                # act_limit, 
                allowable_layers, 
                search = False, 
                conditional = True, 
                meta_batch_size = 1,
                # gumbel_tau = 1.0,
                device = "cpu"
                ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.is_search = search
        self.conditional = conditional
        self.meta_batch_size = meta_batch_size
        self.device = device


        list_of_allowable_layers = list(allowable_layers)

        self.list_of_arcs = []
        
        # for i in range(500):
        #     self.list_of_arcs.extend([
        #         [4, 4],
        #         [8, 8, 8],
        #         [16],
        #         [16, 16, 16],
        #         [32, 32, 32],
        #         [64, 64, 64, 64],
        #         [128, 128, 128, 128],
        #         [256, 256, 256, 256],

        #         [8],
        #         [32],
        #         [16, 256],
        #         [32, 32, 32, 32],
        #         [64],
        #         [128, 128],
        #         [256, 256],

        #         [8, 8],
        #         [16, 8],
        #         [16, 8, 16],
        #         [32, 64, 32],
        #         [64, 64, 64],
        #         [128, 128, 128],
        #         [256, 256, 256],                        
        #     ])
        # self.list_of_arcs.extend([
        #         [4, 4],
        #         [8, 8, 8],
        #         [16],
        #         [16, 16, 16],
        #         [32, 32, 32],
        #         [64, 64, 64, 64],
        #         [128, 128, 128, 128],
        #         [256, 256, 256, 256]
        #     ])
        
        for k in range(1,5):
            self.list_of_arcs.extend(list(product(list_of_allowable_layers, repeat = k)))
        
        # self.list_of_arcs = [(256,256,256) for i in range(1000)]
        
        self.list_of_arcs.sort(key = lambda x:self.get_params(x))

        self.list_of_shape_inds = []
        for arc in self.list_of_arcs:
            shape_ind = [torch.tensor(0).type(torch.FloatTensor).to(self.device)]
            for layer in arc:
                shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
                shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor(self.act_dim * 2).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor(self.act_dim * 2).type(torch.FloatTensor).to(self.device))
            shape_ind = torch.stack(shape_ind).view(-1,1)
            self.list_of_shape_inds.append(shape_ind) 

        self.list_of_shape_inds_lenths = [x.squeeze().numel() for x in self.list_of_shape_inds]
        self.shape_inds_max_len = max(self.list_of_shape_inds_lenths)
        self.arch_max_len = 4
        # pad -1 to the end of each shape_ind
        for i in range(len(self.list_of_shape_inds)):
            num_pad = (self.shape_inds_max_len - self.list_of_shape_inds[i].shape[0])
            self.list_of_shape_inds[i] = torch.cat([self.list_of_shape_inds[i], torch.tensor(-1).to(self.device).repeat(num_pad,1)], 0)
        self.list_of_shape_inds = torch.stack(self.list_of_shape_inds)
        self.list_of_shape_inds = self.list_of_shape_inds.reshape(len(self.list_of_shape_inds),self.shape_inds_max_len)
        self.list_of_arc_indices = np.arange(len(self.list_of_arcs))
        # shuffle the list of arcs indices
        self.all_models = [MlpNetwork(fc_layers=self.list_of_arcs[index], inp_dim = self.obs_dim, out_dim = 2 * self.act_dim) for index in self.list_of_arc_indices]
        np.random.shuffle(self.list_of_arc_indices)
        self.current_model_indices = np.arange(self.meta_batch_size)
        config = {}
        config['max_shape'] = (256, 256, 1, 1)
        config['num_classes'] = 4 * act_dim
        config['num_observations'] = obs_dim
        config['weight_norm'] = True
        config['ve'] = 1 > 1
        config['layernorm'] = True
        config['hid'] = 16

        self.ghn_config = config

        self.ghn = MLP_GHN(**config,
                    debug_level=0, device=self.device).to(self.device)  

        # get all torch devices
        self.all_devices = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
        self.num_current_models_per_device = int(self.meta_batch_size / len(self.all_devices)) 
        self.device_model_list = []
        for device in self.all_devices:
            self.device_model_list.extend([device for i in range(self.num_current_models_per_device)])

    def set_graph(self, indices_vector, shape_ind_vec):
        self.sampled_indices = indices_vector
        self.sampled_shape_inds = shape_ind_vec.view(-1)[shape_ind_vec.view(-1) != -1].unsqueeze(-1)
        self.current_model = [self.all_models[i] for i in self.sampled_indices]
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)


    def get_params(self, net):
        ct = 0
        ct += ((self.obs_dim + 1) *net[0])
        for i in range(len(net)-1):
            ct += ((net[i] + 1) * net[i+1])
        ct += ((net[-1] +1) * 2 * self.act_dim)
        return ct            

    def re_query_uniform_weights(self, repeat_sample = False):
        if not repeat_sample:
            self.sampled_indices = self.list_of_arc_indices[self.current_model_indices]
            # self.list_of_sampled_shape_inds = [self.list_of_shape_inds[index][:self.list_of_shape_inds_lenths[index]] for index in self.sampled_indices]
            self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
            self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]
            # self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)   
            self.current_model_indices += self.meta_batch_size  
            if max(self.current_model_indices) >= len(self.list_of_arc_indices):
                self.current_model_indices = np.arange(self.meta_batch_size)
                # shuffle
                np.random.shuffle(self.list_of_arc_indices)

            self.current_archs = torch.tensor([list(self.list_of_arcs[index]) + [0]*(4-len(self.list_of_arcs[index])) for index in self.sampled_indices]).to(self.device) 
            self.current_model = [MlpNetwork(fc_layers=self.list_of_arcs[index], inp_dim = self.obs_dim, out_dim = 2 * self.act_dim) for index in self.sampled_indices]
            # self.param_counts = [self.get_params(self.list_of_arcs[index]) for index in self.sampled_indices]
            # self.capacities = [get_capacity(self.list_of_arcs[index], self.obs_dim, self.act_dim) for index in self.sampled_indices]
        # _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)
        self.multi_ghns = replicate(self.ghn, self.all_devices)
        for i, device in enumerate(self.all_devices):
            self.multi_ghns[i].default_edges = self.multi_ghns[i].default_edges.to(device)
            self.multi_ghns[i].default_node_feat = self.multi_ghns[i].default_node_feat.to(device)
            sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds[i*self.num_current_models_per_device:(i+1)*self.num_current_models_per_device]).view(-1,1)
            _, embeddings = self.multi_ghns[i](self.current_model[i*self.num_current_models_per_device:(i+1)*self.num_current_models_per_device], return_embeddings=True, shape_ind = sampled_shape_inds.to(device))


    def change_graph(self, repeat_sample = False):
        self.re_query_uniform_weights(repeat_sample)


    def forward(self, state, track=True):
        batch_per_net = int(state.shape[0]//len(self.current_model))

        if track:
            self.shape_ind_per_state_dim = torch.cat([self.current_shape_inds_vec[i].repeat(batch_per_net,1) for i in range(len(self.current_model))])
            self.arch_per_state_dim = torch.cat([self.current_archs[i].repeat(batch_per_net,1) for i in range(len(self.current_model))])
            self.sampled_indices_per_state_dim = torch.cat([torch.tensor([self.sampled_indices[i]]).repeat(batch_per_net) for i in range(len(self.current_model))])
            x = gather([self.current_model[i](state[i*batch_per_net:(i+1)*batch_per_net].to(self.device_model_list[i])) for i in range(len(self.current_model))], self.device)
        else:
            x = torch.cat([self.current_model[i](state[i*batch_per_net:(i+1)*batch_per_net]) for i in range(len(self.current_model))])

        

        if len(x.shape) == 1:    
            mu = x[:x.shape[-1]//2]
            log_std = x[x.shape[-1]//2:]
        else:
            mu = x[:,:x.shape[-1]//2]
            log_std = x[:,x.shape[-1]//2:]


        # log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)        
        return mu, log_std    

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)

        return action, log_prob


    def sample(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mu)
    

    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


    def get_logprob(self,obs, actions, epsilon=1e-6):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(1, keepdim=True)
        return log_prob