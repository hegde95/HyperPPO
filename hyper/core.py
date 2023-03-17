from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn.parallel import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather

from hyper.ghn_modules import MLP_GHN, MlpNetwork

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class hyperActor(nn.Module):

    def __init__(self, 
                act_dim, 
                obs_dim, 
                allowable_layers, 
                meta_batch_size = 1,
                device = "cpu",
                architecture_sampling_mode = "biased",
                multi_gpu = True,
                std_mode = 'single',
                ):
        super().__init__()

        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.meta_batch_size = meta_batch_size
        self.architecture_sampling_mode = architecture_sampling_mode
        self.multi_gpu = multi_gpu
        assert std_mode in ['single', 'multi', 'arch_conditioned'], "std_mode must be one of ['single', 'multi', 'arch_conditioned']"
        self.std_mode = std_mode

        
        # initialize all devices for parallelization on multiple GPUs
        self._initialize_devices(device)

        # initialize all list of shape and architecture indices
        self._initialize_shape_arch_inidices(allowable_layers)

        # initialize all data required for architecture sampling
        self._initialize_architecture_smapling_data()

        # initialize the GHN
        self._initialize_ghn(self.obs_dim, self.act_dim)

        # initialize standard deviation vectors
        self._initialize_std()
        

    def _initialize_std(self):
        ''' Initializes the standard deviation vectors
        '''
        if self.std_mode == 'single':
            self.log_std = nn.Parameter(torch.zeros(1, np.prod(self.act_dim)))
        elif self.std_mode == 'multi':
            self.log_std = nn.ParameterList([
                nn.Parameter(torch.zeros(1, np.prod(self.act_dim)), requires_grad = False)
            for index in self.list_of_arc_indices
            ])
            pass
        elif self.std_mode == 'arch_conditioned':
            self.log_std = nn.Sequential(
                    layer_init(nn.Linear(self.arch_max_len, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 1), std=1.0),
                )
        else:
            raise NotImplementedError  


    def _initialize_architecture_smapling_data(self):
        ''' Initializes all the data required for architecture sampling
        '''
        if self.architecture_sampling_mode == "sequential":
            self.current_model_indices = np.arange(self.meta_batch_size)
        elif self.architecture_sampling_mode == "uniform":
            # self.current_model_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, replace = True)
            pass
        elif self.architecture_sampling_mode == "biased":
            self.arch_sampling_probs = []
            num_unique_num_layers = len(set([len(x) for x in self.list_of_arcs]))
            for i in self.list_of_arc_indices:
                num_layers = len(self.list_of_arcs[i])
                num_archs_with_same_num_layers = len([x for x in self.list_of_arcs if len(x) == num_layers])
                self.arch_sampling_probs.append(1/num_archs_with_same_num_layers)
            self.arch_sampling_probs = (1/num_unique_num_layers)*np.array(self.arch_sampling_probs)





    def _initialize_shape_arch_inidices(self, allowable_layers):
        ''' Creates:
            1. list_of_arcs: list of all possible architectures, sorted by total number of parameters
            2. list of shape indicators: list of shape indicators for each architecture, that can be used as an input to the GHN
            3. list of arc indices: list of indices of the architectures in list_of_arcs, can be used to sample architectures later
        '''

        list_of_allowable_layers = list(allowable_layers)
        self.list_of_arcs = []
        for k in range(1,5):
            self.list_of_arcs.extend(list(product(list_of_allowable_layers, repeat = k)))      
        # TDOD: DELETE THIS LINE, THIS IS DEBUG
        # for i in range(500):
        #     self.list_of_arcs.extend([
        #         [4, 4],
        #         [16],
        #         [16, 16, 16],
        #         [32, 32, 32],
        #         [64, 64, 64, 64],
        #         [128, 128, 128, 128],
        #         [256, 256, 256],
        #         [256, 256, 256, 256],                
        #     ])
        # for i in range(500):
        #     self.list_of_arcs.extend([
        #         [256, 256, 256, 128],                
        #     ])
        self.list_of_arcs.sort(key = lambda x:self.get_params(x))

        self._initialize_shape_inds()

        self.list_of_arc_indices = np.arange(len(self.list_of_arcs))
        self.all_models = [MlpNetwork(fc_layers=self.list_of_arcs[index], inp_dim = self.obs_dim, out_dim = self.act_dim) for index in self.list_of_arc_indices]
        # if self.std_mode == "multi":
        #     self.log_std = nn.ParameterList([nn.Parameter(torch.zeros(1, np.prod(self.act_dim))) for index in self.list_of_arc_indices])
        # shuffle the list of arcs indices
        np.random.shuffle(self.list_of_arc_indices)


    def _initialize_shape_inds(self):
        ''' Creates:
            1. list_of_shape_inds: list of shape indicators for each architecture, that can be used as an input to the GHN
            2. list_of_shape_inds_lenths: list of lengths of each shape indicator, needed since the shape indicators are not all the same length
        '''

        self.list_of_shape_inds = []
        for arc in self.list_of_arcs:
            shape_ind = [torch.tensor(0).type(torch.FloatTensor).to(self.device)]
            for layer in arc:
                shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
                shape_ind.append(torch.tensor(layer).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor(self.act_dim).type(torch.FloatTensor).to(self.device))
            shape_ind.append(torch.tensor(self.act_dim).type(torch.FloatTensor).to(self.device))
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


    def _initialize_devices(self, device):
        ''' Inititalize all devices since we are using multiple GPUs. device_model_list can be used later to assign models to devices quickly
        '''
        if self.multi_gpu:
            self.device = torch.device("cuda:0")            
            
            self.all_devices = [torch.device('cuda:{}'.format(i)) for i in range(torch.cuda.device_count())]
            self.num_current_models_per_device = int(self.meta_batch_size / len(self.all_devices)) 
            self.device_model_list = []
            for device in self.all_devices:
                self.device_model_list.extend([device for i in range(self.num_current_models_per_device)])
        else:
            self.device = device



    def _initialize_ghn(self, obs_dim, act_dim):
        ''' Initialize the GHN that takes in the shape indicators and outputs weights for that corresponding architecture
        '''

        config = {}
        config['max_shape'] = (256, 256, 1, 1)
        config['num_classes'] = 2 * act_dim
        config['num_observations'] = obs_dim
        config['weight_norm'] = True
        config['ve'] = 1 > 1
        config['layernorm'] = True
        config['hid'] = 16
        self.ghn_config = config
        self.ghn = MLP_GHN(**config,
                    debug_level=0, device=self.device).to(self.device)  



    def get_params(self, net):
        ''' Get the number of parameters in a MLP network architecture
        '''
        ct = 0
        ct += ((self.obs_dim + 1) *net[0])
        for i in range(len(net)-1):
            ct += ((net[i] + 1) * net[i+1])
        ct += ((net[-1] +1) * self.act_dim)
        return ct            

    def sample_arc_indices(self, mode = 'sequential'):
        ''' Sample the indices of the architectures to be used for the current model
            Sampling strategies:
            1. layer_biased: sample the indices of the architecture while making sure architectures with fewer layers are sampled more often
            2. sequential: sample the indices of the architecture sequentially
            3. uniform: sample the indices of the architecture uniformly
        '''
        if mode == 'biased':
            self.sampled_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, p = self.arch_sampling_probs, replace=False)
        elif mode == 'sequential':
            self.sampled_indices = self.list_of_arc_indices[self.current_model_indices]
            self.current_model_indices += self.meta_batch_size  
            if max(self.current_model_indices) >= len(self.list_of_arc_indices):
                self.current_model_indices = np.arange(self.meta_batch_size)
                # shuffle
                np.random.shuffle(self.list_of_arc_indices)
        elif mode == 'uniform':
            self.sampled_indices = np.random.choice(self.list_of_arc_indices, self.meta_batch_size, replace=False)
        else:
            raise NotImplementedError



    def set_graph(self, indices_vector, shape_ind_vec):
        ''' Set the graph to be used by the GHN. We can do this only by passing the indices of the 
            architectures we want to use and the shape indicators for those architectures. Then we estimate the 
            weights for those architectures and set it to the current model
        '''

        # delete gradients of the previous log_std, this speeds up training
        if self.std_mode == 'multi':
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = False
                self.log_std[i].grad = None
        self.sampled_indices = indices_vector
        # self.sampled_shape_inds = shape_ind_vec.view(-1)[shape_ind_vec.view(-1) != -1].unsqueeze(-1)
        self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
        self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]   
        self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
        assert (self.sampled_shape_inds == shape_ind_vec.view(-1)[shape_ind_vec.view(-1) != -1].unsqueeze(-1)).all(), 'Shape inds do not match'
        self.current_model = [self.all_models[i] for i in self.sampled_indices]
        self.current_archs = torch.tensor([list(self.list_of_arcs[index]) + [0]*(4-len(self.list_of_arcs[index])) for index in self.sampled_indices]).to(self.device)
        _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)
        if self.std_mode == 'multi':
            for i in self.sampled_indices:
                self.log_std[i].requires_grad = True       
        #     self.current_std = [self.log_std[i] for i in self.sampled_indices]


    def change_graph(self, repeat_sample = False):
        ''' Estimate the weights for the current models.
            If repeat_sample is True, then we re-estimate the weights for the same architectures (i.e. current models does not change)
            If repeat_sample is False, then we sample new architectures (i.e. change the current models) and estimate the weights for those architectures 
        '''
        if not repeat_sample:

            self.sample_arc_indices(mode = self.architecture_sampling_mode)
            
            self.current_shape_inds_vec = [self.list_of_shape_inds[index] for index in self.sampled_indices]
            self.list_of_sampled_shape_inds = [self.current_shape_inds_vec[k][:self.list_of_shape_inds_lenths[index]] for k,index in enumerate(self.sampled_indices)]

            self.current_archs = torch.tensor([list(self.list_of_arcs[index]) + [0]*(4-len(self.list_of_arcs[index])) for index in self.sampled_indices]).to(self.device) 
            self.current_model = [self.all_models[i] for i in self.sampled_indices]
            
            # self.param_counts = [self.get_params(self.list_of_arcs[index]) for index in self.sampled_indices]
            # self.capacities = [get_capacity(self.list_of_arcs[index], self.obs_dim, self.act_dim) for index in self.sampled_indices]
        
        if self.multi_gpu:
            self.multi_ghns = replicate(self.ghn, self.all_devices)
            for i, device in enumerate(self.all_devices):
                sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds[i*self.num_current_models_per_device:(i+1)*self.num_current_models_per_device]).view(-1,1)
                _, embeddings = self.multi_ghns[i](self.current_model[i*self.num_current_models_per_device:(i+1)*self.num_current_models_per_device], return_embeddings=True, shape_ind = sampled_shape_inds.to(device))
        else:
            self.sampled_shape_inds = torch.cat(self.list_of_sampled_shape_inds).view(-1,1)
            _, embeddings = self.ghn(self.current_model, return_embeddings=True, shape_ind = self.sampled_shape_inds)


    def forward(self, state, track=True):
        ''' Do a forward pass through the current models. We split the state batch into chunks of size batch_per_net and pass it through each of the current models
            track: if True, we track the shape indicators, architectures and indices of the current models. 
                We store this information if it is needed for architecture conditioned value functions
        '''
        batch_per_net = int(state.shape[0]//len(self.current_model))

        if track:
            self.shape_ind_per_state_dim = torch.cat([self.current_shape_inds_vec[i].repeat(batch_per_net,1) for i in range(len(self.current_model))])
            self.arch_per_state_dim = torch.cat([self.current_archs[i].repeat(batch_per_net,1) for i in range(len(self.current_model))])
            self.sampled_indices_per_state_dim = torch.cat([torch.tensor([self.sampled_indices[i]]).repeat(batch_per_net) for i in range(len(self.current_model))])
            
        if self.multi_gpu:    
            x = gather(parallel_apply(self.current_model, [state[i*batch_per_net:(i+1)*batch_per_net].to(self.device_model_list[i]) for i in range(len(self.current_model))]), self.device)
        else:
            x = torch.cat(parallel_apply(self.current_model, [state[i*batch_per_net:(i+1)*batch_per_net] for i in range(len(self.current_model))]))
        
        mu = x
        action_logstd = self.get_logstd(state, mu, batch_per_net)

        return mu, action_logstd    

    def get_logstd(self, state, mu, batch_per_net):
        if self.std_mode == 'single':
            return self.log_std.expand_as(mu)
            
        elif self.std_mode == 'multi':
            return torch.cat([self.log_std[i].expand(batch_per_net,6) for i in self.sampled_indices])



    ############################################################### forward helper functions, mostly only for debugging purposes ######################################################
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