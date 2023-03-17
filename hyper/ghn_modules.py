from hyper.model import *
import time

from hyper.utils import capacity, default_device

import numpy as np
import copy



def get_activation(activation):
    if activation is not None:
        if activation == 'relu':
            f = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            f = nn.LeakyReLU()
        elif activation == 'selu':
            f = nn.SELU()
        elif activation == 'elu':
            f = nn.ELU()
        elif activation == 'rrelu':
            f = nn.RReLU()
        elif activation == 'sigmoid':
            f = nn.Sigmoid()
        else:
            raise NotImplementedError(activation)
    else:
        f = nn.Identity()

    return f


def named_layered_modules(model):
    if hasattr(model, 'module'):  # in case of multigpu model
        model = model.module
    layers = model._n_cells if hasattr(model, '_n_cells') else 1
    layered_modules = [[] for _ in range(layers)]
    for module_name, m in model.named_modules():
        is_w = hasattr(m, 'weight') and m.weight is not None
        is_b = hasattr(m, 'bias') and m.bias is not None

        if is_w or is_b:
            if module_name.startswith('module.'):
                module_name = module_name[module_name.find('.') + 1:]
            cell_ind = get_cell_ind(module_name, layers)
            if is_w:
                layered_modules[cell_ind].append(
                    {'param_name': module_name + '.weight', 'module': m, 'is_w': True, 'sz': m.weight.shape})
            if is_b:
                layered_modules[cell_ind].append(
                    {'param_name': module_name + '.bias', 'module': m, 'is_w': False, 'sz': m.bias.shape})

    return layered_modules


class MLP_GHN(nn.Module):
    r"""
    Graph HyperNetwork based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    (https://arxiv.org/abs/1810.05749)

    """
    def __init__(self,
                 max_shape,
                 num_classes,
                 num_observations,
                 hypernet='gatedgnn',
                 decoder='conv',
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 hid=32,
                 device = 'cuda',
                 debug_level=0):
        super(MLP_GHN, self).__init__()

        assert len(max_shape) == 4, max_shape
        self.layernorm = layernorm
        self.weight_norm = weight_norm
        self.ve = ve
        self.debug_level = debug_level
        self.num_classes = num_classes

        if layernorm:
            self.ln = nn.LayerNorm(hid)

        self.embed = torch.nn.Embedding(3, hid)

        self.shape_enc3 = nn.Linear(1,hid).to(device)
        if hypernet == 'gatedgnn':
            self.gnn = GatedGNN(in_features=hid, ve=False)
        elif hypernet == 'mlp':
            self.gnn = MLP(in_features=hid, hid=(hid, hid))
        else:
            raise NotImplementedError(hypernet)

        if decoder == 'conv':
            fn_dec, layers = ConvDecoder, (hid * 4, hid * 8)
        elif decoder == 'mlp':
            fn_dec, layers = MLPDecoder, (hid * 2, )
        else:
            raise NotImplementedError(decoder)
        self.decoder = fn_dec(in_features=hid,
                              hid=layers,
                              out_shape=max_shape,
                              num_classes=num_classes)

        max_ch = max(max_shape[:2])
        self.decoder_1d = MLP(hid, hid=(hid * 2, 2 * max_ch),
                              last_activation=None)
        self.bias_class = nn.Sequential(nn.ReLU(),
                                        nn.Linear(max_ch, num_classes))

        self.default_edges = torch.zeros((50,4))
        self.default_edges[:,0] = torch.arange(len(self.default_edges[:,0]))
        self.default_edges[:,1] = torch.arange(len(self.default_edges[:,0])) + 1
        self.default_edges[:,2] = 1
        self.default_edges = self.default_edges.long().to(device)
        self.default_edges = nn.Parameter(self.default_edges, requires_grad=False)

        self.default_node_feat = torch.zeros(50).long().to(device)
        self.default_node_feat = nn.Parameter(self.default_node_feat, requires_grad=False)




    @staticmethod
    def load(checkpoint_path, config, debug_level=1, device=default_device(), verbose=False):
        state_dict = torch.load(checkpoint_path, map_location=device)
        ghn = MLP_GHN(**config, debug_level=debug_level, device = device).to(device).eval()
        ghn.load_state_dict(state_dict)
        if verbose:
            print('GHN with {} parameters loaded from epoch {}.'.format(capacity(ghn)[1], 1234))
        return ghn


    def forward(self, nets_torch, shape_ind = None, return_embeddings=False, predict_class_layers=True, bn_train=True):
        r"""
        Predict parameters for a list of >=1 networks.
        :param nets_torch: one network or a list of networks, each is based on nn.Module.
                           In case of evaluation, only one network can be passed.
        :param graphs: GraphBatch object in case of training.
                       For evaluation, graphs can be None and will be constructed on the fly given the nets_torch in this case.
        :param return_embeddings: True to return the node embeddings obtained after the last graph propagation step.
                                  return_embeddings=True is used for property prediction experiments.
        :param predict_class_layers: default=True predicts all parameters including the classification layers.
                                     predict_class_layers=False is used in fine-tuning experiments.
        :param bn_train: default=True sets BN layers in nets_torch into the training mode (required to evaluate predicted parameters)
                        bn_train=False is used in fine-tuning experiments
        :return: nets_torch with predicted parameters and node embeddings if return_embeddings=True
        """

        param_groups, params_map = self._map_net_params(nets_torch, self.debug_level > 0)
        # param_groups1, params_map1 = self._map_net_params(nets_torch[0], self.debug_level > 0)
        # param_groups2, params_map2 = self._map_net_params(nets_torch[1], self.debug_level > 0)

        # x_before_gnn = self.shape_enc(params_map, predict_class_layers=predict_class_layers)
        # x_before_gnn = self.shape_enc2(shape_ind)
        x_before_gnn = self.shape_enc3(shape_ind)

        all_graph_edges = []
        all_node_feat = []
        nets_torch = [nets_torch] if isinstance(nets_torch, nn.Module) else nets_torch
        for i in range(len(nets_torch)):
            graph_edges = copy.deepcopy(self.default_edges[0:len(list((nets_torch if isinstance(nets_torch, list) else [nets_torch])[i].named_parameters()))])
            node_feat = copy.deepcopy(self.default_node_feat[0:graph_edges.shape[0] + 1])
            graph_edges[:,3] = i
            node_feat += i

            all_graph_edges.append(graph_edges)
            all_node_feat.append(node_feat)

        all_graph_edges = torch.cat(all_graph_edges, dim=0)
        all_node_feat = torch.cat(all_node_feat, dim=0)

        # graph_edges = self.default_edges[0:len(list((nets_torch if isinstance(nets_torch, list) else [nets_torch])[0].named_parameters()))]
        # node_feat = self.default_node_feat[0:graph_edges.shape[0] + 1]

        # Update node embeddings using a GatedGNN, MLP or another model
        x = self.gnn(x_before_gnn, all_graph_edges, all_node_feat)

        if self.layernorm:
            x = self.ln(x)

        # Predict max-sized parameters for a batch of nets using decoders
        w = {}
        for key, inds in param_groups.items():
            if len(inds) == 0:
                continue
            x_ = x[torch.tensor(inds, device=x.device)]
            if key == 'cls_w':
                w[key] = self.decoder(x_, (1, 1), class_pred=False)
            elif key.startswith('4d'):
                sz = tuple(map(int, key.split('-')[1:]))
                w[key] = self.decoder(x_, sz, class_pred=False)
            else:
                w[key] = self.decoder_1d(x_).view(len(inds), 2, -1)#.clone()
                if key == 'cls_b':
                    w[key] = self.bias_class(w[key])

        # Transfer predicted parameters (w) to the networks
        n_tensors, n_params = 0, 0
        for matched, key, w_ind in params_map.values():

            if w_ind is None:
                continue  # e.g. pooling

            if not predict_class_layers and key in ['cls_w', 'cls_b']:
                continue  # do not set the classification parameters when fine-tuning

            m, sz, is_w = matched['module'], matched['sz'], matched['is_w']
            for it in range(2 if (len(sz) == 1 and is_w) else 1):

                if len(sz) == 1:
                    # separately set for BN/LN biases as they are
                    # not represented as separate nodes in graphs
                    w_ = w[key][w_ind][1 - is_w + it]
                    if it == 1:
                        assert (type(m) in NormLayers and key == '1d'), \
                            (type(m), key)
                else:
                    w_ = w[key][w_ind]

                sz_set = self._set_params(m, self._tile_params(w_, sz), is_w=is_w & ~it)
                n_tensors += 1
                n_params += torch.prod(torch.tensor(sz_set))

        if not self.training and bn_train:

            def bn_set_train(module):
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.training = True
            for net in nets_torch:
                net.apply(bn_set_train)
            # nets_torch.apply(bn_set_train)  # set BN layers to the training mode to enable evaluation without having running statistics

        return (nets_torch, x) if return_embeddings else nets_torch


    def _map_net_params(self, nets_torch, sanity_check=False):
        r"""
        Matches the parameters in the models with the nodes in the graph.
        Performs additional steps.
        :param graphs: GraphBatch object
        :param nets_torch: a single neural network of a list
        :param sanity_check:
        :return: mapping, params_map
        """
        mapping = {}
        params_map = {}

        nets_torch = [nets_torch] if type(nets_torch) not in [tuple, list] else nets_torch

        param_ind = 0
        node_infos = []
        n_nodes = []
        for j in range(len(nets_torch)):
            len_node_info = len(list(nets_torch[j].named_parameters()))
            node_infos.append([[(i+1, name, 'conv' if 'weight' in name else 'bias', (3,16,1,1) if 'weight' in name else None, True if i==(len_node_info-2) else False, True if i==(len_node_info-1) else False) for i, (name, param) in enumerate(nets_torch[j].named_parameters())]])
            n_nodes.append(len_node_info + 1)
        # for b, net in enumerate(nets_torch):
        for b, (node_info, net) in enumerate(zip(node_infos, nets_torch)):        
            target_modules = named_layered_modules(net)

            # param_ind = torch.sum(graphs.n_nodes[:b]).item()
            param_ind = sum(n_nodes[:b])

            for cell_id in range(len(node_info)):
                matched_names = []
                for (node_ind, param_name, name, sz, last_weight, last_bias) in node_info[cell_id]:

                    matched = []
                    for m in target_modules[cell_id]:
                        if m['param_name'].startswith(param_name):
                            matched.append(m)
                            if not sanity_check:
                                break
                    if len(matched) > 1:
                        raise ValueError(cell_id, node_ind, param_name, name, [
                            (t, (m.weight if is_w else m.bias).shape) for
                            t, m, is_w in matched])
                    elif len(matched) == 0:
                        if sz is not None:
                            params_map[param_ind + node_ind] = ({'sz': sz}, None, None)

                        if sanity_check:
                            for pattern in ['input', 'sum', 'concat', 'pool', 'glob_avg', 'msa', 'cse']:
                                good = name.find(pattern) >= 0
                                if good:
                                    break
                            assert good, \
                                (cell_id, param_name, name,
                                 node_info[cell_id],
                                 target_modules[cell_id])
                    else:
                        matched_names.append(matched[0]['param_name'])
                        sz = matched[0]['sz']
                        if len(sz) == 1:
                            key = 'cls_b' if last_bias else '1d'
                        else:
                            key = 'cls_w'
                        # else:
                        #     key = '4d-%d-%d' % ((1, 1) if len(sz) == 2 else sz[2:])
                        if key not in mapping:
                            mapping[key] = []
                        params_map[param_ind + node_ind] = (matched[0], key, len(mapping[key]))
                        mapping[key].append(param_ind + node_ind)

                assert len(matched_names) == len(set(matched_names)), (
                    'all matched names must be unique to avoid predicting the same paramters for different moduels',
                    len(matched_names), len(set(matched_names)))
                matched_names = set(matched_names)

                # Prune redundant ops in Network by setting their params to None
                for m in target_modules[cell_id]:
                    if m['is_w'] and m['param_name'] not in matched_names:
                        m['module'].weight = None
                        if hasattr(m['module'], 'bias') and m['module'].bias is not None:
                            m['module'].bias = None

        return mapping, params_map


    def _tile_params(self, w, target_shape):
        r"""
        Makes the shape of predicted parameter tensors the same as the target shape by tiling/slicing across channels dimensions.
        :param w: predicted tensor, for example of shape (64, 64, 11, 11)
        :param target_shape: tuple, for example (512, 256, 3, 3)
        :return: tensor of shape target_shape
        """
        t, s = target_shape, w.shape

        # Slice first to avoid tiling a larger tensor
        if len(t) == 1:
            if len(s) == 2:
                w = w[:min(t[0], s[0]), 0]
            elif len(s) > 2:
                w = w[:min(t[0], s[0]), 0, 0, 0]
        elif len(t) == 2:
            if len(s) > 2:
                w = w[:min(t[0], s[0]), :min(t[1], s[1]), 0, 0]
        else:
            w = w[:min(t[0], s[0]), :min(t[1], s[1]), :min(t[2], s[2]), :min(t[3], s[3])]

        s = w.shape
        assert len(s) == len(t), (s, t)

        # Tile out_channels
        if t[0] > s[0]:
            n_out = t[0] // s[0] + 1
            if len(t) == 1:
                w = w.repeat(n_out)[:t[0]]
            elif len(t) == 2:
                w = w.repeat((n_out, 1))[:t[0]]
            else:
                w = w.repeat((n_out, 1, 1, 1))[:t[0]]

        # Tile in_channels
        if len(t) > 1:
            if t[1] > s[1]:
                n_in = t[1] // s[1] + 1
                if len(t) == 2:
                    w = w.repeat((1, n_in))[:, :t[1]]
                else:
                    w = w.repeat((1, n_in, 1, 1))[:, :t[1]]

        # Chop out any extra bits tiled
        if len(t) == 1:
            w = w[:t[0]]
        elif len(t) == 2:
            w = w[:t[0], :t[1]]
        else:
            w = w[:t[0], :t[1], :t[2], :t[3]]

        return w


    def _set_params(self, module, tensor, is_w):
        r"""
        Copies the predicted parameter tensor to the appropriate field of the module object.
        :param module: nn.Module
        :param tensor: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: the shape of the copied tensor
        """
        if self.weight_norm:
            tensor = self._normalize(module, tensor, is_w)
        key = 'weight' if is_w else 'bias'
        target_param = module.weight if is_w else module.bias
        sz_target = target_param.shape
        if self.training:
            module.__dict__[key] = tensor  # set the value avoiding the internal logic of PyTorch
            # update parameters, so that named_parameters() will return tensors
            # with gradients (for multigpu and other cases)
            module._parameters[key] = tensor
        else:
            assert isinstance(target_param, nn.Parameter) or isinstance(target_param, torch.Tensor), type(target_param)
            # copy to make sure there is no sharing of memory
            target_param.data = tensor.clone()

        set_param = module.weight if is_w else module.bias
        assert sz_target == set_param.shape, (sz_target, set_param.shape)
        return set_param.shape


    def _normalize(self, module, p, is_w):
        r"""
        Normalizes the predicted parameter tensor according to the Fan-In scheme described in the paper.
        :param module: nn.Module
        :param p: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: normalized predicted tensor
        """
        if p.dim() > 1:

            sz = p.shape

            if len(sz) > 2 and sz[2] >= 11 and sz[0] == 1:
                assert isinstance(module, PosEnc), (sz, module)
                return p    # do not normalize positional encoding weights

            no_relu = len(sz) > 2 and (sz[1] == 1 or sz[2] < sz[3])
            if no_relu:
                # layers not followed by relu
                beta = 1.
            else:
                # for layers followed by rely increase the weight scale
                beta = 2.

            # fan-out:
            # p = p * (beta / (sz[0] * p[0, 0].numel())) ** 0.5

            # fan-in:
            p = p * (beta / p[0].numel()) ** 0.5

        else:

            if is_w:
                p = 2 * torch.sigmoid(0.5 * p)  # BN/LN norm weight is [0,2]
            else:
                p = torch.tanh(0.2 * p)         # bias is [-1,1]

        return p

class SimpleShapeEncoder(nn.Module):
    """ A simpler version of the Shape encoder. NO LOOKUPS. Directly feed the shape_ind. 
    Initialize with the array of unique shapes needed to be encoded
    """
    def __init__(self, hid, channels):
        super(SimpleShapeEncoder, self).__init__()

        # self.channels = np.unique([0,1,2,4,8,16,32,64,128,256,512,1024,num_observations,num_classes])
        # self.channels = channels
        self.embed_channel = torch.nn.Embedding(len(channels) + 1, hid)
    
    def forward(self, shape_ind):
        """ 
        Args: 
            x: Encoded nodes
            shape_ind: a tensor array giving a unique value to each shape (therefore bypassing lookup). This way gradients can pass through this vector.
        """
        shape_embed = torch.cat(
            (self.embed_channel(shape_ind[:, 0]),
            #  self.embed_channel(shape_ind[:, 1])
             ), dim=1)
        return shape_embed

class ShapeEncoder(nn.Module):
    def __init__(self, hid, num_observations, num_classes, max_shape, debug_level=0):
        super(ShapeEncoder, self).__init__()

        assert max_shape[2] == max_shape[3], max_shape
        self.debug_level = debug_level
        self.num_classes = num_classes
        self.ch_steps = (2**3, 2**6, 2**12, 2**13)
        self.channels = np.unique([1, 3, 4, num_classes, num_observations, 8, 17, 376] +
                                  list(range(self.ch_steps[0], self.ch_steps[1], 2**3)) +
                                  list(range(self.ch_steps[1], self.ch_steps[2], 2**4)) +
                                  list(range(self.ch_steps[2], self.ch_steps[3] + 1, 2**5)))

        self.spatial = np.unique(list(range(1, max(12, max_shape[3]), 2)) + [14, 16])

        # create a look up dictionary for faster determining the channel shape index
        # include shapes not seen during training by assigning them the the closest seen values
        self.channels_lookup = {c: i for i, c in enumerate(self.channels)}
        self.channels_lookup_training = copy.deepcopy(self.channels_lookup)
        for c in range(4, self.ch_steps[0]):
            self.channels_lookup[c] = self.channels_lookup[self.ch_steps[0]]  # 4-7 channels will be treated as 8 channels
        for c in range(1, self.channels[-1]):
            if c not in self.channels_lookup:
                self.channels_lookup[c] = self.channels_lookup[self.channels[np.argmin(abs(self.channels - c))]]

        self.spatial_lookup = {c: i for i, c in enumerate(self.spatial)}
        self.spatial_lookup_training = copy.deepcopy(self.spatial_lookup)
        self.spatial_lookup[2] = self.spatial_lookup[3]  # 2x2 (not seen during training) will be treated as 3x3
        for c in range(1, self.spatial[-1]):
            if c not in self.spatial_lookup:
                self.spatial_lookup[c] = self.spatial_lookup[self.spatial[np.argmin(abs(self.spatial - c))]]

        n_ch, n_s = len(self.channels), len(self.spatial)
        # self.embed_spatial = torch.nn.Embedding(n_s + 1, hid // 4)
        self.embed_channel = torch.nn.Embedding(n_ch + 1, hid // 2)

        # self.register_buffer('dummy_ind', torch.tensor([n_ch, n_ch, n_s, n_s], dtype=torch.long).view(1, 4),
        #                      persistent=False)
        self.register_buffer('dummy_ind', torch.tensor([n_ch, n_ch, 0, 0], dtype=torch.long).view(1, 4),
                             persistent=False)


    def forward(self, params_map, predict_class_layers=True):
        shape_ind = self.dummy_ind.repeat(len(params_map) + 1, 1)

        self.printed_warning = False
        for node_ind in params_map:
            sz = params_map[node_ind][0]['sz']
            if sz is None:
                continue

            sz_org = sz
            if len(sz) == 1:
                sz = (sz[0], 1)
            if len(sz) == 2:
                sz = (sz[0], sz[1], 1, 1)
            assert len(sz) == 4, sz

            # if not predict_class_layers and params_map[node_ind][1] in ['cls_w', 'cls_b']:
            #     # keep the classification shape as though the GHN is used on the dataset it was trained on
            #     sz = (self.num_classes, *sz[1:])

            recognized_sz = 0
            for i in range(4):
                # if not in the dictionary, then use the maximum shape
                if i < 2:  # for out/in channel dimensions
                    shape_ind[node_ind, i] = self.channels_lookup[sz[i] if sz[i] in self.channels_lookup else self.channels[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.channels_lookup_training)
                else:  # for kernel height/width
                    shape_ind[node_ind, i] = self.spatial_lookup[sz[i] if sz[i] in self.spatial_lookup else self.spatial[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.spatial_lookup_training)

            if self.debug_level and not self.printed_warning:  # print a warning once per architecture
                if recognized_sz != 4:
                    print( 'WARNING: unrecognized shape %s, so the closest shape at index %s will be used instead.' % (
                        sz_org, ([self.channels[c.item()] if i < 2 else self.spatial[c.item()] for i, c in
                                  enumerate(shape_ind[node_ind])])))
                    self.printed_warning = True

        shape_embed = torch.cat(
            (self.embed_channel(shape_ind[:, 0]),
             self.embed_channel(shape_ind[:, 1])
            #  self.embed_spatial(shape_ind[:, 2]),
            #  self.embed_spatial(shape_ind[:, 3])
             ), dim=1)

        # return x + shape_embed
        return shape_embed


class ConvDecoder(nn.Module):
    def __init__(self,
                 in_features=64,
                 hid=(128, 256),
                 out_shape=None,
                 num_classes=None):
        super(ConvDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.fc = nn.Sequential(nn.Linear(in_features,
                                          hid[0] * np.prod(out_shape[2:])),
                                nn.ReLU())

        conv = []
        for j, n_hid in enumerate(hid):
            n_out = np.prod(out_shape[:2]) if j == len(hid) - 1 else hid[j + 1]
            conv.extend([nn.Conv2d(n_hid, n_out, 1),
                         get_activation(None if j == len(hid) - 1 else 'relu')])

        self.conv = nn.Sequential(*conv)
        self.class_layer_predictor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_shape[0], num_classes, 1))


    def forward(self, x, max_shape=(1,1), class_pred=False):

        N = x.shape[0]
        x = self.fc(x).view(N, -1, *self.out_shape[2:])  # N,128,11,11
        out_shape = self.out_shape
        if sum(max_shape) > 0:
            x = x[:, :, :max_shape[0], :max_shape[1]]
            out_shape = (out_shape[0], out_shape[1], max_shape[0], max_shape[1])

        x = self.conv(x).view(N, *out_shape)  # N, out, in, h, w

        if class_pred:
            x = self.class_layer_predictor(x[:, :, :, :, 0])  # N, num_classes, 64, 1
            x = x[:, :, :, 0]  # N, num_classes, 64

        return x



class MLPDecoder(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(64,),
                 out_shape=None,
                 num_classes=None):
        super(MLPDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.mlp = MLP(in_features=in_features,
                       hid=(*hid, np.prod(out_shape)),
                       activation='relu',
                       last_activation=None)
        self.class_layer_predictor = nn.Sequential(
            get_activation('relu'),
            nn.Linear(hid[0], num_classes * out_shape[0]))


    def forward(self, x, max_shape=(1,1), class_pred=False):
        if class_pred:
            x = list(self.mlp.fc.children())[0](x)  # shared first layer
            x = self.class_layer_predictor(x)  # N, 1000, 64, 1
            x = x.view(x.shape[0], self.num_classes, self.out_shape[1])
        else:
            x = self.mlp(x).view(-1, *self.out_shape)
            if sum(max_shape) > 0:
                x = x[:, :, :, :max_shape[0], :max_shape[1]]
        return x


class MLP(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(32, 32),
                 activation='relu',
                 last_activation='same'):
        super(MLP, self).__init__()

        assert len(hid) > 0, hid
        fc = []
        for j, n in enumerate(hid):
            fc.extend([nn.Linear(in_features if j == 0 else hid[j - 1], n),
                       get_activation(last_activation if
                                      (j == len(hid) - 1 and
                                       last_activation != 'same')
                                      else activation)])
        self.fc = nn.Sequential(*fc)


    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            x = x[0]
        return self.fc(x)

class GatedGNN(nn.Module):
    r"""
    Gated Graph Neural Network based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    Performs node feature propagation according to Eq. 3 and 4 in the paper.
    """
    def __init__(self,
                 in_features=32,
                 ve=False,
                 T=1):
        """
        Initializes Gated Graph Neural Network.
        :param in_features: how many features in each node.
        :param ve: use virtual edges defined according to Eq. 4 in the paper.
        :param T: number of forward+backward graph traversal steps.
        """
        super(GatedGNN, self).__init__()
        self.in_features = in_features
        self.hid = in_features
        self.ve = ve
        self.T = T
        self.mlp = MLP(in_features, hid=( (self.hid // 2) if ve else self.hid, self.hid))
        if ve:
            self.mlp_ve = MLP(in_features, hid=(self.hid, self.hid))

        self.gru = nn.GRUCell(self.hid, self.hid)  # shared across all nodes/cells in a graph


    def forward(self, x, edges, node_graph_ind):
        r"""
        Updates node features by sequentially traversing the graph in the forward and backward directions.
        :param x: (N, C) node features, where N is the total number of nodes in a batch of B graphs, C is node feature dimensionality.
        :param edges: (M, 4) tensor of edges, where M is the total number of edges;
                       first column in edges is the row indices of edges,
                       second column in edges is the column indices of edges,
                       third column in edges is the shortest path distance between the nodes,
                       fourth column in edges is the graph indices (from 0 to B-1) within a batch for each edge.
        :param node_graph_ind: (N,) tensor of graph indices (from 0 to B-1) within a batch for each node.
        :return: updated (N, C) node features.
        """

        assert x.dim() == 2 and edges.dim() == 2 and edges.shape[1] == 4, (x.shape, edges.shape)
        n_nodes = torch.unique(node_graph_ind, return_counts=True)[1]

        B, C = len(n_nodes), x.shape[1]  # batch size, features

        ve, edge_graph_ind = edges[:, 2], edges[:, 3]

        assert n_nodes.sum() == len(x), (n_nodes.sum(), x.shape)

        is_1hop = ve == 1
        if self.ve:
            ve = ve.view(-1, 1)   # according to Eq. 4 in the paper

        traversal_orders = [1, 0]  # forward, backward

        edge_offset = torch.cumsum(F.pad(n_nodes[:-1], (1, 0)), 0)[edge_graph_ind]
        node_inds = torch.cat([torch.arange(n) for n in n_nodes]).to(x.device).view(-1, 1)

        # Parallelize computation of indices and masks of one/all hops
        # This will slightly speed up the operations in the main loop
        # But indexing of the GPU tensors (used in the main loop) for some reason remains slow, see
        # https://github.com/pytorch/pytorch/issues/29973 for more info
        all_nodes = torch.arange(edges[:, 1].max() + 1, device=x.device)
        masks_1hop, masks_all = {}, {}
        for order in traversal_orders:
            masks_all[order] = edges[:, order].view(1, -1) == all_nodes.view(-1, 1)
            masks_1hop[order] = masks_all[order] & is_1hop.view(1, -1)
        mask2d = node_inds == all_nodes.view(1, -1)
        edge_graph_ind = edge_graph_ind.view(-1, 1).expand(-1, C)

        hx = x  # initial hidden node features

        seen_inds = []
        # Main loop
        for t in range(self.T):
            for order in traversal_orders:  # forward, backward
                start = edges[:, 1 - order] + edge_offset                           # node indices from which the message will be passed further
                for node in (all_nodes if order else torch.flipud(all_nodes)):

                    # Compute the message by aggregating features from neighbors
                    e_1hop = torch.nonzero(masks_1hop[order][node, :]).view(-1)
                    m = self.mlp(hx[start[e_1hop]])                                 # transform node features of all 1-hop neighbors
                    m = torch.zeros(B, C, dtype=m.dtype, device=m.device).scatter_add_(0, edge_graph_ind[e_1hop], m)     # sum the transformed features into a (B,C) tensor
                    if self.ve:
                        e = torch.nonzero(masks_all[order][node, :]).view(-1)       # virtual edges connected to node
                        m_ve = self.mlp_ve(hx[start[e]]) / ve[e].to(m)              # transform node features of all ve-hop neighbors
                        m = m.scatter_add_(0, edge_graph_ind[e], m_ve)              # sum m and m_ve according to Eq. 4 in the paper

                    # Udpate node hidden states in parallel for a batch of graphs
                    ind = torch.nonzero(mask2d[:, node]).view(-1)
                    if B > 1:
                        m = m[node_graph_ind[ind]]
                    hx[ind] = self.gru(m, hx[ind]).to(hx)  # 'to(hx)' is to make automatic mixed precision work
                    seen_inds.extend(ind.tolist())

        return hx

