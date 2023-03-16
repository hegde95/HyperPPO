import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEnc(nn.Module):
    def __init__(self, C, ks):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, C, ks, ks))

    def forward(self, x):
        return  x + self.weight
        
NormLayers = [nn.BatchNorm2d, nn.LayerNorm]


def get_cell_ind(param_name, layers=1):
    if param_name.find('cells.') >= 0:
        pos1 = len('cells.')
        pos2 = pos1 + param_name[pos1:].find('.')
        cell_ind = int(param_name[pos1: pos2])
    elif param_name.startswith('classifier') or param_name.startswith('auxiliary'):
        cell_ind = layers - 1
    elif layers == 1 or param_name.startswith('stem') or param_name.startswith('pos_enc'):
        cell_ind = 0
    else:
        cell_ind = None

    return cell_ind



class MlpNetwork(nn.Module):

    def __init__(self,
                 fc_layers=0,
                 inp_dim = 0,
                 out_dim = 0,
                 ):
        super(MlpNetwork, self).__init__()

        self.expected_input_sz = inp_dim
        if fc_layers[0] == 0:
            fc = [nn.Linear(inp_dim, out_dim)]
        else:
            fc = [nn.Linear(inp_dim, fc_layers[0])]
            for i in range(len(fc_layers) - 1):
                # assert fc_dim > 0, fc_dim
                fc.append(nn.ReLU(inplace=True))
                # fc.append(nn.Dropout(p=0.5, inplace=False))
                fc.append(nn.Linear(in_features=fc_layers[i], out_features=fc_layers[i+1]))
            fc.append(nn.ReLU(inplace=True))
            fc.append(nn.Linear(in_features=fc_layers[-1], out_features=out_dim))
        self.classifier = nn.Sequential(*fc)

    def forward(self, input):

        out = self.classifier(input)

        return out