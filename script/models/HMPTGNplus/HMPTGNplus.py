import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from script.models.HMPTGNplus.layers import HMPGNNplus, HypMPGRU, HMPTemporal
from script.hgcn.manifolds import PoincareBall
from script.models.BaseModel import BaseModel
import geotorch

class HMPTGNplus(BaseModel):
    def __init__(self, args):
        super(HMPTGNplus, self).__init__(args)
        self.manifold_name = args.manifold
        self.manifold = PoincareBall()
        self.device = args.device

        self.c_min = args.min_curvature
        self.c_max = args.max_curvature
        self.c = nn.Parameter(torch.Tensor([args.curvature] * args.training_length), requires_grad=True)
        #self.c_memory = self.c.detach().clone()
        self.c_default = torch.Tensor([args.curvature]).to(self.device)
        
        self.window_size = args.hmp_casual_conv_kernel_size ** args.hmp_casual_conv_depth
        self.history_initial = torch.ones(args.num_nodes, args.nout).to(self.device)
        self.linear_init = nn.Linear(args.nfeat, args.nhid, bias=False)
        self.feat = nn.Parameter((torch.ones(args.num_nodes, args.nfeat)), requires_grad=True)
        self.hiddens = None
        self.nout = args.nout
        self.reset_parameters()
        self.init_hiddens()
        self.cat = True
        self.c_max = args.max_curvature
        self.c_min = args.min_curvature
        self.spatial_module = HMPGNNplus(self.manifold, args.nhid, args.nout, args.spatial_order)
        self.temporal_module = HMPTemporal(self.manifold, args.nout, self.device, args.hmp_casual_conv_depth,
                                                    args.hmp_casual_conv_kernel_size, use_bias = args.bias, nonlinear='softmax')
        self.recurrent_module = HypMPGRU(args)

    def reset_parameters(self):
        glorot(self.feat)
        glorot(self.linear_init.weight)
        glorot(self.history_initial)

    def initHyperX(self, x, c=1.0):
        if self.manifold_name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c), c)
        return x

    def htc(self, x, c=1.0):
        x = self.manifold.proj(x, c)
        h = self.manifold.curvature_map(self.hiddens[-1], self.c_default, c)
        return self.manifold.sqdist(x, h, c).mean()

    def init_hiddens(self):
        self.hiddens = [self.initHyperX(self.history_initial, self.c_default).unsqueeze(0)] * self.window_size
        return self.hiddens

    # replace all nodes
    def update_hiddens_all_with(self, z_t):
        z_t = z_t.unsqueeze(0)
        self.hiddens.pop(0)  # [element0, element1, element2] remove the first element0
        self.hiddens.append(z_t.clone().detach().requires_grad_(False))  # [element1, element2, z_t]
        return z_t

    # replace current nodes state
    def update_hiddens_with(self, z_t, nodes):
        last_z = self.hiddens[-1].detach_().clone().requires_grad_(False)
        last_z[nodes, :] = z_t[nodes, :].detach_().clone().requires_grad_(False)
        last_z = last_z.unsqueeze(0)
        self.hiddens.pop(0)  # [element0, element1, element2] remove the first element0
        self.hiddens.append(last_z)  # [element1, element2, z_t]
        return last_z

    def forward(self, list_edge_index, timestamp, x=None, weight=None):

        ## add clipping
        if self.c[timestamp] < self.c_min or self.c[timestamp] > self.c_max:
            self.c[timestamp].data.clamp_(self.c_min, self.c_max)
        
        if x is None:
            x = self.initHyperX(self.linear(self.feat), self.c[timestamp])
        else:
            x = self.initHyperX(self.linear(x), self.c[timestamp])


        x_ = self.spatial_module(x, list_edge_index, self.c[timestamp])
        x = self.manifold.mobius_add(x, x_, self.c[timestamp]) #comment this line in Disease

        hiddens = torch.cat(self.hiddens, dim=0)
        hiddens = self.manifold.curvature_map(hiddens, self.c_default, self.c[timestamp])
        general_hiddens = self.temporal_module(hiddens, self.c[timestamp])[-1]

        h = self.recurrent_module(x, general_hiddens, self.c[timestamp])
        h_ = self.manifold.curvature_map(h, self.c[timestamp], self.c_default)
        return h, self.c[timestamp], h_
    