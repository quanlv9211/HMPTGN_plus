import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from script.hgcn.layers.hyplayers import HMPGCNConv, HypMPGRU
from script.hgcn.manifolds import PoincareBall
from script.models.BaseModel import BaseModel
import geotorch

class HMPTGN(BaseModel):
    def __init__(self, args):
        super(HMPTGN, self).__init__(args)
        self.manifold_name = args.manifold
        self.manifold = PoincareBall()
        self.c = [args.curvature] * 3
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)), requires_grad=True)
        self.linear = nn.Linear(args.nfeat, args.nhid)
        self.hidden_initial = torch.ones(args.num_nodes, args.nhid).to(args.device)
        self.init_hiddens()
        self.use_hta = args.use_hta
        self.layer1 = HMPGCNConv(self.manifold, 2 * args.nout, args.nhid, self.c[0], self.c[1],
                                   dropout=args.dropout)
        self.gru = HypMPGRU(args, c=self.c[2])
        self.nhid = args.nhid
        self.nout = args.nout
        self.cat = True
        self.r = nn.Linear(self.nout, 1, bias=False)
        self.Q = nn.Linear(self.nhid, self.nout, bias=False)
        self.num_window = args.nb_window
        self.reset_parameters()

    def reset_parameters(self):
        geotorch.orthogonal(self.Q, "weight")
        geotorch.orthogonal(self.r, "weight")
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)

    def init_hiddens(self):
        self.hiddens = [self.initHyperX(self.hidden_initial)] * self.num_window
        return self.hiddens

    def HMP_weighted_hiddens(self, hidden_window, c):
        # temporal self-attention
        tmp = torch.tanh(self.Q(hidden_window))
        e = self.r(tmp)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [-1, self.num_window, self.nout])  # N x T x D
        a = torch.reshape(a, [a.shape[1], a.shape[0], a.shape[2]])               # N x T x 1

        z = self.manifold.p2k(hidden_window_new, c) # N x T x D
        lamb = self.manifold.lorenz_factor(z, c, keepdim=True)  # N x T x 1
        w = lamb * a    # N x T x 1
        w_sum = w.sum(1, keepdim=True).squeeze(-1) # N x 1
        z = z * w                    # N x T x D
        z_t = z.sum(1, keepdim=True).squeeze(1) # N x D
        z_t = z_t / w_sum
        output = self.manifold.k2p(z_t, c)
        return output


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

    def htc(self, x):
        x = self.manifold.proj(x, self.c[2])
        h = self.manifold.proj(self.hiddens[-1], self.c[2])
        return self.manifold.sqdist(x, h, self.c[2]).mean()

    def forward(self, edge_index, x=None, weight=None):
        if x is None:
            x = self.initHyperX(self.linear(self.feat), self.c[0])
        else:
            x = self.initHyperX(self.linear(x), self.c[0])

        if self.cat:
            x = torch.cat([x, self.hiddens[-1]], dim=1)
            x = self.initHyperX(x, self.c[0])

        x = self.layer1(x, edge_index)

        # GRU layer
        hlist = torch.cat([hidden for hidden in self.hiddens], dim=0)
        h = self.HMP_weighted_hiddens(hlist, self.c[2])
        x = self.gru(x, h)
        return x