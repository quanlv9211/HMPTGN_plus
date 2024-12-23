import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.inits import glorot
from script.models.BaseModel import BaseModel
from torch.nn import Parameter

MAX_LOGSTD = 10


# refer to: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gat.py
class DGAT(BaseModel):
    def __init__(self, args):
        super(DGAT, self).__init__(args)
        self.layer1 = GATConv(2 * args.nhid, args.nhid // 2, args.heads, dropout=args.dropout)
        self.layer2 = GATConv(args.nhid // 2 * args.heads, args.nhid, heads=1, dropout=args.dropout, concat=False)
        self.dropout1 = args.dropout
        self.dropout2 = args.dropout
        self.act = F.relu


# refer to https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py
class DGCN(BaseModel):
    def __init__(self, args):
        super(DGCN, self).__init__(args)
        self.layer1 = GCNConv(2 * args.nhid, 2 * args.nhid)
        self.layer2 = GCNConv(2 * args.nhid, args.nhid)
        self.dropout1 = 0.3
        self.dropout2 = 0.3
        self.act = F.relu
        self.Q = Parameter(torch.ones((args.nhid, args.nhid)), requires_grad=True)
        self.r = Parameter(torch.ones((args.nhid, 1)), requires_grad=True)
        self.reset_parameter()

    def weighted_hiddens(self, hidden_window):
        e = torch.matmul(torch.tanh(torch.matmul(hidden_window, self.Q)), self.r)
        e_reshaped = torch.reshape(e, (self.num_window, -1))
        a = F.softmax(e_reshaped, dim=0).unsqueeze(2)
        hidden_window_new = torch.reshape(hidden_window, [self.num_window, -1, self.nhid])
        s = torch.mean(a * hidden_window_new, dim=0)
        return s

    def reset_parameter(self):
        glorot(self.Q)
        glorot(self.r)
        glorot(self.feat)
        glorot(self.linear.weight)
        glorot(self.hidden_initial)