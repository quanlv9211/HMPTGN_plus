"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from script.hgcn.manifolds import PoincareBall, Hyperboloid
from torch_geometric.utils import to_dense_adj
import itertools
import geotorch


class HGATConv(nn.Module):
    """
    Hyperbolic graph convolution layer.。
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, act=F.leaky_relu,
                 dropout=0.6, att_dropout=0.6, use_bias=True, heads=2, concat=False):
        super(HGATConv, self).__init__()
        out_features = out_features * heads
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout, use_bias=use_bias)
        self.agg = HypAttAgg(manifold, c_in, out_features, att_dropout, heads=heads, concat=concat)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HGCNConv(nn.Module):
    """
    Hyperbolic graph convolution layer, from hgcn。
    """

    def __init__(self, manifold, in_features, out_features, c_in=1.0, c_out=1.0, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        super(HGCNConv, self).__init__()
        self.c_in = c_in
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout)
        self.agg = HypAgg(manifold, c_in, out_features, bias=use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HMPGCNConv(nn.Module):
    def __init__(self, manifold, in_features, out_features, c_in=1.0, c_out=1.0, dropout=0.0, act=F.leaky_relu,
                 use_bias=True, agg_type=1):
        super(HMPGCNConv, self).__init__()
        self.c_in = c_in
        self.linear = HypMPLinear(manifold, in_features, out_features, c_in, dropout=dropout)
        if agg_type==1:
            self.agg = HypMPAgg(manifold, c_in)
        elif agg_type==2:
            self.agg = HypMPAgg2(manifold, c_in)
        self.hyp_act = HypMPAct(manifold, c_out, act)
        self.manifold = manifold

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h

class HMPGCNConv2(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, spatial_order=1, dropout=0.0, use_bias=True):
        super(HMPGCNConv2, self).__init__()
        self.spatial_order=spatial_order
        self.graph_layer = HMPGCNConv(manifold, in_features, out_features, c, c, dropout=dropout, use_bias=use_bias, agg_type=2)
        self.manifold = manifold
        self.c = c

    def forward(self, x, list_edge_index):
        h = None
        for i in range(self.spatial_order):
            output = self.graph_layer(x, list_edge_index[i])
            if i == 0:
                h = output
            else:
                h = self.manifold.mobius_add(h, output, c=self.c)
        return h
    
class HMPGCNConvplus(nn.Module):
    def __init__(self, manifold, in_features, out_features, c=1.0, dropout=0.0, act=F.leaky_relu,
                 use_bias=True):
        super(HMPGCNConvplus, self).__init__()
        self.c = Parameter(torch.Tensor([c]), requires_grad=True)
        # debug:
        #print('HMPGCN trainable_curvature confirmed:',self.c.requires_grad)
        self.linear = HypMPLinear(manifold, in_features, out_features, self.c, dropout=dropout)
        self.agg = HypMPAgg(manifold, self.c, fixed_curvature=False)
        self.hyp_act = HypMPAct(manifold, self.c, act)
        self.manifold = manifold

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h
    
    def get_curvature(self):
        return self.c

class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout=0.6, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAggAtt(MessagePassing):
    """
    Hyperbolic aggregation layer using mlp.
    """

    def __init__(self, manifold, c, out_features, bias=True):
        super(HypAggAtt, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_bias = bias
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_j = torch.nn.functional.embedding(edge_j, x_tangent)
        x_i = torch.nn.functional.embedding(edge_i, x_tangent)

        norm = self.mlp(torch.cat([x_i, x_j], dim=1))
        norm = softmax(norm, edge_i, x_i.size(0)).view(-1, 1)
        support = norm.view(-1, 1) * x_j
        support_t_curv = scatter(support, edge_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t_curv, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold, c, out_features, bias=True):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index=None):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_j = torch.nn.functional.embedding(node_j, x_tangent)
        support = norm.view(-1, 1) * x_j
        support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAttAgg(MessagePassing):
    def __init__(self, manifold, c, out_features, att_dropout=0.6, heads=1, concat=False):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return support_t


# refer to: https://github.com/ferrine/hyrnn/blob/master/hyrnn/nets.py

class HypGRU(nn.Module):
    def __init__(self, args, c):
        super(HypGRU, self).__init__()
        self.manifold = PoincareBall()
        self.c = c
        self.nhid = args.nhid
        self.weight_ih = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True)
        self.weight_hh = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True)
        self.tanh = nn.Tanh()
        if args.bias:
            self.bias = nn.Parameter(torch.ones(3, args.nhid) * 1e-5, requires_grad=False)
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hyperx, hyperh):
        out = self.mobius_gru_cell(hyperx, hyperh, self.weight_ih, self.weight_hh, self.bias)
        return out

    def mobius_gru_cell(self, input, hx, weight_ih, weight_hh, bias, nonlin=None):
        W_ir, W_ih, W_iz = weight_ih.chunk(3)
        b_r, b_h, b_z = bias
        W_hr, W_hh, W_hz = weight_hh.chunk(3)

        z_t = self.manifold.logmap0(self.one_rnn_transform(W_hz, hx, W_iz, input, b_z), self.c).sigmoid()
        r_t = self.manifold.logmap0(self.one_rnn_transform(W_hr, hx, W_ir, input, b_r), self.c).sigmoid()

        rh_t = self.manifold.mobius_pointwise_mul(r_t, hx, c=self.c)


        h_tilde = self.tanh(self.one_rnn_transform(W_hh, rh_t, W_ih, input, b_h)) # tanh

        delta_h = self.manifold.mobius_add(-hx, h_tilde, c=self.c)
        zdelta = self.manifold.mobius_pointwise_mul(z_t, delta_h, c=self.c)
        h_out = self.manifold.mobius_add(hx, zdelta, c=self.c)
        return h_out

    def one_rnn_transform(self, W, h, U, x, b):
        W_otimes_h = self.manifold.mobius_matvec(W, h, self.c)
        U_otimes_x = self.manifold.mobius_matvec(U, x, self.c)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, self.c)
        return self.manifold.proj(self.manifold.mobius_add(Wh_plus_Ux, b, self.c), self.c)

    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output

class HypMPLinear(nn.Module):
    """
    Hyperbolic (no tangent) linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout=0.0, use_bias=True):
        super(HypMPLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        # debug
        #print('HMPLinear trainable_curvature confirmed:',self.c.requires_grad)
        self.p = dropout
        self.dropout = nn.Dropout(self.p)
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(1, out_features), requires_grad=True)
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        geotorch.orthogonal(self.weight, "weight")
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, x):
        if self.p > 0.0:
            res = self.manifold.proj(self.dropout(self.weight(x)), self.c)
        else:
            res = self.manifold.proj(self.weight(x), self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypMPAct(Module):
    """
    Hyperbolic (no tangent) activation layer.
    """

    def __init__(self, manifold, c, act):
        super(HypMPAct, self).__init__()
        self.manifold = manifold
        self.c = c
        # debug
        #print('HMPAct trainable_curvature confirmed:',self.c.requires_grad)
        #self.c_threshold=10.0
        ### Be careful
        self.act = act

    def forward(self, x):
        xt = self.act(x)
        return xt

    def extra_repr(self):
        return 'c={}'.format(
            self.c
        )

class HypMPAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold, c, fixed_curvature=True):
        super(HypMPAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        # debug
        #print('HMPAgg trainable_curvature confirmed:',self.c.requires_grad)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):

        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        s = self.manifold.p2k(x, self.c)
        node_i = edge_index[0]
        node_j = edge_index[1]
        lamb = self.manifold.lorenz_factor(s, keepdim=True)
        lamb = torch.nn.functional.embedding(node_j, lamb)
        norm = norm.view(-1, 1) # len(node_j) x 1
        support_w = norm * lamb # len(node_j) x 1
        s_j = torch.nn.functional.embedding(node_j, s)
        s_j = support_w * s_j
        tmp = scatter(support_w, node_i, dim=0, dim_size=x.size(0))
        s_out = scatter(s_j, node_i, dim=0, dim_size=x.size(0))
        s_out = s_out / tmp
        output = self.manifold.k2p(s_out, self.c)
        return output

class HypMPAgg2(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold, c, fixed_curvature=True):
        super(HypMPAgg2, self).__init__()
        self.manifold = manifold
        self.c = c
        # debug
        #print('HMPAgg trainable_curvature confirmed:',self.c.requires_grad)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index):
        # x: N x D
        edge_index, _ = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = torch.nn.functional.embedding(node_i, x) # len(node_i) x D
        x_j = torch.nn.functional.embedding(node_j, x) # len(node_j) x D
        dist = self.manifold.sqdist(x_i, x_j, c=self.c) 
        dist = dist.unsqueeze(1)
        dist = torch.exp(dist * (-1))
        dist_total = scatter(dist, node_i, dim=0, dim_size=x.size(0))
        dist_total_ = torch.nn.functional.embedding(node_j, dist_total)
        dist_norm = dist / dist_total_

        s = self.manifold.p2k(x, self.c)
        lamb = self.manifold.lorenz_factor(s, keepdim=True)
        lamb = torch.nn.functional.embedding(node_j, lamb)
        # norm = norm.view(-1, 1) # len(node_j) x 1
        support_w = dist_norm * lamb # len(node_j) x 1
        s_j = torch.nn.functional.embedding(node_j, s)
        s_j = support_w * s_j
        tmp = scatter(support_w, node_i, dim=0, dim_size=x.size(0))
        tmp = tmp.clamp(min=1e-10)
        s_out = scatter(s_j, node_i, dim=0, dim_size=x.size(0))
        s_out = s_out / tmp
        output = self.manifold.k2p(s_out, self.c)
        return output

class HypMPGRU(nn.Module):
    def __init__(self, args, c, fixed_curvature=True):
        super(HypMPGRU, self).__init__()
        self.manifold = PoincareBall()
        if fixed_curvature:
            self.c = c
        else:
            self.c = Parameter(torch.Tensor([c]), requires_grad=True)
        # debug
        #print('HMPGRU trainable_curvature confirmed:',self.c.requires_grad)
        self.nhid = args.nhid
        self.W_ir = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_ih = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_iz = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hr = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hh = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hz = nn.Linear(self.nhid, self.nhid, bias=False)
        if args.bias:
            self.bias = nn.Parameter(self.toHyperX(torch.ones(3, self.nhid) * 1e-5), requires_grad=True)
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        geotorch.orthogonal(self.W_ir, "weight")
        geotorch.orthogonal(self.W_ih, "weight")
        geotorch.orthogonal(self.W_iz, "weight")
        geotorch.orthogonal(self.W_hr, "weight")
        geotorch.orthogonal(self.W_hh, "weight")
        geotorch.orthogonal(self.W_hz, "weight")

    def get_curvature(self):
        return self.c

    def set_curvature(self, new_c):
        self.c = new_c
        return

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def forward(self, hyperx, hyperh):
        out = self.mobius_gru_cell(hyperx, hyperh, self.bias)
        return out

    def mobius_gru_cell(self, input, hx, bias, nonlin=None):
        b_r, b_h, b_z = self.bias

        z_t = self.one_rnn_transform(self.W_hz, hx, self.W_iz, input, b_z).sigmoid()
        r_t = self.one_rnn_transform(self.W_hr, hx, self.W_ir, input, b_r).sigmoid()

        rh_t = r_t * hx

        h_tilde = torch.tanh(self.one_rnn_transform(self.W_hh, rh_t, self.W_ih, input, b_h)) # tanh

        hx = hx * z_t
        h_tilde = h_tilde * (1 - z_t)
        h_out = self.manifold.mobius_add(h_tilde, hx, c=self.c)
        return h_out

    def one_rnn_transform(self, W, h, U, x, b):
        W_otimes_h = W(h)
        U_otimes_x = U(x)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, self.c)
        return self.manifold.mobius_add(Wh_plus_Ux, b, self.c)

    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output



class HypStructuralAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, device, c=1.0, act=F.relu, dropout=0.0):
        super(HypStructuralAttentionLayer, self).__init__()

        self.manifold = Hyperboloid()
        self.c = c
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        self.linear = HypLinear(self.manifold, self.in_features, self.out_features, self.c, dropout=dropout)
        self.act = HypAct(self.manifold, self.c, self.c, act)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    def structural_attention(self, h, edge_index):
        edge_index, norm = self.norm(edge_index, h.shape[0])
        h_k = self.manifold.h2k(h)
        node_i = edge_index[0]
        node_j = edge_index[1]
        lamb = self.manifold.lorenz_factor(h_k, keepdim=True)
        lamb = torch.nn.functional.embedding(node_j, lamb) # N'x 1
        tmp1 = torch.nn.functional.embedding(node_i, h) # N' x D
        tmp2 = torch.nn.functional.embedding(node_j, h) # N' x D
        distance_ij = torch.exp(self.manifold.sqdist(tmp1, tmp2, self.c) * (-1)) # N' x 1
        alpha = scatter(distance_ij, node_i, dim=0, dim_size=h.size(0)) # N x 1
        alpha_j = torch.nn.functional.embedding(node_j, alpha) # N' x 1
        softmax_alpha_j = distance_ij / alpha_j
        h_k_j = torch.nn.functional.embedding(node_j, h_k) # N' x D
        h_k_j = h_k_j * lamb * softmax_alpha_j
        w_j = lamb * softmax_alpha_j
        m_k = scatter(h_k_j, node_i, dim=0, dim_size=h.size(0))
        w = scatter(w_j, node_i, dim=0, dim_size=h.size(0))
        m_k = m_k / w
        output = self.manifold.k2h(m_k, self.device)
        return output, edge_index


    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h, edge_index = self.structural_attention(h, edge_index)
        h = self.act.forward(h)
        return h, edge_index


class HypTemporalAttentionLayer(nn.Module):
    def __init__(self, features, device, num_time_steps, c=1.0):
        super(HypTemporalAttentionLayer, self).__init__()

        self.manifold = Hyperboloid()
        self.c = c
        self.device = device
        self.w_q = Parameter(torch.Tensor(num_time_steps, num_time_steps), requires_grad=True)
        self.w_k = Parameter(torch.Tensor(num_time_steps, num_time_steps), requires_grad=True)
        self.b_q = Parameter(torch.ones(1, features), requires_grad=True)
        self.b_k = Parameter(torch.ones(1, features), requires_grad=True)
        self.a = Parameter(torch.ones(1), requires_grad=True)
        self.c = Parameter(torch.ones(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.w_q)
        glorot(self.w_k)

    def forward(self, list_h_by_time, time_index, mask):
        # each element in list_h_by_time will have the shape of N x 1 x D
        big_h = torch.cat(list_h_by_time, dim=1) # N x T x D
        n, t, d = big_h.shape
        tmp = torch.reshape(big_h, (n * t, d))
        q = self.one_transform(self.w_q, big_h, self.b_q) # (N x T) x D
        k = self.one_transform(self.w_k, big_h, self.b_k) # (N x T) x D
        tmp_k = self.manifold.h2k(tmp) # (N x T) x D - 1
        q = torch.reshape(q, (n, t, d)) # N x T x D
        k = torch.reshape(k, (n, t, d)) # N x T x D
        v = torch.reshape(tmp_k, (n, t, d - 1)) # N x T x D - 1

        t_m = time_index[0]
        t_n = time_index[1]
        q_t_m = q[:, t_m, :] # N x (TxT) x D
        k_t_n = k[:, t_n, :] # N x (TxT) x D
        distance_tmn = self.manifold.sqdist(q_t_m, k_t_n, self.c) * (-1) # N x (TxT) x 1
        s_tmn = self.a * distance_tmn - self.c + mask # N x (TxT) x 1
        s_tmn = torch.exp(s_tmn) # N x (TxT) x 1
        s_tmn_sum = scatter(s_tmn, t_m, dim=1) # N x T x 1
        s_tmn_sum = s_tmn_sum[:, t_n, :] # N x (TxT) x 1
        softmax_beta_tmn = s_tmn / s_tmn_sum # N x (TxT) x 1

        v = torch.reshape(v, (n * t, d - 1)) # (N x T) x D
        lamb = self.manifold.lorenz_factor(v, keepdim=True) # (N x T) x 1
        lamb = torch.reshape(lamb, (n, t, 1))
        lamb = lamb[:, t_n, :] # N x (TxT) x 1
        v = torch.reshape(v, (n, t, d - 1)) # N x T x D
        v = v[:, t_n, :] # N x (TxT) x D

        z_tmn = v * lamb * softmax_beta_tmn # N x (TxT) x D
        w_tmn = lamb * softmax_beta_tmn # N x (TxT) x 1
        z_tmn_sum = scatter(z_tmn, t_m, dim=1) # N x T x D
        w_tmn_sum = scatter(w_tmn, t_m, dim=1) # N x T x 1
        output = z_tmn_sum / w_tmn_sum # N x T x D
        output = torch.reshape(output, (n * t, d - 1)) # (N x T) x D - 1
        output = self.manifold.k2h(output, self.device)
        output = torch.reshape(output, (n , t, d)) # N x T x D
        return output

    
    def one_transform(self, W, h, b):
        n, t, d = h.shape
        h = torch.reshape(h, (t, n * d))
        W_otimes_h = self.manifold.mobius_matvec(W, h, self.c, rev=False)
        W_otimes_h = torch.reshape(W_otimes_h, (n * t, d))
        return self.manifold.proj(self.manifold.mobius_add(W_otimes_h, b, self.c), self.c)