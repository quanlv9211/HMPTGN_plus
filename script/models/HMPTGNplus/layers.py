import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from script.hgcn.manifolds import PoincareBall, Euclidean
from torch_scatter import scatter, scatter_add
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch.nn.modules.module import Module
from torch_geometric.nn.conv import MessagePassing
import geotorch
import itertools
import math


class HypMPLinear(nn.Module):
    """
    Hyperbolic (no tangent) linear layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias=True):
        super(HypMPLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(1, out_features), requires_grad=True)
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        geotorch.orthogonal(self.weight, "weight")
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, x, c_):
        res = self.manifold.proj(self.weight(x), c_)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias, c_)
            hyp_bias = self.manifold.expmap0(bias, c_)
            res = self.manifold.mobius_add(res, hyp_bias, c=c_)
            res = self.manifold.proj(res, c_)
        return res


class HypMPAct(Module):
    """
    Hyperbolic (no tangent) activation layer.
    """

    def __init__(self, manifold, act):
        super(HypMPAct, self).__init__()
        self.manifold = manifold
        self.act = act

    def forward(self, x, c_):
        xt = self.act(x)
        return xt


class HypMPAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """

    def __init__(self, manifold):
        super(HypMPAgg, self).__init__()
        self.manifold = manifold

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

    def forward(self, x, edge_index, c_):

        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        s = self.manifold.p2k(x, c_)
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
        output = self.manifold.k2p(s_out, c_)
        return output


class HypMPAgg2(MessagePassing):
    """
    Hyperbolic aggregation layer using distance.
    """

    def __init__(self, manifold):
        super(HypMPAgg2, self).__init__()
        self.manifold = manifold

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

    def forward(self, x, edge_index, c_):
        # x: N x D
        edge_index, _ = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = torch.nn.functional.embedding(node_i, x) # len(node_i) x D
        x_j = torch.nn.functional.embedding(node_j, x) # len(node_j) x D
        dist = self.manifold.sqdist(x_i, x_j, c=c_) 
        dist = dist.unsqueeze(1)
        dist = torch.exp(dist * (-1))
        dist_total = scatter(dist, node_i, dim=0, dim_size=x.size(0))
        dist_total_ = torch.nn.functional.embedding(node_j, dist_total)
        dist_norm = dist / dist_total_

        s = self.manifold.p2k(x, c_)
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
        output = self.manifold.k2p(s_out, c_)
        return output


class HMPGNNplus(nn.Module):
    def __init__(self, manifold, in_features, out_features, spatial_order=1, dropout=0.0, use_bias=True, agg_type=1,  act=F.leaky_relu):
        super(HMPGNNplus, self).__init__()
        self.spatial_order=spatial_order
        self.linear = HypMPLinear(manifold, in_features, out_features, use_bias)
        if agg_type==1:
            self.agg = HypMPAgg(manifold)
        elif agg_type==2:
            self.agg = HypMPAgg2(manifold) 
        self.hyp_act = HypMPAct(manifold, act)
        self.manifold = manifold

    def forward(self, x, list_edge_index, c_):
        x_ = self.linear.forward(x, c_)
        h = None
        for i in range(self.spatial_order):
            output = self.agg.forward(x_, list_edge_index[i], c_)
            if i == 0:
                h = output
            else:
                h = self.manifold.mobius_add(h, output, c=c_)
        h = self.hyp_act.forward(h, c_)
        return h

class HMPGNN(nn.Module):
    def __init__(self, manifold, in_features, out_features, spatial_order=1, dropout=0.0, use_bias=True, agg_type=1,  act=F.leaky_relu):
        super(HMPGNN, self).__init__()
        self.spatial_order=spatial_order
        self.linear = HypMPLinear(manifold, in_features, out_features, use_bias)
        self.agg = HypMPAgg(manifold)
        self.hyp_act = HypMPAct(manifold, act)
        self.manifold = manifold

    def forward(self, x, edge_index, c_):
        x_ = self.linear.forward(x, c_)
        h = self.agg.forward(x_, edge_index, c_)
        h = self.hyp_act.forward(h, c_)
        return h


class HypMPGRU(nn.Module):
    def __init__(self, args):
        super(HypMPGRU, self).__init__()
        self.manifold = PoincareBall()

        self.nhid = args.nhid
        self.W_ir = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_ih = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_iz = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hr = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hh = nn.Linear(self.nhid, self.nhid, bias=False)
        self.W_hz = nn.Linear(self.nhid, self.nhid, bias=False)
        if args.bias:
            self.bias = nn.Parameter(torch.ones(3, self.nhid) * 1e-5, requires_grad=True)
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

    def toHyperX(self, x, c=1.0):
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def forward(self, hyperx, hyperh, c_):
        out = self.mobius_gru_cell(hyperx, hyperh, self.bias, c_)
        return out

    def mobius_gru_cell(self, input, hx, bias, c_, nonlin=None):
        b_r, b_h, b_z = bias

        b_r = self.toHyperX(b_r, c_)
        b_h = self.toHyperX(b_h, c_)
        b_z = self.toHyperX(b_z, c_)

        z_t = torch.sigmoid(self.one_rnn_transform(self.W_hz, hx, self.W_iz, input, b_z, c_))
        r_t = torch.sigmoid(self.one_rnn_transform(self.W_hr, hx, self.W_ir, input, b_r, c_))

        rh_t = r_t * hx

        h_tilde = torch.tanh(self.one_rnn_transform(self.W_hh, rh_t, self.W_ih, input, b_h, c_)) # tanh

        hx = hx * z_t
        h_tilde = h_tilde * (1 - z_t)
        h_out = self.manifold.mobius_add(h_tilde, hx, c=c_)
        return h_out

    def one_rnn_transform(self, W, h, U, x, b, c_):
        W_otimes_h = W(h)
        U_otimes_x = U(x)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x, c_)
        return self.manifold.mobius_add(Wh_plus_Ux, b, c_)


class HMPDCA_Dist(nn.Module):
    # This is the Distance-version
    def __init__(self, in_dim, out_dim, manifold, device, kernel_size, nonlinear, dilation_step=1, stride=1, use_bias=True):
        super(HMPDCA_Dist, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.manifold = manifold
        self.device = device
        self.kernel_size = kernel_size
        self.dilation_step = dilation_step
        self.stride = stride
        self.pad = (kernel_size - 1) * dilation_step
        self.use_bias = use_bias
        if nonlinear == 'sigmoid':
          self.act = nn.Sigmoid()
        elif nonlinear == 'softmax':
          self.act = nn.Softmax(dim=-1)


    def forward(self, hidden_window, c_):

        # hidden_window: (Delta T) x N x D
        original_window_size = hidden_window.shape[0]
        padding = torch.zeros((self.pad, hidden_window.shape[1], hidden_window.shape[2])).to(self.device) # (pad) x N x D
        pad_hidden_window = torch.cat([padding, hidden_window], dim=0) # (Delta T + pad) x N x D
 
        # creating (kernel) indexes for each time stamp
        list_index = []
        for t in range(self.pad, self.pad + original_window_size):
            indexes = np.arange(t, t + self.kernel_size * (-self.dilation_step), -self.dilation_step)
            list_index.append(indexes)
        list_index = torch.tensor(np.array(list_index))

        # creating kernel window tensor for each time stamp
        h = pad_hidden_window[list_index]   # (Delta T) x (Kernel Size) x N x D
        h = torch.reshape(h, [h.shape[0], h.shape[2], h.shape[1], h.shape[3]]) # (Delta T) x N x (Kernel Size) x D
       

        # compute distance score
        h_last = hidden_window.unsqueeze(2) # (Delta T) x N x 1 x D
        kernel = self.manifold.sqdist(h, h_last, c=c_) # (Delta T) x N x (Kernel Size)
        kernel = kernel.unsqueeze(-1) # (Delta T) x N x (Kernel Size) x 1

        # aggregation
        z = self.manifold.p2k(h, c_) # (Delta T) x N x (Kernel Size) x D
        lamb = self.manifold.lorenz_factor(z, c_, keepdim=True)  # (Delta T) x N x (Kernel Size) x 1
        kernel_ = self.act(kernel) # (Delta T) x N x (Kernel Size) x D
        weight = lamb * kernel_ # (Delta T) x N x (Kernel Size) x 1
        weight_sum = weight.sum(2, keepdim=True).squeeze(-1) # (Delta T) x N x 1
        z = z * weight # (Delta T) x N x (Kernel Size) x D
        z_t = z.sum(2, keepdim=True).squeeze(2) # (Delta T) x N x D
        weight_sum = weight_sum.clamp(min=1e-15)
        z_t = z_t / weight_sum # (Delta T) x N x Dh
        output = self.manifold.k2p(z_t, c_) # (Delta T) x N x D

        return output


class HMPResidualLayer(nn.Module):
    def __init__(self, res_dim, skip_dim, manifold, device, kernel_size, nonlinear, dilation_step=1, use_bias=True, agg_type="distance"):
        super(HMPResidualLayer, self).__init__()
        self.res_dim = res_dim
        self.skip_dim = skip_dim
        self.manifold = manifold
        self.device = device
        self.kernel_size = kernel_size
        self.dilation_step = dilation_step
        self.device = device
        self.use_bias = use_bias

        self.agg_filter = HMPDCA_Dist(self.res_dim, self.res_dim, self.manifold, self.device, self.kernel_size,
                                                nonlinear, self.dilation_step, nonlinear)
        self.agg_gate = HMPDCA_Dist(self.res_dim, self.res_dim, self.manifold, self.device, self.kernel_size,
                                                nonlinear, self.dilation_step, nonlinear)

        self.w_res = nn.Linear(self.res_dim, self.res_dim, bias=False)
        self.w_skip = nn.Linear(self.res_dim, self.skip_dim, bias=False)
        self.bias_res = nn.Parameter(torch.Tensor(1, self.res_dim), requires_grad=True)
        self.bias_skip = nn.Parameter(torch.Tensor(1, self.skip_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        geotorch.orthogonal(self.w_res, "weight")
        geotorch.orthogonal(self.w_skip, "weight")
        glorot(self.bias_res)
        glorot(self.bias_skip)

    def forward(self, hidden_window, c_):
        # hidden_window: (Delta T) x N x D
        h_filter = torch.tanh(self.agg_filter(hidden_window, c_))
        h_gate = torch.sigmoid(self.agg_gate(hidden_window, c_))
        fx = h_filter * h_gate
        fx = self.w_res(fx)
        if self.use_bias:
            bias_res = self.manifold.proj_tan0(self.bias_res, c_)
            hyp_bias_res = self.manifold.expmap0(bias_res, c_)
            fx = self.manifold.mobius_add(fx, hyp_bias_res, c=c_)
            fx = self.manifold.proj(fx, c_)
        skip = self.w_skip(fx) # (Delta T) x N x D_skip
        if self.use_bias:
            bias_skip = self.manifold.proj_tan0(self.bias_skip, c_)
            hyp_bias_skip = self.manifold.expmap0(bias_skip, c_)
            skip = self.manifold.mobius_add(skip, hyp_bias_skip, c=c_)
            skip = self.manifold.proj(skip, c_)

        residual = self.manifold.mobius_add(fx, hidden_window, c_) # (Delta T) x N x D
        return skip, residual


class HMPTemporal(nn.Module):
    def __init__(self, manifold, out_dim, device, dilation_depth, kernel_size, agg_type, use_bias=True, nonlinear='softmax'):
        super(HMPTemporal, self).__init__()
        self.manifold = manifold
        self.device = device
        self.nout = out_dim
        self.residual_size = out_dim
        self.skip_size = out_dim
        self.casual_agg_depth = dilation_depth
        self.casual_agg_kernel_size = kernel_size
        self.use_bias = use_bias
        self.dilated_stack = nn.ModuleList(
            [HMPResidualLayer(self.residual_size, self.skip_size, self.manifold, self.device, self.casual_agg_kernel_size,
                            nonlinear, self.casual_agg_kernel_size ** layer, self.use_bias, agg_type)
             for layer in range(self.casual_agg_depth)])



    def forward(self, hidden_window, c_):
        # hidden_window: (Delta T) x N x D
        skips = []
        for layer in self.dilated_stack:
            skip, hidden_window = layer(hidden_window, c_)
            skips.append(skip.unsqueeze(0))
        out = torch.cat(skips, dim=0) # Depth x (Delta T) x N x D
        out = out.mean(dim=0) # (Delta T) x N x D
        out = self.manifold.proj(out, c_)

        return out
