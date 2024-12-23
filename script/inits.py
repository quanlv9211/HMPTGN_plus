import math
import numpy as np
import torch
import time
import networkx as nx
from scipy.sparse import coo_matrix
from script.utils.util import logger
from tqdm import tqdm
from torch_geometric.utils import remove_self_loops
from script.config import args


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def xavier_init(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return torch.Tensor(initial)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def prepare(data, t, detection=False):
    if detection == False:
        # obtain adj index
        edge_index = data['edge_index_list'][t].long().to(args.device)  # torch edge index
        pos_index = data['pedges'][t].long().to(args.device)  # torch edge index
        neg_index = data['nedges'][t].long().to(args.device)  # torch edge index
        new_pos_index = data['new_pedges'][t].long().to(args.device)  # torch edge index
        new_neg_index = data['new_nedges'][t].long().to(args.device)  # torch edge index
        # 2.Obtain current updated nodes
        # nodes = list(np.intersect1d(pos_index.numpy(), neg_index.numpy()))
        # 2.Obtain full related nodes
        nodes = list(np.union1d(pos_index.cpu().numpy(), neg_index.cpu().numpy()))
        weights = None
        return edge_index, pos_index, neg_index, nodes, weights, new_pos_index, new_neg_index

    if detection == True:
        train_pos_edge_index = data['gdata'][t].train_pos_edge_index.long().to(args.device)

        val_pos_edge_index = data['gdata'][t].val_pos_edge_index.long().to(args.device)
        val_neg_edge_index = data['gdata'][t].val_neg_edge_index.long().to(args.device)

        test_pos_edge_index = data['gdata'][t].test_pos_edge_index.long().to(args.device)
        test_neg_edge_index = data['gdata'][t].test_neg_edge_index.long().to(args.device)
        return train_pos_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index

def prepare_dilated_edge_index(data, spatial_dilated_factors, device):
    dilated_edge_index_list = []
    logger.info('computing spatial dilated edge list ...')
    for edge_index in tqdm(data['edge_index_list']):
        adj = coo_matrix(([1] * len(edge_index[0]), (list(edge_index[0]), list(edge_index[1]))), shape=(data['num_nodes'], data['num_nodes']), dtype=int)
        adj = adj.tocsr() + adj.transpose().tocsr()
        dilated_edge_index = []
        for factor in spatial_dilated_factors:
            exponent = 1
            adj_exp = adj
            while exponent < factor:
                adj_exp = adj_exp.dot(adj)
                exponent += 1
            adj_exp = adj_exp.tocoo()
            adj_exp.eliminate_zeros()
            coords = np.vstack((adj_exp.row, adj_exp.col)).transpose()
            np.random.shuffle(coords)
            coords, _ = remove_self_loops(torch.tensor(coords.transpose(), dtype=torch.long))
            dilated_edge_index.append(coords.to(device))
        dilated_edge_index_list.append(dilated_edge_index)
    return dilated_edge_index_list

def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()
    hyps = []
    for i in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp: ', time.time() - curr_time)
    return max(hyps)