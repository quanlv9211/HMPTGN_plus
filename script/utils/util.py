from ftplib import all_errors
import random
import os
#from tkinter import ALL
import torch
import numpy as np
import logging

logger = logging.getLogger()
disease_path = '../data/input/cached/disease/disease_lp.feats.npz'


def mkdirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def write_to_text(args, results, seed):
    if args.to_compute_results:
        with open(args.result_txt, "a") as f:
            data = str(seed) + '\t' + '\t'.join(['{:.6f}'.format(result) for result in results[1:]]) + '\n'
            f.write(data)
    else:
        print('Not write to file')


def compute_mean_std(args):
    if args.to_compute_results:
        results = np.loadtxt(args.result_txt, delimiter='\t')
        results = results[:, 1:]
        mean_list = np.mean(results, axis=0)
        std_list = np.std(results, axis=0)
        with open(args.result_txt, "a") as f:
            data = 'results: ' + '\t'.join(
                ['{:.2f}+{:.2f}'.format(mean * 100, std * 100) for mean, std in zip(mean_list, std_list)]) + '\n'
            f.write(data)
    else:
        print('Not write to file')


def set_random(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('>> fixed random seed')


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def adjust_learning_rate(lr, optimizer, epoch):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def filters(edge_index, threshold):
    return np.where(np.where(edge_index >= threshold, 1, 0).sum(axis=0) == 0)


def neg_sampling_basic(edge_index, num_nodes):
    import random
    num_neg_samples = edge_index.shape[1]
    idx = edge_index[0] * num_nodes + edge_index[1]
    rng = range(num_nodes ** 2)
    perm = torch.tensor(random.sample(rng, num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
        perm[rest] = tmp
        rest = rest[mask.nonzero().view(-1)]
    row, col = torch.true_divide(perm, num_nodes), perm % num_nodes
    return torch.stack([row, col], dim=0).long().numpy()


def negative_sampling(adjs, cum_nodes):
    '''
    negative sampling each time stamp without sampling previous time
    '''

    history_perm = []
    neg_adjs = []
    for t in range(len(adjs)):
        edge_index = adjs[t]
        num_nodes = len(cum_nodes[t])
        num_neg_samples = edge_index.shape[1]  # get number of negative sampling nodes
        current_perm = edge_index[0] * num_nodes + edge_index[1]  # obtain the total hash slots
        idx = np.concatenate([current_perm, history_perm])  # expand the slots with historical slots
        rng = range(num_nodes ** 2)  # get range of negative sampling
        perm = torch.tensor(random.sample(rng, num_neg_samples))  # get negative perm
        mask = torch.from_numpy(np.isin(perm, idx)).to(torch.bool)
        rest = mask.nonzero().view(-1)
        while rest.numel() > 0:  # pragma: no cover
            tmp = torch.tensor(random.sample(rng, rest.size(0)))
            mask = torch.from_numpy(np.isin(tmp, idx)).to(torch.bool)
            perm[rest] = tmp
            rest = rest[mask.nonzero().view(-1)]
        # row, col = perm / num_nodes, perm % num_nodes
        row, col = torch.true_divide(perm, num_nodes), perm % num_nodes
        neg_adjs.append(torch.stack([row, col], dim=0).long())
        history_perm += list(perm.numpy())
    return neg_adjs


def total_pairwise_similarity(x):
    # x = F.normalize(x)
    z = torch.matmul(x, x.transpose(0, 1))
    n = z.size(1) * (z.size(1) - 1)  # A_2^n
    total_similarity = torch.sum(z) - torch.sum(z.diagonal())
    average = total_similarity / n
    return 1 - average

def historical_negative_sampling_data(data, test_shots, train_shots):
    '''
    This function will return positive_edge_index_list and negative_edge_index_list
    according to historical NS strategy proposed by this paper:
        Link: arxiv.org/abs/2207.10128
    '''
    all_edge_indexs = data['edge_index_list']
    num_nodes = data['num_nodes']
    l = []
    for i in train_shots:
        l.append(all_edge_indexs[i].long())
    e_train = torch.cat(l, dim=1).T   # bidirection
    pos_edge_index, neg_edge_index = {}, {}

    for i in test_shots:
        pos_index = data['pedges'][i] # [2, ...] , direction
        pos_edge_index[i] = pos_index.long()

        num_false = pos_index.shape[1]
        edges_false = []
        # historical NS
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, num_nodes)
            idx_j = np.random.randint(0, num_nodes)
            if idx_i == idx_j:
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            pos_edge = pos_index.T
            if ismember([idx_i, idx_j], np.array(e_train)): 
                if ismember([idx_i, idx_j], np.array(pos_edge)):
                    continue
                if ismember([idx_j, idx_i], np.array(pos_edge)):
                    continue
                edges_false.append([idx_i, idx_j])
            
        # the remaining negative edges are sampled by the random NS strategy
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, num_nodes)
            idx_j = np.random.randint(0, num_nodes)
            if idx_i == idx_j:
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            pos_edge = pos_index.T
            if ismember([idx_i, idx_j], np.array(pos_edge)):
                continue
            if ismember([idx_j, idx_i], np.array(pos_edge)):
                continue
            edges_false.append([idx_i, idx_j])

        neg_index = torch.Tensor(edges_false).T.long()
        neg_edge_index[i] = neg_index

    return pos_edge_index, neg_edge_index

def inductive_negative_sampling_data(data, test_shots, train_shots):
    '''
    This function will return positive_edge_index_list and negative_edge_index_list
    according to inductive NS strategy proposed by this paper:
        Link: arxiv.org/abs/2207.10128
    '''
    all_edge_indexs = data['edge_index_list']
    num_nodes = data['num_nodes']
    l, m = [], []
    for i in train_shots:
        l.append(all_edge_indexs[i].long())
    e_train = torch.cat(l, dim=1).T   # bidirection
    for i in test_shots:
        m.append(all_edge_indexs[i].long())
    e_test = torch.cat(m, dim=1).T   # bidirection
    pos_edge_index, neg_edge_index = {}, {}

    for i in test_shots:
        pos_index = data['pedges'][i] # [2, ...] , direction
        pos_edge_index[i] = pos_index.long()

        num_false = pos_index.shape[1]
        edges_false = []
        # inductive NS
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, num_nodes)
            idx_j = np.random.randint(0, num_nodes)
            if idx_i == idx_j:
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            pos_edge = pos_index.T
            if ismember([idx_i, idx_j], np.array(e_train)): 
                continue
            if ismember([idx_i, idx_j], np.array(pos_edge)):
                continue
            if ismember([idx_j, idx_i], np.array(pos_edge)):
                continue
            if ismember([idx_i, idx_j], np.array(e_test)):
                edges_false.append([idx_i, idx_j])
            
        # the remaining negative edges are sampled by the random NS strategy
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, num_nodes)
            idx_j = np.random.randint(0, num_nodes)
            if idx_i == idx_j:
                continue
            if edges_false:
                if ismember([idx_j, idx_i], np.array(edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(edges_false)):
                    continue
            pos_edge = pos_index.T
            if ismember([idx_i, idx_j], np.array(pos_edge)):
                continue
            if ismember([idx_j, idx_i], np.array(pos_edge)):
                continue
            edges_false.append([idx_i, idx_j])

        neg_index = torch.Tensor(edges_false).T.long()
        neg_edge_index[i] = neg_index

    return pos_edge_index, neg_edge_index

def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
