from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import pandas as pd
import pickle

from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import add_self_loops
import random
from random import shuffle
from train_multidataset import pos2key


class DomainsDataset(Dataset):
    """ Customized dataset for each domain"""

    def __init__(self, X, Y):
        self.X = X                           # set data
        self.Y = Y                           # set key

    def __len__(self):
        return len(self.X)                   # return length

    def __getitem__(self, idx):
        # return list of batch data [data, key]
        return [self.X[idx], self.Y[idx]]


def printinfo(dataset, train_domains, valid_domains, test_domains):
    print('======================')
    print(dataset)
    print(f'Number of training domains: {len(train_domains)}')
    print(f'Number of valid domains: {len(valid_domains)}')
    print(f'Number of test domains: {len(test_domains)}')


def load_data(dataset, batchSize, seed=None):
    root = 'data'
    if seed is not None:
        np.random.seed(seed)
    if dataset in ['pm25', 'temp']:
        point_path = os.path.join(root, dataset, dataset)+'.pkl'
        with open(point_path, 'rb') as f:
            dict = pickle.load(f)
        dict_l = list(dict.items())

        np.random.shuffle(dict_l)  # return None, inplace
        train_valid_split = int(int(8) / 10 * len(dict_l))
        valid_test_split = int(1 / 10 * len(dict_l))

        train_domains = dict_l[:train_valid_split]
        valid_domains = dict_l[train_valid_split:train_valid_split+valid_test_split]
        test_domains = dict_l[train_valid_split+valid_test_split:]

        printinfo(dataset, train_domains, valid_domains, test_domains)

        pos_key_train = ([i[0] for i in train_domains])
        num_samples = [i[1].shape[0] for i in train_domains]
        pos_key_train = np.repeat(pos_key_train, num_samples, axis=0)

        train_samples = [i[1] for i in train_domains]
        train_samples = np.vstack(train_samples)

        domains_dataset = DomainsDataset(train_samples, pos_key_train)
        dl = DataLoader(domains_dataset, batch_size=batchSize,
                        shuffle=True)  # train

        # S_0= np.apply_along_axis(lambda i:[i[0].split('_')[0],i[0].split('_')[1]],axis=1, arr=train_domains)
        S_0_key = np.apply_along_axis(
            lambda i: i[0], axis=1, arr=train_domains)
        return dl, S_0_key, train_domains, valid_domains, test_domains

    elif dataset == 'flu':
        point_path = os.path.join(root, 'influenza_outbreak', 'flu_data.pkl')
    elif dataset in ['argentina', 'brazil', 'chile', 'colombia', 'ecuador', 'el salvador', 'mexico', 'paraguay', 'uruguay', 'venezuela']:
        point_path = os.path.join(root, 'civil_unrest', dataset)+'.pkl'
    else:
        raise Exception('Dataset not recognized.')

    with open(point_path, 'rb') as f:
        data = pickle.load(f)

    # check the number of events in each domain
    # datay=data[:,:,0]
    # num_of_1=datay.sum(axis=1)
    # print("min events number in domains {0:2}".format(num_of_1.min()))

    # shuffle(data) #shuffle all axis including column!!wrong
    np.random.shuffle(data)
    # assert(len(set([pos2key(i[0,-2:]) for i in data]))==data.shape[0])
    # argentina has two same location -31.42008329999999
    train_valid_split = int(int(8) / 10 * len(data))
    # from train_reg_graph import pos2key [pos2key(i[0,-2:]) for i in data]
    valid_test_split = int(1 / 10 * len(data))

    train_domains = data[:train_valid_split]
    valid_domains = data[train_valid_split:train_valid_split+valid_test_split]
    test_domains = data[train_valid_split+valid_test_split:]

    printinfo(dataset, train_domains, valid_domains, test_domains)

    return train_domains, valid_domains, test_domains
