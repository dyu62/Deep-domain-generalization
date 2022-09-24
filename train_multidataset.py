from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve, auc
import dataset
from utils.utils import triplets
from random import shuffle
import pickle
import argparse
import os
import random
import torch
import pandas as pd
import numpy as np
import time
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import json
from model import Einterp, downstreamMLP, parameter_decoder
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
torch.autograd.set_detect_anomaly(True)


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def pos2key(pos):
    key = "{:08.4f}".format(pos[0])+'_'+"{:08.4f}".format(pos[1])
    return key


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dataset', type=str, default="False")
    parser.add_argument('--edge_rep', type=str, default="True")
    parser.add_argument('--model', type=str, default="SIGNN")

    parser.add_argument('--dataset', type=str, default='pm25')  # 'temp'
    parser.add_argument('--manualSeed', type=str, default="False")
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--test_per_round', type=int,
                        default=10)  # test after x epochs
    parser.add_argument('--patience', type=int,
                        default=50)  # test after x epochs
    parser.add_argument('--nepoch', type=int, default=11)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--activation', type=str, default='relu')  # 'lrelu'
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--norm_loss_coef', type=float, default=0.1)
    parser.add_argument('--num_neighbors', type=int, default=20)
    # embedding interpolation
    parser.add_argument('--E_size', type=int, default=32)
    parser.add_argument('--h_ch_Einter', type=int, default=32)
    parser.add_argument('--localdepth', type=int, default=3)
    parser.add_argument('--num_interactions', type=int, default=3)
    parser.add_argument('--combinedepth', type=int, default=3)
    # decoder
    parser.add_argument('--h_ch_dec', type=int, default=265)
    parser.add_argument('--hlayer_num_dec', type=int, default=2)

    # downstream hyperparms
    parser.add_argument('--h_ch', type=int, default=64)
    parser.add_argument('--out_ch', type=int, default=1)
    parser.add_argument('--activation_CNN', type=str, default='relu')
    args = parser.parse_args()
    args.split_dataset = True if args.split_dataset == "True" else False
    args.edge_rep = True if args.edge_rep == "True" else False
    args.manualSeed = True if args.manualSeed == "True" else False
    args.out_ch = 1
    return args


def main(args, dl, S_0_key, valid_domains, test_domains, flag):
    if flag:
        return
    criterion_l1 = torch.nn.L1Loss()  # reduction='sum'
    criterion_mse = torch.nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def blue(x): return '\033[94m' + x + '\033[0m'
    def red(x): return '\033[31m' + x + '\033[0m'

    # init E_dict
    ##
    def init_E(shape=args.E_size):
        noise = torch.randn(shape)
        return noise
    E_dict = {}
    E_to_optim = set()
    for i in S_0_key:
        E_dict[i] = init_E().to(device).requires_grad_()
        E_to_optim.add(E_dict[i])
    S_0 = np.float32([[i.split('_')[0], i.split('_')[1]] for i in S_0_key])
    S_0 = torch.tensor(S_0, dtype=torch.float32, device=device)

    if args.dataset in ['pm25', "temp"]:
        x_in = 9
    elif args.dataset == 'flu':
        x_in = 545
    elif args.dataset in ['argentina', 'brazil', 'chile', 'colombia', 'ecuador', 'el salvador', 'mexico', 'paraguay', 'uruguay', 'venezuela']:
        x_in = 923
    else:
        raise Exception('Dataset not recognized.')

    if args.model == "SIGNN":
        model = downstreamMLP()
        Einter_model = Einterp(h_channel=args.h_ch_Einter, Esize=args.E_size,
                               localdepth=args.localdepth, num_interactions=args.num_interactions, combinedepth=args.combinedepth)
        downstream_paranum = (x_in*args.h_ch)+(args.h_ch*args.h_ch) + \
            (args.h_ch*args.out_ch)+(args.h_ch*2)+args.out_ch
        decoder = parameter_decoder(in_ch=args.E_size, h_ch=args.h_ch_dec, hlayer_num=args.hlayer_num_dec,
                                    out_ch=downstream_paranum, activation=args.activation_CNN, dropout=True)
        Einter_model.to(device)
        decoder.to(device)
        model = model.to(device)
        optimizer = torch.optim.Adam(
            list(decoder.parameters())+list(Einter_model.parameters()), lr=args.lr)
    else:
        raise Exception('only support SIGNN')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)
    optimizer2 = torch.optim.Adam(
        list(E_to_optim), lr=optimizer.param_groups[0]['lr'])

    def train(domains, Einter_model, decoder, model, E_dict, S_0_key, S_0):
        epochloss = 0
        y_hat, y_true, y_hat_logit = [], [], []
        optimizer.zero_grad()
        if args.model == "SIGNN":
            Einter_model.train()
            decoder.train()
            edge_index = knn_graph(S_0, k=args.num_neighbors)
            num_nodes = S_0.shape[0]
            edge_index_2rd, num_2nd_neighbors, edx_1st, edx_2nd = triplets(
                edge_index, num_nodes)
            # is_source = torch.zeros(S_0.shape[0] ,dtype=torch.bool,device=device)

            for i, data in enumerate(domains):  # Iterate over the domains
                if args.dataset in ['pm25', "temp"]:
                    domain_key = data[0]
                    index = np.argwhere(domain_key == S_0_key).item()
                else:
                    domain_key = pos2key(data[0, -2:])
                    index = np.argwhere(domain_key == np.array(S_0_key)).item()
                x_E = []
                for i in S_0_key:
                    x_E.append(E_dict[i])
                x_E[index] = torch.zeros(args.E_size).to(device)
                x_E = torch.stack(x_E)
                is_source = torch.ones(
                    S_0.shape[0], dtype=torch.bool, device=device)
                is_source[index] = False
                E = Einter_model(S_0, edge_index, edge_index_2rd,
                                 edx_1st, edx_2nd, x_E, is_source, args.edge_rep)
                loss2 = args.norm_loss_coef*torch.norm(E, 2)
                # E_dict[domain_key]=E[index].detach()
                if args.dataset in ['pm25', "temp"]:
                    X, Y = torch.tensor(
                        data[1][:, 1:-2], dtype=torch.float32, device=device), torch.tensor(data[1][:, 0])
                else:
                    X, Y = torch.tensor(
                        data[:, 1:-2], dtype=torch.float32, device=device), torch.tensor(data[:, 0])

                num_sample = 180
                split_num = int(len(Y)/num_sample)
                if args.split_dataset == False or split_num == 0:
                    Para = decoder(E[index][None, :])
                    model.updatepara(x_in, args.h_ch, args.out_ch, Para)

                    if args.dataset not in ['pm25', 'temp']:
                        # binary classification
                        pred = torch.sigmoid(model(X)).cpu()
                        loss1 = F.binary_cross_entropy(
                            pred.reshape(-1, 1), Y.reshape(-1, 1))
                        y_hat_logit += list(pred.detach().numpy().reshape(-1))
                        pred = torch.as_tensor(
                            (pred.detach() - 0.5) > 0).float()
                    else:  # ['pm25','temp']
                        pred = model(X).cpu()
                        loss1 = criterion_l1(
                            pred.reshape(-1, 1), Y.reshape(-1, 1))
                    y_hat += list(pred.detach().numpy().reshape(-1))
                    y_true += list(Y.detach().numpy().reshape(-1))

                    loss = loss1+loss2
                else:
                    loss = 0
                    for j in range(split_num):
                        Xj, Yj = X[num_sample*j:num_sample *
                                   (j+1)], Y[num_sample*j:num_sample*(j+1)]
                        Para = decoder(E[index][None, :])
                        model.updatepara(x_in, args.h_ch, args.out_ch, Para)
                        if args.dataset not in ['pm25', 'temp']:
                            # binary classification
                            pred = torch.sigmoid(model(Xj)).cpu()
                            loss1 = F.binary_cross_entropy(
                                pred.reshape(-1, 1), Yj.reshape(-1, 1))
                            y_hat_logit += list(pred.detach().numpy().reshape(-1))
                            pred = torch.as_tensor(
                                (pred.detach() - 0.5) > 0).float()
                        else:  # ['pm25','temp']
                            pred = model(Xj).cpu()
                            loss1 = criterion_l1(
                                pred.reshape(-1, 1), Yj.reshape(-1, 1))
                        y_hat += list(pred.detach().numpy().reshape(-1))
                        y_true += list(Yj.detach().numpy().reshape(-1))
                        loss += (loss1+loss2)

                loss.backward()
                epochloss += loss
                optimizer.step()
                optimizer.zero_grad()
                optimizer2.step()
                optimizer2.zero_grad()

        # print(time.time()-time1)
        return epochloss.item()/len(domains), E_dict, y_hat, y_true, y_hat_logit

    def test(test_domains, Einter_model, decoder, model, E_dict, S_0_key, S_0):
        y_hat, y_true, y_hat_logit = [], [], []
        loss_total, pred_num = 0, 0
        num_nodes = S_0.shape[0]+1

        if args.model == "SIGNN":
            Einter_model.eval()
            decoder.eval()
            x_E = [torch.zeros(args.E_size).to(device)]
            for i in S_0_key:
                x_E.append(E_dict[i])
            x_E = torch.stack(x_E)
            is_source = torch.ones(
                S_0.shape[0]+1, dtype=torch.bool, device=device)
            is_source[0] = False
            for i, data in enumerate(test_domains):  # Iterate over the domains
                if args.dataset in ['pm25', "temp"]:
                    s = torch.tensor([float(data[0].split('_')[0]), float(
                        data[0].split('_')[1])], dtype=torch.float32, device=device)[None, :]
                else:
                    s = torch.tensor(
                        data[0, -2:], dtype=torch.float32, device=device)[None, :]
                S = torch.cat([s, S_0], 0)
                edge_index = knn_graph(S, k=args.num_neighbors)
                edge_index_2rd, num_2nd_neighbors, edx_1st, edx_2nd = triplets(
                    edge_index, num_nodes)

                E = Einter_model(S, edge_index, edge_index_2rd,
                                 edx_1st, edx_2nd, x_E, is_source, args.edge_rep)
                if args.dataset in ['pm25', "temp"]:
                    X, Y = torch.tensor(
                        data[1][:, 1:-2], dtype=torch.float32, device=device), torch.tensor(data[1][:, 0])
                else:
                    X, Y = torch.tensor(
                        data[:, 1:-2], dtype=torch.float32, device=device), torch.tensor(data[:, 0])
                # Para=decoder(E.mean(axis=0)[None,:])
                Para = decoder(E[0][None, :])
                model.updatepara(x_in, args.h_ch, args.out_ch, Para)
                if args.dataset not in ['pm25', 'temp']:
                    # binary classification
                    pred = torch.sigmoid(model(X)).cpu()
                    loss1 = F.binary_cross_entropy(
                        pred.reshape(-1, 1), Y.reshape(-1, 1))
                    y_hat_logit += list(pred.detach().numpy().reshape(-1))
                    pred = torch.as_tensor((pred.detach() - 0.5) > 0).float()
                else:  # ['pm25','temp']
                    pred = model(X).cpu()
                    loss1 = criterion_l1(pred.reshape(-1, 1), Y.reshape(-1, 1))
                loss = loss1
                loss_total += loss.detach() * len(Y.reshape(-1, 1))
                pred_num += len(Y.reshape(-1, 1))

                y_hat += list(pred.detach().numpy().reshape(-1))
                y_true += list(Y.detach().numpy().reshape(-1))
        return loss_total/pred_num, y_hat, y_true, y_hat_logit

    if True:
        best_Einter_model = copy.deepcopy(Einter_model)
        best_decoder = copy.deepcopy(decoder)
        best_E_dict = copy.deepcopy(E_dict)
        best_val = 1e10
        old_lr = 100000
        for epoch in range(args.nepoch):
            train_loss, E_dict, y_hat, y_true, y_hat_logit = train(
                train_domains, Einter_model, decoder, model, E_dict, S_0_key, S_0)
            if args.dataset in ['pm25', "temp"]:
                train_mae = mean_absolute_error(y_true, y_hat)
                print(
                    (f"epoch[{epoch:d}] train_loss : {train_loss:.3f} train_mae : {train_mae:.3f}"))
            else:
                roc = roc_auc_score(y_true, y_hat_logit)
                print(
                    (f"epoch[{epoch:d}] train_loss : {train_loss:.3f}  roc: {roc:.3f}"))
            if epoch % args.test_per_round == 0:
                val_loss, yhat_val, ytrue_val, y_hat_logit_val = test(
                    valid_domains, Einter_model, decoder, model, E_dict, S_0_key, S_0)
                # test_loss, yhat_test, ytrue_test,y_hat_logit_test = test(test_domains,Einter_model,decoder,model,E_dict,S_0_key,S_0)
                if args.dataset in ['pm25', "temp"]:
                    val_mae = mean_absolute_error(ytrue_val, yhat_val)
                    print(
                        blue(f"epoch[{epoch:d}] val_mae : {val_mae:.3f}"))
                    val = val_mae
                else:
                    roc = roc_auc_score(ytrue_val, y_hat_logit_val)
                    print(blue(
                        f"epoch[{epoch:d}] val roc: {roc:.3f}"))
                    val = -roc
                if val < best_val:
                    best_val = val
                    best_Einter_model = copy.deepcopy(Einter_model)
                    best_decoder = copy.deepcopy(decoder)
                    best_E_dict = copy.deepcopy(E_dict)
            if epoch >= 50:
                lr = scheduler.optimizer.param_groups[0]['lr']
                if old_lr != lr:
                    print(red('lr'), epoch, (lr), sep=', ')
                    old_lr = lr
                    # old_lr2=lr2
                scheduler.step(val)

    valid_loss, yhat_val, ytrue_val, y_hat_logit_val = test(
        valid_domains, best_Einter_model, best_decoder, model, best_E_dict, S_0_key, S_0)
    test_loss, yhat_test, ytrue_test, y_hat_logit_test = test(
        test_domains, best_Einter_model, best_decoder, model, best_E_dict, S_0_key, S_0)
    if args.dataset in ['pm25', "temp"]:
        valid_mae = mean_absolute_error(ytrue_val, yhat_val)
        print(
            blue(f"best_val  valid_mae: {valid_mae:.3f}"))

        test_mae = mean_absolute_error(ytrue_test, yhat_test)
        print(
            blue(f"best_test test_mae: {test_mae:.3f}"))
    else:
        roc = roc_auc_score(ytrue_val, y_hat_logit_val)
        print(blue(
            f"best_val  roc: {roc:.3f}"))
        roc = roc_auc_score(ytrue_test, y_hat_logit_test)
        print(blue(
            f"best_test roc: {roc:.3f}"))
    print("done")


if __name__ == '__main__':
    args = get_args()
    if args.manualSeed:
        Seed = args.random_seed
    else:
        Seed = random.randint(1, 10000)
    print("Random Seed: ", Seed)
    random.seed(Seed)
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    flag = 0
    if args.dataset in ['pm25', "temp"]:
        dl, S_0_key, train_domains, valid_domains, test_domains = dataset.load_data(
            args.dataset, args.batchSize)
    elif args.dataset == 'flu' or args.dataset in ['argentina', 'brazil', 'chile', 'colombia', 'ecuador', 'el salvador', 'mexico', 'paraguay', 'uruguay', 'venezuela']:
        train_domains, valid_domains, test_domains = dataset.load_data(
            args.dataset, args.batchSize)
        print("event numbers of train, val, test:", train_domains[:, :, 0].sum(
        ), valid_domains[:, :, 0].sum(), test_domains[:, :, 0].sum())
        if train_domains[:, :, 0].sum()*valid_domains[:, :, 0].sum()*test_domains[:, :, 0].sum() == 0:
            print("0 data")
            flag = 1
        dl = None
        S_0_key = [pos2key(i[0, -2:]) for i in train_domains]
        S_0_key = unique(S_0_key)
    else:
        raise Exception('Dataset not recognized.')

    main(args, dl, S_0_key, valid_domains, test_domains, flag)
