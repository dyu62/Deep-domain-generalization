import imp
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv import GATConv, GATv2Conv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch_scatter import scatter
from torch import Tensor
from utils.utils import triplets, get_angle, GaussianSmearing
from torch.nn import ModuleList
from math import pi as PI


class Einterp(nn.Module):
    def __init__(self,  h_channel=16, Esize=32, localdepth=2, num_interactions=3, combinedepth=3, transposition='True', batchnorm="False"):
        super(Einterp, self).__init__()
        self.training = True
        self.h_channel = h_channel
        self.Esize = Esize
        self.localdepth = localdepth
        self.num_interactions = num_interactions
        self.combinedepth = combinedepth
        self.batchnorm = batchnorm

        self.activation = nn.ReLU()

        self.mlps = ModuleList()
        num_gaussians = (1, 12)
        self.theta_expansion = GaussianSmearing(-PI, PI, num_gaussians[1])
        mlp_dist = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                mlp_dist.append(Linear(num_gaussians[0], h_channel))
            else:
                mlp_dist.append(Linear(h_channel, h_channel))
            if self.batchnorm == "True":
                mlp_dist.append(nn.BatchNorm1d(h_channel))
            mlp_dist.append(self.activation)
        mlp_theta = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                mlp_theta.append(
                    Linear(num_gaussians[1], h_channel)
                )
            else:
                mlp_theta.append(Linear(h_channel, h_channel))
            if self.batchnorm == "True":
                mlp_theta.append(nn.BatchNorm1d(h_channel))
            mlp_theta.append(self.activation)
        mlp_coords = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                mlp_coords.append(
                    Linear(2, h_channel)
                )
            else:
                mlp_coords.append(Linear(h_channel, h_channel))
            if self.batchnorm == "True":
                mlp_coords.append(nn.BatchNorm1d(h_channel))
            mlp_coords.append(self.activation)
        self.mlps.append(mlp_dist)
        self.mlps.append(mlp_theta)
        self.mlps.append(mlp_coords)

        self.interactions = ModuleList()
        for i in range(self.num_interactions):
            block = SPNN(
                in_ch=self.Esize,
                hidden_channels=self.h_channel,
                activation=self.activation,
                combinedepth=self.combinedepth,
                batchnorm=self.batchnorm,
                input_cat_num=2  # 2*MLP_out_channel_num dist+theta or two coords
            )
            self.interactions.append(block)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(2):
            for lin in self.mlps[i]:
                if isinstance(lin, Linear):
                    torch.nn.init.xavier_uniform_(lin.weight)
                    lin.bias.data.fill_(0)
        for block in self.interactions:
            block.reset_parameters()

    def forward(self, coords, edge_index, edge_index_2rd, edx_1st, edx_2nd, E, is_source, edge_rep):
        geo = coords
        distances = {}
        thetas = {}
        if edge_rep:
            i, j, k = edge_index_2rd
            # i_to_j_dis = (geo[j] - geo[i]).norm(p=2, dim=1)
            # distances[1] = scatter(i_to_j_dis,edx_1st,dim=0,dim_size=edge_index.shape[1],reduce='mean')
            distances[1] = (geo[edge_index[0]] -
                            geo[edge_index[1]]).norm(p=2, dim=1)
            theta_ijk = get_angle(geo[j] - geo[i], geo[k] - geo[j])
            v1 = torch.cross(F.pad(
                geo[j] - geo[i], (0, 1)), F.pad(geo[k] - geo[j], (0, 1)), dim=1)[..., 2]  # normal
            # v2=torch.ones_like(v1,device=v1.device,dtype=v1.dtype)
            flag = torch.sign((v1))
            flag[flag == 0] = -1
            thetas[1] = scatter(theta_ijk * flag, edx_2nd, dim=0,
                                dim_size=edge_index.shape[1], reduce='min')
            thetas[1] = self.theta_expansion(thetas[1])

            geo_encoding_1st = distances[1][:, None]
            geo_encoding_1st[geo_encoding_1st ==
                             0] = 1E-10  # avoid inf after pow
            geo_encoding_1st = torch.pow(geo_encoding_1st, -1)
            for lin in self.mlps[0]:
                geo_encoding_1st = lin(geo_encoding_1st)
            geo_encoding_2nd = thetas[1]
            for lin in self.mlps[1]:
                geo_encoding_2nd = lin(geo_encoding_2nd)
            geo_encoding = torch.cat(
                [geo_encoding_1st, geo_encoding_2nd], dim=-1)
        else:
            coords_j = geo[edge_index[0]]
            coords_i = geo[edge_index[1]]
            for lin in self.mlps[2]:
                coords_j = lin(coords_j)
                coords_i = lin(coords_i)
            geo_encoding = torch.cat([coords_j, coords_i], dim=-1)
        node_feature = E
        for interaction in self.interactions:
            node_feature = interaction(
                node_feature, geo_encoding, edge_index, is_source)
        return node_feature


class SPNN(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        hidden_channels,
        read2="add",
        activation=torch.nn.ReLU(),
        combinedepth=3,
        batchnorm="False",
        input_cat_num=2
    ):
        super(SPNN, self).__init__()
        self.activation = activation
        self.read2 = read2
        self.combinedepth = combinedepth
        self.batchnorm = batchnorm
        self.input_cat_num = input_cat_num
        self.att = Parameter(torch.Tensor(
            1, hidden_channels), requires_grad=True)

        self.combine = ModuleList()
        for i in range(self.combinedepth + 1):
            if i == 0:
                # ?*hidden_num+2*MLP_out_channel_num
                self.combine.append(
                    Linear(in_ch*2+hidden_channels * self.input_cat_num, hidden_channels))
            else:
                self.combine.append(Linear(hidden_channels, hidden_channels))
            if self.batchnorm == "True":
                self.combine.append(nn.BatchNorm1d(hidden_channels))
            self.combine.append(self.activation)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.combine:
            if isinstance(lin, Linear):
                torch.nn.init.xavier_uniform_(lin.weight)
                lin.bias.data.fill_(0)
        glorot(self.att)

    def forward(self, node_feature, geo_encoding, edge_index, is_source):
        j, i = edge_index
        E = node_feature.clone()
        if node_feature is None:
            concatenated_vector = geo_encoding
        else:
            node_attr_0st = node_feature[i]
            node_attr_1st = node_feature[j]
            concatenated_vector = torch.cat(
                [
                    node_attr_0st,
                    node_attr_1st,
                    geo_encoding,
                ],
                dim=-1,
            )
        x_i = concatenated_vector
        for lin in self.combine:
            x_i = lin(x_i)

        E_j = E[edge_index[0]]
        x_i = F.leaky_relu(x_i)
        alpha = F.leaky_relu(x_i * self.att).sum(dim=-1)
        alpha = softmax(alpha, edge_index[1])

        message = E_j * alpha.unsqueeze(-1)
        out_E = scatter(message, edge_index[1], dim=0, reduce=self.read2)
        E[~is_source] = out_E[~is_source]
        return E


class downstreamMLP(nn.Module):
    # doing downstream task
    def __init__(self):
        super(downstreamMLP, self).__init__()

        self.m_list = None
        self.bias_list = None

    def updatepara(self, in_ch, h_ch, out_ch, Para):
        # paranum=(in_ch*2+g_ch)*out_ch+out_ch+out_ch
        m1_size = (in_ch)*h_ch
        b1_size = h_ch
        m2_size = (h_ch)*h_ch
        b2_size = h_ch
        m3_size = h_ch*out_ch
        b3_size = out_ch

        m_1 = Para[:, :m1_size]  # 2*50
        m_2 = Para[:, m1_size:m1_size+m2_size]  # 50*50
        m_3 = Para[:, m1_size+m2_size:m1_size+m2_size+m3_size]  # 50*1
        b_1 = Para[:, m1_size+m2_size+m3_size:m1_size +
                   m2_size+m3_size+b1_size]  # 1*50
        b_2 = Para[:, m1_size+m2_size+m3_size+b1_size:m1_size +
                   m2_size+m3_size+b1_size+b2_size]  # 1*50
        b_3 = Para[:, -b3_size:]  # 1
        self.m_list, self.bias_list = [m_1.view(-1, in_ch, h_ch), m_2.view(-1, h_ch, h_ch), m_3.view(-1, h_ch, out_ch)], \
            [b_1.view(-1, 1, h_ch), b_2.view(-1, 1, h_ch),
             b_3.view(-1, 1, out_ch)]

    def forward(self, x):
        pred = x[:, None, :]
        for i, m in enumerate(self.m_list):
            if i != len(self.m_list)-1:
                pred = torch.relu(
                    torch.add(torch.matmul(pred, m), self.bias_list[i]))
            else:
                pred = (torch.add(torch.matmul(pred, m), self.bias_list[i]))
                # torch.sigmoid
        return pred


class parameter_decoder(nn.Module):
    def __init__(self, in_ch=128, h_ch=256, hlayer_num=1, out_ch=1001, activation='relu', dropout=False):
        super(parameter_decoder, self).__init__()
        act_list = [
            "relu",
            "lrelu"
        ]
        act_val = [
            torch.nn.ReLU(),
            torch.nn.LeakyReLU()
        ]
        for act_i, act in enumerate(act_list):
            if activation == act:
                self.activation = act_val[act_i]
        self.in_ch = in_ch
        self.h_ch = h_ch
        self.hlayer_num = hlayer_num
        self.dropout = dropout

        self.mlp = torch.nn.ModuleList()
        if self.hlayer_num == 0:
            self.mlp.append(nn.Linear(in_ch, out_ch))
        else:
            for i in range(self.hlayer_num):
                if i == 0:
                    self.mlp.append(nn.Linear(in_ch, h_ch))
                else:
                    self.mlp.append(nn.Linear(h_ch, h_ch))
                self.mlp.append(self.activation)
                if self.dropout:
                    self.mlp.append(nn.Dropout(p=0.5))
            self.mlp.append(nn.Linear(h_ch, out_ch))

    def forward(self, E):
        for layer in self.mlp:
            E = layer(E)
        return E
