import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path
import numpy as np
import dgl.function as fn
from torch_geometric.nn import global_mean_pool
#from torch_scatter import scatter_mean, scatter_add, scatter_std
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn import Set2Set
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import SAGEConv, EGNNConv, GraphConv

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class MLP(torch.nn.Module):

    def __init__(self, num_features, num_classes, dims=16):
        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, dims),
            torch.nn.ReLU(),
            torch.nn.Linear(dims, num_classes))

    def forward(self, x):
        x = self.mlp(x)
        return x

class GIN(nn.Module):
	def __init__(self, input_dim, hidden_dim=64):
		super().__init__()
		self.ginlayers = nn.ModuleList()
		self.batch_norms = nn.ModuleList()
		num_layers = 5
		for layer in range(num_layers - 1):  # excluding the input layer
			if layer == 0:
				mlp = MLP(input_dim, hidden_dim, hidden_dim)
			else:
				mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
			self.ginlayers.append(GINConv(mlp, learn_eps=False))  # set to True if learning epsilon
			self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
	def forward(self, g, h):
		for i, layer in enumerate(self.ginlayers):
			h = layer(g, h)
			h = self.batch_norms[i](h)
			h = F.relu(h)
		return h



class EGNNs(nn.Module):
	def __init__(self, in_feats, hidden_dim=64, h_feats=3):
		super(EGNNs, self).__init__()
		self.conv1 = EGNNConv(in_feats, hidden_dim, hidden_dim)
		self.conv2 = EGNNConv(hidden_dim, hidden_dim, hidden_dim)
		self.conv3 = EGNNConv(hidden_dim, hidden_dim, hidden_dim)
		self.conv4 = EGNNConv(hidden_dim, hidden_dim, hidden_dim)
		self.conv5 = EGNNConv(hidden_dim, hidden_dim, hidden_dim)
		self.batchnorm1 = nn.BatchNorm1d(in_feats);
		self.batchnorm2 = nn.BatchNorm1d(in_feats)
		self.batchnorm3 = nn.BatchNorm1d(in_feats);
		self.batchnorm4 = nn.BatchNorm1d(in_feats);
 
	def forward(self, g, in_feat, noise=0):
		coord_feat = g.ndata['c']
		if noise == 1:
			coord_feat += g.ndata['n']
		h, coord_feat = self.conv1(g, in_feat, coord_feat)
		h = self.batchnorm1(h)
		h = F.relu(h)

		h, coord_feat = self.conv2(g, h, coord_feat)
		h = self.batchnorm2(h)
		h = F.relu(h)

		h, coord_feat = self.conv3(g, h, coord_feat)
		h = self.batchnorm3(h)
		h = F.relu(h)

		h, _ = self.conv4(g, h, coord_feat)
		h = self.batchnorm4(h)
		h = F.relu(h)
 
		return h, coord_feat

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.pytorch.conv.GraphConv(num_features, hidden_dim * 2, allow_zero_in_degree=True)
        self.conv2 = dgl.nn.pytorch.conv.GraphConv(hidden_dim * 2, hidden_dim * 2, allow_zero_in_degree=True)
        self.conv3 = dgl.nn.pytorch.conv.GraphConv(hidden_dim * 2, hidden_dim, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat);
        h = F.relu(h)
        h = self.conv2(g, h);
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        return h
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv3 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        return h


class Mainmodel_finetuning(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, num_classes, cp_filename, encoder,
                 dims=10):
        super().__init__()
        self.tau = 1.0
        self.dataset = args.dataset
        self.readout = args.readout_f
        self.s2s = Set2Set(hidden_dim, 2, 1)
        self.num_classes = num_classes

        self.in_dim = args.d_transfer
 

        self.batch_size = args.batch_size
        self.useAtt = args.useAtt
        self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)

        self.hidden_dim = hidden_dim
        self.k_transition = k_transition
        self.reduce_d = torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.num_nodes = -1
        self.device = args.device
        print(f" self.device args.device {self.device}")

        self.dataset = args.dataset
        self.tasks = ['ZINC', 'Peptides-struct', 'FreeSolv', 'ESOL', 'Lipo', 'QM9']
        if args.task == "graph_regression":
            self.predict = nn.Sequential(
                nn.Linear(self.hidden_dim, dims),
                nn.ReLU(),
                nn.Linear(dims, 1))
        elif args.task == "graph_classification":
            self.predict = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, num_classes)
            )
        else:
            print(f"checking mainmodel_finetuning task ...")

        self.MLP = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        if encoder == "GIN":
            self.Encoder1 = GIN(self.in_dim, hidden_dim)
            self.Encoder2 = GIN(self.in_dim, hidden_dim)
        elif encoder == "GCN":
            self.Encoder1 = GCN(self.in_dim, hidden_dim)
            self.Encoder2 = GCN(self.in_dim, hidden_dim)
        elif encoder == "GraphSAGE":
            self.Encoder1 = GraphSAGE(self.in_dim, hidden_dim)
            self.Encoder2 = GraphSAGE(self.in_dim, hidden_dim)
        else:
            print("Bug there is no pre-defined Encoders")
            raise SystemExit()
        print(f"Loading pre-trained model .pt  {cp_filename} ... ")

        self.model = torch.load(cp_filename, map_location=args.device)
   

    def forward(self, batch_g, batch_x, flatten_batch_subgraphs, x_subs, flatten_batch_fgs, x_fgs, device, batch_size=16):
        noise = batch_g.ndata['n']
        self.batch_size = batch_size
        nodes_list = batch_g.batch_num_nodes()

        batch_x = batch_x.type(torch.int)
        x_subs = x_subs.type(torch.int)
        x_fgs = x_fgs.type(torch.int)

        _, _, _, _, _, _, _, z1, z2 = self.model.extract_features(
            nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, flatten_batch_fgs, x_fgs, noise, device,mode='ft')

        interaction_map = torch.cat((z1, z2), -1)  # z1 + z2   --> 2d

        interaction_map = self.MLP(interaction_map)  # 2d ---> d

        if self.dataset in self.tasks:
            return self.predict(interaction_map), 0, 0, 0
        else:
            sig = nn.Sigmoid()
            pre = sig(self.predict(interaction_map))
            return pre, interaction_map, 0, 0

    def loss(self, scores, targets):
        loss = nn.BCELoss()
        l = loss(scores.float(), targets.float())
        return l

    def loss_CrossEntropy(self, scores, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(scores.to(torch.float32), targets.squeeze(dim=-1))
        return loss

    def loss_RMSE(self, scores, targets):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(scores, targets))
        return loss

    def BCEWithLogitsLoss(self, scores, targets):
        loss = nn.BCEWithLogitsLoss()(scores, targets)

        return loss

    def lossMAE(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss


from torch.distributions import Normal
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus


# Encoder architecture
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
 
        self.net = nn.Sequential(
            nn.Linear(2 * self.z_dim, self.z_dim),
            nn.LeakyReLU(0.2, True),
 
            nn.Linear(self.z_dim, self.z_dim * 2))

    def forward(self, x):
 
        params = self.net(x)
 
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive
 

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class MIEstimator(nn.Module):
    def __init__(self, size1, size2):  # 128 64
        super(MIEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 128),
 
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 64),
 
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
 

        pos = self.net(x)
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))

        return -softplus(-pos).mean() - softplus(neg).mean()


import torch.nn.functional as F


class Mainmodel(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_layers, num_heads, k_transition, encoder, dims=100):
        super().__init__()
        self.tau = 1.0
        self.recons_type = args.recons_type
        self.useAtt = args.useAtt

        self.readout = args.readout_f
        self.hidden_dim = hidden_dim
        self.k_transition = k_transition
       

        self.encoder_2d = Encoder(hidden_dim)
        self.encoder_3d = self.encoder_2d  # Encoder(hidden_dim)
        self.in_dim = args.d_transfer
        self.transfer_d = nn.Linear(in_dim, self.in_dim, bias=False)

        self.embedding_h = nn.Linear(self.in_dim, hidden_dim, bias=False)

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.in_dim);
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.in_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data);
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
 
        self.device = args.device
        print(f" pt args.device {args.device}")
 
        self.MLP = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        self.MLP3d = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3))
        if encoder == "GIN":
            self.Encoder1 = GIN(self.in_dim, hidden_dim)
            self.Encoder2 = EGNNs(self.in_dim, hidden_dim)
        else:
            print("Bug there is no pre-defined Encoders")
            raise SystemExit()

        self.mi_estimator1 = MIEstimator(self.hidden_dim * 2, self.hidden_dim)
        self.mi_estimator2 = MIEstimator(self.hidden_dim * 2, self.hidden_dim)
        self.mu = 0.01
        self.eta = 0.003

    def _compute_2d_3d_loss(self, batch_g, subgraphs_2D, subgraphs_3D, batch_size):

        loss_3d_2d = self.loss_recon_adj(subgraphs_3D, batch_g, batch_size)

        dis = batch_g.ndata['d']
        num_nodes = subgraphs_2D.size(0)  # 32
        loss_2d_3d = 0
        for i in range(num_nodes):
            for j in range(20):
                 
                loss_2d_3d += torch.abs(torch.norm(subgraphs_2D[i] - subgraphs_2D[j]) - dis[i][j])
        loss_2d_3d /= num_nodes
        return loss_2d_3d, loss_3d_2d

    def _compute_2d_loss(self, batch_g, subgraphs_2D):
        row_num, col_num = subgraphs_2D.size()
        adj = batch_g.adj().to_dense()
        recon_interaction_map = torch.mm(subgraphs_2D, subgraphs_2D.t())

        loss = torch.sum((recon_interaction_map - adj) ** 2) / (row_num * col_num)
        return loss

    def _compute_3d_loss(self, subgraphs_3D_noise, noise_total):
        epsilon = self.MLP3d(subgraphs_3D_noise)
        num_nodes = subgraphs_3D_noise.size(0)  # 32
        loss_3d = 0
        cos = F.cosine_similarity(epsilon, noise_total, dim=1)
        for i in range(num_nodes):
            loss_3d += 1 - cos[i]
        loss_3d /= num_nodes
        return loss_3d
 
    def _compute_kl_loss(self, h_2D_readout, h_3D_readout, nodes_list, device):
        loss_sum = 0
        z = len(nodes_list)
        z1_all = torch.tensor(()).to(device);
 
        z2_all = torch.tensor(()).to(device);
        for i in range(z):
            h_2d = h_2D_readout[i, :].reshape(1, -1);
            h_3d = h_3D_readout[i, :].reshape(1, -1)
 
            p_z1_given_v1 = self.encoder_2d(h_2d);
            p_z2_given_v2 = self.encoder_3d(h_3d)
 
            z1 = p_z1_given_v1.rsample();
            z2 = p_z2_given_v2.rsample()

            mi_gradient1 = self.mi_estimator1(h_2d, z2).mean();
            mi_gradient2 = self.mi_estimator2(h_3d, z1).mean()
            mi_gradient = (mi_gradient1 + mi_gradient2) / 2

            kl_23 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1);
            kl_32 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
            skl = (kl_23 + kl_32).mean() / 2

            loss = - mi_gradient * self.mu + self.eta * skl
            loss_sum += loss
 
            z1_all = torch.cat((z1_all, z1), 0);
            z2_all = torch.cat((z2_all, z2), 0)
 
        return loss_sum, z1_all, z2_all

    def forward(self, batch_g, batch_x, flatten_batch_subgraphs, x_subs, flatten_batch_fgs, x_fgs, device,
                batch_size=16, mode='pt'):
        noise = batch_g.ndata['n']
        self.batch_size = batch_size
        nodes_list = batch_g.batch_num_nodes()
 
        self.device = device
 

        batch_x = batch_x.type(torch.int);
        x_subs = x_subs.type(torch.int);
        x_fgs = x_fgs.type(torch.int)

        h_2D_readout, h_3D_readout, subgraphs_2D, subgraphs_3D, subgraphs_3D_noise, noise_total, mi_Loss, z1, z2 = self.extract_features(
            nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, flatten_batch_fgs, x_fgs, noise, device,
            mode='pt')

        if mode == 'pt':
            loss_2d_2_3d, loss_3d_2_2d = self._compute_2d_3d_loss(batch_g, subgraphs_2D, subgraphs_3D, self.batch_size)
            loss_3D = self._compute_3d_loss(subgraphs_3D_noise, noise_total)
            loss_2D = self._compute_2d_loss(batch_g, subgraphs_2D)

        return z1, z2, h_2D_readout, h_3D_readout, mi_Loss, loss_2d_2_3d, loss_3d_2_2d, loss_2D, loss_3D

    def extract_features(self, nodes_list, batch_g, batch_x, flatten_batch_subgraphs, x_subs, flatten_batch_fgs, x_fgs,
                         noise, device, mode='pt'):
 
        x_subs = self.x_embedding1(x_subs[:, 0]) + self.x_embedding2(x_subs[:, 1])
        x_fgs = self.x_embedding1(x_fgs[:, 0]) + self.x_embedding2(x_fgs[:, 1])

 

        # 2D Encoders
        khop_subgraphs_2D = self.Encoder1(flatten_batch_subgraphs, x_subs);
        fcs_subgraphs_2D = self.Encoder1(flatten_batch_fgs, x_fgs)

        # 2D readout
        flatten_batch_subgraphs.ndata['x'] = khop_subgraphs_2D;
        khop_subgraphs_2D_readout = dgl.sum_nodes(flatten_batch_subgraphs, 'x')
        flatten_batch_fgs.ndata['x'] = fcs_subgraphs_2D;
        fg_subgraphs_2D_readout = dgl.sum_nodes(flatten_batch_fgs, 'x')

        # 3D Encoders
        khop_subgraphs_3D, _ = self.Encoder2(flatten_batch_subgraphs, x_subs);
        fcs_subgraphs_3D, _ = self.Encoder2(flatten_batch_fgs, x_fgs)

        if mode == 'pt':  # noise encoding
            khop_subgraphs_3D_noise, k_hop_noise = self.Encoder2(flatten_batch_subgraphs, x_subs, noise=1)
            fcs_subgraphs_3D_noise, fgs_noise = self.Encoder2(flatten_batch_fgs, x_fgs, noise=1)

        # 3D readout
        flatten_batch_subgraphs.ndata['x'] = khop_subgraphs_3D;
        khop_subgraphs_3D_readout = dgl.sum_nodes(flatten_batch_subgraphs, 'x')

        flatten_batch_fgs.ndata['x'] = fcs_subgraphs_3D;
        fg_subgraphs_3D_readout = dgl.sum_nodes(flatten_batch_fgs, 'x')

        # add normalization
        khop_subgraphs_2D_readout = F.normalize(khop_subgraphs_2D_readout, dim=1);
        fg_subgraphs_2D_readout = F.normalize(fg_subgraphs_2D_readout, dim=1)
        khop_subgraphs_3D_readout = F.normalize(khop_subgraphs_3D_readout, dim=1);
        fg_subgraphs_3D_readout = F.normalize(fg_subgraphs_3D_readout, dim=1)

        if mode == 'pt':
            # 3D readout noise information
            flatten_batch_subgraphs.ndata['x'] = khop_subgraphs_3D_noise;
            khop_subgraphs_3D_noise = dgl.sum_nodes(flatten_batch_subgraphs, 'x')
            flatten_batch_fgs.ndata['x'] = fcs_subgraphs_3D_noise;
            fcs_subgraphs_3D_noise = dgl.sum_nodes(flatten_batch_fgs, 'x')

            flatten_batch_subgraphs.ndata['n'] = k_hop_noise;
            k_hop_noise = dgl.sum_nodes(flatten_batch_subgraphs, 'n')
            flatten_batch_fgs.ndata['n'] = fgs_noise;
            fgs_noise = dgl.sum_nodes(flatten_batch_fgs, 'n')

            # add normalization
            khop_subgraphs_3D_noise = F.normalize(khop_subgraphs_3D_noise, dim=1);
            fcs_subgraphs_3D_noise = F.normalize(fcs_subgraphs_3D_noise, dim=1)
            k_hop_noise = F.normalize(k_hop_noise, dim=1);
            fgs_noise = F.normalize(fgs_noise, dim=1)

            subgraphs_3D_noise = torch.cat((khop_subgraphs_3D_noise, fcs_subgraphs_3D_noise), -1)
            noise_total = k_hop_noise + fgs_noise

        # concat k-hop and fgs
        subgraphs_2D = torch.cat((khop_subgraphs_2D_readout, fg_subgraphs_2D_readout), -1)
        subgraphs_3D = torch.cat((khop_subgraphs_3D_readout, fg_subgraphs_3D_readout), -1)

        tup_subgraphs_2D_readout = torch.split(subgraphs_2D, tuple(nodes_list))
        tup_subgraphs_3D_readout = torch.split(subgraphs_3D, tuple(nodes_list))
        num_graphs = len(nodes_list)
        h_2D_readout = torch.tensor(()).to(device);
        h_3D_readout = torch.tensor(()).to(device)
        # if cross:
        for k in range(num_graphs):
            interaction_map = torch.mm(tup_subgraphs_2D_readout[k], tup_subgraphs_3D_readout[k].t())

            h_rev_2D = self.cross_attention(interaction_map, tup_subgraphs_2D_readout[k])  # n x 2d
            h_rev_3D = self.cross_attention(interaction_map, tup_subgraphs_3D_readout[k])  # n x 2d

            h_2D_sum = torch.sum(h_rev_2D, dim=0, keepdim=True);
            h_3D_sum = torch.sum(h_rev_3D, dim=0, keepdim=True)

            h_2D_readout = torch.cat((h_2D_readout, h_2D_sum), 0);
            h_3D_readout = torch.cat((h_3D_readout, h_3D_sum), 0)

        mi_Loss, z1, z2 = self._compute_kl_loss(h_2D_readout, h_3D_readout, nodes_list, device)

        if mode == 'pt':
            return h_2D_readout, h_3D_readout, subgraphs_2D, subgraphs_3D, subgraphs_3D_noise, noise_total, mi_Loss, z1, z2
        if mode == 'ft':
            return h_2D_readout, h_3D_readout, subgraphs_2D, subgraphs_3D, None, None, mi_Loss, z1, z2

    def cross_attention(self, A, t):
        A_fm = F.softmax(A, dim=1)
        # Perform element-wise computation
        return A_fm @ t

    ####################################################### end shared pre-training

    def loss(self, scores, targets):
        loss = nn.BCELoss()
        l = loss(scores.float(), targets.float())
        return l

    def loss_X(self, batch_g, interaction_map):
        interaction_map_X = self.reconstructX(interaction_map)
        loss = F.mse_loss(interaction_map_X, batch_g.ndata['x'])
        return loss

    def loss_recon_adj(self, interaction_map, org_graph, batch_size=16):
        row_num, col_num = interaction_map.size()
        adj = org_graph.adj().to_dense()
        recon_interaction_map = torch.mm(interaction_map, interaction_map.t())

        loss = torch.sum((recon_interaction_map - adj) ** 2) / (row_num * col_num)
        return loss

    def loss_recon(self, interaction_map, trans_logM, nodes_list):

        sp_interaction_map = torch.split(interaction_map, tuple(nodes_list))
        loss = 0
        z = len(nodes_list)
        for k in range(z):
            h = torch.mm(sp_interaction_map[k], sp_interaction_map[k].t()).to(self.device)
            row_num, col_num = h.size()
            for i in range(self.k_transition):
                loss += torch.sum(((h - (torch.FloatTensor(trans_logM[k][i])).to(self.device)) ** 2)) / (
                        row_num * col_num)
        loss = loss / (self.k_transition)
        return loss


class Reconstruct_X(torch.nn.Module):
    def __init__(self, inp, outp, dims=128):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(inp, inp / 2),
            torch.nn.ReLU(),
            torch.nn.Linear(inp / 2, outp))

    def forward(self, x):
        x = self.mlp(x)
        return x
 
 

class Reconstruct_X(torch.nn.Module):
    def __init__(self, inp, outp, dims=128):
        super().__init__()
 

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(inp, dims * 2),
            torch.nn.SELU(),
            torch.nn.Linear(dims * 2, outp))

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLPA(torch.nn.Module):

    def __init__(self, in_feats, dim_h, dim_z):
        super(MLPA, self).__init__()

        self.gcn_mean = torch.nn.Sequential(
            torch.nn.Linear(in_feats, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_z)
        )

    def forward(self, hidden):
        # GCN encoder
        Z = self.gcn_mean(hidden)
        # inner product decoder
        adj_logits = Z @ Z.T
        return adj_logits


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L = nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)

        return y


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}

    return func


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: (((edges.data[field])) / scale_constant)}

    return func

