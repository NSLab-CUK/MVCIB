import argparse
import copy
import logging
import math
from molecules import MoleculeDataset
import dgl
from itertools import chain
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

loss_func = nn.CrossEntropyLoss()
from metrics import accuracy_TU, MAE
import time
from pathlib import Path
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import os
import random
import dgl
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='scipy._lib.messagestream.MessageStream')

from models import Mainmodel
from util import getM_logM, load_dgl, get_A_D, load_dgl_benzene, get_mol, motif_decomp, get_3D

from script_classification import run_node_classification, run_node_clustering, update_evaluation_value

np.seterr(divide='ignore')


def collate(self, samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    batched_graph = dgl.batch(graphs)

    return batched_graph, labels


from models import Mainmodel  # , Mainmodel_continue
from torch.utils.data import DataLoader


def run_pretraining(model, pre_train_loader1, optimizer, batch_size, device):
    best_epoch = 0
    best_model = model
    best_loss = 100000000

    for epoch in range(1, args.pt_epoches):
 
        epoch_train_loss, epoch_mi_Loss, epoch_loss_2d_2_3d, epoch_loss_3d_2_2d, epoch_loss_2D, epoch_loss_3D = train_epoch_pre_training(
            model, args,
            optimizer, device,
            pre_train_loader1,
            epoch, 1,
            batch_size)
        # Gradient clipping
        max_norm = 1.0  # Example maximum norm
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if best_loss >= epoch_train_loss:
            best_model = model
            best_epoch = epoch
            best_loss = epoch_train_loss
        if epoch - best_epoch > 50:
            break
        # if epoch % 1 == 0:
        msg = "Epoch:%d		|Best_epoch:%d	|Train_loss:%0.4f	|mi:%0.4f	|2d_2_3d:%0.4f	|3d_2_2d:%0.4f	|2d:%0.4f	|3d:%0.4f" % (
            epoch, best_epoch, epoch_train_loss, epoch_mi_Loss, epoch_loss_2d_2_3d, epoch_loss_3d_2_2d, epoch_loss_2D,
            epoch_loss_3D)
        print(msg)
        if epoch % 10 == 0:
            file_name_cpt = args.output_path + f'pre_training_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}_{str(epoch)}.pt'
            torch.save(best_model, file_name_cpt)
            time.sleep(0.1)

    return best_model, best_epoch
 
def run(i, dataset_full1, feature1):
    model = Mainmodel(args, feature1, hidden_dim=args.dims, num_layers=args.num_layers,
                      num_heads=args.num_heads, k_transition=args.k_transition, encoder=args.encoder).to(device)
    best_model = model
    best_loss = 100000000
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    data_all1 = dataset_full1.data_all

    batch_size = args.batch_size

    pre_train_loader1 = DataLoader(data_all1, batch_size=batch_size, shuffle=True,
                                   collate_fn=dataset_full1.collate)  # Pre-training model

    if args.pretrained_mode == 1:
 
        file_name_cpt = args.output_path + f'pre_trained_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
        best_model, _ = run_pretraining(model, pre_train_loader1, optimizer, batch_size, device)
        torch.save(best_model, file_name_cpt)
        time.sleep(0.1)

    print(f"\nFinished pretraining models ...")
    return 0
 
from dgl.data.utils import save_graphs, load_graphs
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')
 
def get_num_features(ds):
    dim = args.feature_list
    if len(args.dataset_list) != len(dim):
        print(f"len features erro ")
        raise SystemExit()
    f = -1
    for i in range(len(args.dataset_list)):
        if args.dataset_list[i] == ds:
            f = dim[i]
    return f

import time
def load_graphdataset(dataset_name):
    args.dataset = dataset_name
    num_features = 2;
    args.num_features = 2  # get_num_features(dataset_name)

    start_time = time.time()
    graph_lists, graph_labels = load_graphs("pts/" + dataset_name + "_k_transition_" + str(args.k_transition) + ".bin")
    end_time = time.time() ; execution_time = end_time - start_time
    print(f" Checking dataset {args.dataset} Execution time: {execution_time:.6f} seconds")
    start_time = time.time()

    print(f"loading _subgraphs_khop_ 0")
    k_subgraph_lists = torch.load("pts/" + dataset_name + "_subgraphs_khop_" + str(args.k_transition) + "_0.pt")  #
    k_subgraph_lists = k_subgraph_lists['set_subgraphs']
 

    end_time = time.time() ; execution_time = end_time - start_time

    print(f" Loading subgraphs Execution time: {execution_time:.6f} seconds")
    start_time = time.time()

    print(f"loading _subgraphs_fgs 0")
    fg_subgraph_lists = torch.load("pts/" + dataset_name + "_subgraphs_fgs_0.pt")  #
    fg_subgraph_lists = fg_subgraph_lists['set_subgraphs']
 
    end_time = time.time() ; execution_time = end_time - start_time
    print(f" Loading FGs Execution time: {execution_time:.6f} seconds")
    print(
        f"len(subgraph_lists): {len(k_subgraph_lists)} | len(fg_subgraph_lists): {len(fg_subgraph_lists)} | len(graph_lists): {len(graph_lists)}")
 
    samples_all = []
    checking_label = []
    num_node_list = []
    for i in range(len(k_subgraph_lists)):
        current_graph = graph_lists[i]
        current_label = graph_labels['glabel'][i]
        checking_label.append(current_label)
        num_node_list.append(current_graph.num_nodes())
        current_subgraphs = k_subgraph_lists[i]
        current_subgraphs_fgs = fg_subgraph_lists[i]

        if len(current_subgraphs) != len(current_subgraphs_fgs):
            print(f"len problem: {len(current_subgraphs)}  != {len(current_subgraphs_fgs)}")
            raise SystemExit()
        # current_trans_logM = trans_logMs[i]
        pair = (current_graph, current_label, current_subgraphs, current_subgraphs_fgs)
        samples_all.append(pair)
    random.shuffle(samples_all)
    dataset_full = LoadData(samples_all, 'pre_training')

    return dataset_full, num_features


import pandas as pd
from rdkit import Chem

def main():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = args.dataset + "-" + timestr + ".log"
    Path("./exp_logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename="exp_logs/" + log_file, filemode="w", level=logging.INFO)
    logging.info("Starting on device: %s", device)
    logging.info("Config: %s ", args)

    args.output_path = "outputs/"
 
    file_name = "original_datasets/chembl_pretraining_data.xlsx"
    path = "pts/" + args.dataset + "_k_transition_" + str(args.k_transition) + ".bin"
    if not os.path.exists(path):
        print(f" generate_graphs ...")
        generate_graphs(file_name)
    dataset_full1, feature1 = load_graphdataset(args.dataset)

    runs_acc = []
    for i in tqdm(range(args.run_times)):
        acc = run(i, dataset_full1, feature1)
        runs_acc.append(acc)


def simile2networkx(smile):
    mol = Chem.MolFromSmiles(smile)
    # Get adjacency matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)
 
    G = nx.from_numpy_array(adjacency_matrix)
    return G


def generate_graphs(file_name):
    graph_ds = GraphClassificationDataset()
    graph_labels = [];
    set_k_subgraphs = [];
    set_FG_subgraphs = [];
    range_saved_list = [100000,200000,300000,400000]
    miss = 0
    num_mis = 0
    data = pd.read_excel(file_name)
    count = 0
    for i, row in data.iterrows():
        count += 1
 
        if i % 50 == 0:
            print(f"processing mol {i}")
            time.sleep(0.1)
        smile = data['smiles'][i]
        y = 1  
        G = simile2networkx(smile)
 
        try:
            g, dist_tensor = load_dgl_benzene(G, smile);
            mol = get_mol(smile);
            cliques = motif_decomp(mol);
            coords_tensor = get_3D(mol)
 
            graph_ds.graph_lists.append(g)
            # trans_logMs.append(trans_logM)
            graph_labels.append(y)
            graph_ds.graph_labels = {"glabel": torch.tensor(graph_labels)}
            node_ids = g.nodes()
            k_subgraphs = []
            for individual_node in node_ids:
                sg = dgl.khop_in_subgraph(g, individual_node, k=args.k_transition)[0]
                ids = sg.ndata[dgl.NID]
                sg.ndata['c'] = coords_tensor[ids]
                # sg.ndata['d'] = dist_tensor[ids]

                k_subgraphs.append(sg)
            set_k_subgraphs.append(k_subgraphs)
            # extract FGs
            fg_subgraphs = []
            for ind_node in node_ids:
                for cliq in cliques:
                    if ind_node in cliq:
                        sg = dgl.node_subgraph(g, cliq)
                        sg.ndata['c'] = coords_tensor[cliq]
 
                        fg_subgraphs.append(sg)
                        mis_fg = 1
                        continue
 
            set_FG_subgraphs.append(fg_subgraphs)

            if i in range_saved_list or i == 430709: # 430710
                id_str = str(int(i/100000) -1 )
                torch.save({"set_subgraphs": set_k_subgraphs },
                           "pts/" + args.dataset + "_subgraphs_khop_" + str(args.k_transition) + "_" + id_str + ".pt")
                print(f"_subgraphs_khop_ saved {id_str}. ...")
                time.sleep(0.5)
                torch.save({"set_subgraphs": set_FG_subgraphs }, "pts/" + args.dataset + "_subgraphs_fgs" + "_" + id_str + ".pt")
                print(f"_subgraphs_fgs saved  {id_str}. ...")
                time.sleep(0.5)
                set_k_subgraphs = []; set_FG_subgraphs = []
        except:
            miss += 1
            print(f'Missing loading dgl graph: {i} | count {count}')
    print(f"total dgl graph missing: {miss} | num_mis: {num_mis}")

    save_graphs("pts/" + args.dataset + "_k_transition_" + str(args.k_transition) + ".bin", graph_ds.graph_lists, graph_ds.graph_labels)
    print(f"graph dataset saved. ended ...")

# return graph_ds

def LoadData(samples_all, DATASET_NAME):
    return MoleculeDataset(samples_all, DATASET_NAME)


def train_epoch_pre_training(model, args, optimizer, device, data_loader, epoch, k_transition, batch_size=16):
    model.train()

    epoch_loss = 0;
 
    count = 0
    epoch_mi_Loss = 0;
    epoch_loss_2d_2_3d = 0;
    epoch_loss_3d_2_2d = 0;
    epoch_loss_2D = 0;
    epoch_loss_3D = 0
    for iter, (batch_graphs, _, batch_subgraphs, batch_fgs) in enumerate(data_loader):
        count = iter
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['x'].float().to(device)
        edge_index = batch_graphs.edges()

        optimizer.zero_grad()

        flatten_batch_subgraphs = list(chain.from_iterable(batch_subgraphs))
        flatten_batch_subgraphs = dgl.batch(flatten_batch_subgraphs).to(device)
        x_subs = flatten_batch_subgraphs.ndata['x'].float().to(device)

        flatten_batch_fgs = list(chain.from_iterable(batch_fgs))
        flatten_batch_fgs = dgl.batch(flatten_batch_fgs).to(device)
        x_fgs = flatten_batch_fgs.ndata['x'].float().to(device)
 
        z1, z2, h_2D_readout, h_3D_readout, mi_Loss, loss_2d_2_3d, loss_3d_2_2d, loss_2D, loss_3D = model.forward(
            batch_graphs, batch_x,
            flatten_batch_subgraphs, x_subs,
            flatten_batch_fgs, x_fgs,
            device, batch_size)
        parameters = []
        for parameter in model.parameters():
            parameters.append(parameter.view(-1))
        loss = mi_Loss + loss_2d_2_3d + loss_3d_2_2d + loss_2D + loss_3D

        loss.backward()
 
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_mi_Loss += mi_Loss;
        epoch_loss_2d_2_3d += loss_2d_2_3d;
        epoch_loss_3d_2_2d += loss_3d_2_2d;
        epoch_loss_2D += loss_2D;
        epoch_loss_3D += loss_3D

    epoch_loss /= (count + 1)
    epoch_mi_Loss /= (count + 1)
    epoch_loss_2d_2_3d /= (count + 1)
    epoch_loss_3d_2_2d /= (count + 1)
    epoch_loss_2D /= (count + 1)
    epoch_loss_3D /= (count + 1)
    return epoch_loss, epoch_mi_Loss, epoch_loss_2d_2_3d, epoch_loss_3d_2_2d, epoch_loss_2D, epoch_loss_3D


class GraphClassificationDataset:
    def __init__(self):
        self.graph_lists = []  # A list of DGLGraph objects
        self.graph_labels = []
        self.subgraphs = []

    def add(self, g):
        self.graph_lists.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
 
        return self.graphs[i], self.labels[i], self.subgraphs[i]


def load_bias(g):
    M, logM = getM_logM(g, kstep=args.k_transition)
    return M, logM
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments")

    #
    parser.add_argument("--dataset", default="pre-train", help="Dataset")
    parser.add_argument("--model", default="Mainmodel", help="GNN Model")

    parser.add_argument("--run_times", type=int, default=1)

    parser.add_argument("--drop", type=float, default=0.1, help="dropout")
    parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

    # adding args
    parser.add_argument("--device", default="cuda:0", help="GPU ids")

    # transfer learning
    parser.add_argument("--pretrained_mode", type=int, default=1)
    parser.add_argument("--domain_adapt", type=int, default=0)

    parser.add_argument("--d_transfer", type=int, default=300)
    parser.add_argument("--layer_relax", type=int, default=0)
    parser.add_argument("--readout_f", default="sum")  # mean set2set sum
    # transfer learning

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--testmode", type=int, default=0)

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--pt_epoches", type=int, default=200)
    parser.add_argument("--ft_epoches", type=int, default=200)
    parser.add_argument("--useAtt", type=int, default=1)
    parser.add_argument("--dims", type=int, default=300, help="hidden dims")
    parser.add_argument("--task", default="graph_classification")
    parser.add_argument("--encoder", default="GIN")
    parser.add_argument("--recons_type", default="adj")
    parser.add_argument("--k_transition", type=int, default = 3)
    parser.add_argument("--angstrom", type=float, default = 1.5) 
    parser.add_argument("--num_layers", type=int, default= 4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--output_path", default="outputs/", help="outputs model")

    parser.add_argument("--pre_training", default="1", help="pre_training or not")
    parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
    parser.add_argument("--file_name", default="outputs_excels.xlsx", help="file_name dataset")

    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    main()




