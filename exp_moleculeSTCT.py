import argparse
import copy
import logging
import math
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

from gnnutils import make_masks, train, test, add_original_graph, load_webkb, load_planetoid, load_wiki, load_bgp, \
    load_film, load_airports, load_amazon, load_coauthor, load_WikiCS, load_crocodile, load_Cora_ML

from util import get_B_sim_phi, getM_logM, load_dgl, get_A_D, load_dgl_fromPyG

from models import Mainmodel, Mainmodel_finetuning

 

from rdkit import Chem, RDLogger
import warnings

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='scipy._lib.messagestream.MessageStream')
RDLogger.DisableLog('rdApp.*')
from util import getM_logM, load_dgl, get_A_D, load_dgl_benzene, get_mol, motif_decomp, get_3D

from script_classification import run_node_classification, run_node_clustering, update_evaluation_value

np.seterr(divide='ignore')


def collate(self, samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels)).unsqueeze(1)
    batched_graph = dgl.batch(graphs)

    return batched_graph, labels


def filter_rels(data, r):
    data = copy.deepcopy(data)
    mask = data.edge_color <= r
    data.edge_index = data.edge_index[:, mask]
    data.edge_weight = data.edge_weight[mask]
    data.edge_color = data.edge_color[mask]
    return data


from torch.utils.data import DataLoader


def run_pretraining(model, pre_train_loader, optimizer, batch_size, device):
    best_model = model
    best_loss = 100000000
    for epoch in range(1, args.pt_epoches):
        epoch_train_loss, KL_Loss, contrastive_loss, reconstruction_loss = train_epoch(model, args, optimizer, device,
                                                                                       pre_train_loader, epoch, 1,
                                                                                       batch_size)
        if best_loss >= epoch_train_loss:
            best_model = model
            best_epoch = epoch
            best_loss = epoch_train_loss
        if epoch - best_epoch > 50:
            break
 
        msg = "Epoch:%d	|Best_epoch:%d	|Train_loss:%0.4f	|KL_Loss:%0.4f	|contrastive_loss:%0.4f	|reconstruction_loss:%0.4f	" \
              % (epoch, best_epoch, epoch_train_loss, KL_Loss, contrastive_loss, reconstruction_loss)
        print(msg)
    return best_model


def run(i, dataset_full, num_features, num_classes):
    model = Mainmodel(args, num_features, hidden_dim=args.dims, num_layers=args.num_layers,
                      num_heads=args.num_heads, k_transition=args.k_transition, encoder=args.encoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    best_model = model
    best_loss = 100000000

    trainset, valset, testset = dataset_full.train, dataset_full.val, dataset_full.test
    data_all = dataset_full.data_all
    print("\nTraining Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    batch_size = args.batch_size
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=dataset_full.collate)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=dataset_full.collate)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=dataset_full.collate)
    pre_train_loader = DataLoader(data_all, batch_size=batch_size, shuffle=True,
                                  collate_fn=dataset_full.collate)  # Pre-training model

    # transfer learning
    file_name_cpt = args.output_path + f'{args.pretrained_ds}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
    # if pretained is true, only run pretraining
    if args.pretrained_mode == 1:
        if not os.path.exists(file_name_cpt):
            best_model, best_epoch = run_pretraining(model, pre_train_loader, optimizer, batch_size, device)
            torch.save(best_model, file_name_cpt)
            update_evaluation_value(args.file_name, 'Best_epoch', args.index_excel, best_epoch)
            print(f"\nFinished pre-trained model ...")
        else:
            print(f"\nexists pretrained model, quiting ...")
        raise SystemExit()

    print(f"\nFine tunning the pre-trained model ...")
    time.sleep(0.1)
    # end transfer learning

    runs_acc = []

    for i in tqdm(range(1)):
        print(f'\nrun time: {i}')
        acc, best_epoch = run_epoch_graph_classification(best_model, train_loader, val_loader, test_loader,
                                                         num_features, file_name_cpt, batch_size=batch_size)
        runs_acc.append(acc)
        time.sleep(0.1)

    runs_acc = np.array(runs_acc) * 100

    final_msg = "Graph classification: Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
    print(final_msg)

    return 0


from train_molsider import train_epoch, evaluate_network, train_epoch_graph_classification, train_epoch_domainadaptation
from models import Mainmodel, Mainmodel_finetuning


def run_epoch_graph_classification(model, train_loader, val_loader, test_loader, num_features, file_name, batch_size):
    model = Mainmodel_finetuning(args, num_features, hidden_dim=args.dims, num_layers=args.num_layers,
                                 num_heads=args.num_heads, k_transition=args.k_transition, num_classes=args.num_classes,
                                 cp_filename=file_name, encoder=args.encoder).to(device)

    best_model = model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_loss = 100000000
    t0 = time.time()
    per_epoch_time = []
    epoch_train_AUCs, epoch_val_AUCs, epoch_test_AUCs = [], [], []
    epoch_train_losses, epoch_val_losses = [], []
    for epoch in range(1, args.ft_epoches):
        start = time.time()
        epoch_train_loss, epoch_train_auc, optimizer = train_epoch_graph_classification(args, model, optimizer, device,
                                                                                        train_loader, epoch, batch_size)

        epoch_val_loss, epoch_val_auc = evaluate_network(args, model, optimizer, device, val_loader, epoch,
                                                         batch_size)  # (model, device, val_loader, epoch)
        _, epoch_test_auc = evaluate_network(args, model, optimizer, device, test_loader, epoch,
                                             batch_size)  # (model, device, test_loader, epoch)
        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)
        epoch_train_AUCs.append(epoch_train_auc)
        epoch_val_AUCs.append(epoch_val_auc)
        epoch_test_AUCs.append(epoch_test_auc)
        if best_loss >= epoch_train_loss:
            best_model = model;
            best_epoch = epoch;
            best_loss = epoch_train_loss
        if epoch - best_epoch > 50:
            print(f"Finish epoch - best_epoch > 100")
            break
        if epoch % 1 == 0:
            print(
                f'Epoch: {epoch}	|Best_epoch: {best_epoch}	|Train_loss: {np.round(epoch_train_loss, 6)}	|Val_loss: {np.round(epoch_val_loss, 6)}	| Train_auc: {np.round(epoch_train_auc, 6)}	| Val_auc: {np.round(epoch_val_auc, 6)}	| epoch_test_auc: {np.round(epoch_test_auc, 6)} ')
        per_epoch_time.append(time.time() - start)
        # Stop training after params['max_time'] hours
        if time.time() - t0 > 48 * 3600:
            print('-' * 89)
            print("Max_time for training elapsed")
            break
    _, test_acc = evaluate_network(args, best_model, optimizer, device, test_loader, epoch,
                                   batch_size)  # (best_model, device, test_loader, epoch)
    index = epoch_val_AUCs.index(max(epoch_val_AUCs))
    test_auc = epoch_test_AUCs[index]
    train_auc = epoch_train_AUCs[index]
    print("Train AUC: {:.4f}".format(train_auc))
    print("Test AUC: {:.4f}".format(test_auc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    return test_acc, best_epoch


##################################################################################################################################

import collections
from collections import defaultdict
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset, ZINC, MoleculeNet
from dgl.data.utils import save_graphs, load_graphs


def main():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file = args.dataset + "-" + timestr + ".log"
    Path("./exp_logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename="exp_logs/" + log_file, filemode="w", level=logging.INFO)
    logging.info("Starting on device: %s", device)
    logging.info("Config: %s ", args)

    graph_lists = []
    graph_labels = []

    print(f'Checking dataset {args.dataset}')
    if args.dataset in ["MUV", "SIDER", "Tox21", "ToxCast", "ClinTox"]:
        dataset = MoleculeNet(root='original_datasets/' + args.dataset, name=args.dataset)
        args.num_classes = dataset.num_classes
        args.num_features = 2  # dataset.num_features
        num_features = args.num_features
        num_classes = args.num_classes
    else:
        raise NotImplementedError
        raise SystemExit()
    path = "pts/" + args.dataset + "_k_transition_" + str(args.k_transition) + ".bin"

    if not os.path.exists(path):
        print(f" generate_graphs ...")
        generate_graphs_new(dataset)

    print(f"#Loading graphs and labels...")
    dataset_full = load_graphdataset(args.dataset)

    runs_acc = []

    for i in tqdm(range(args.run_times)):
        acc = run(i, dataset_full, num_features, num_classes)
        runs_acc.append(acc)


def load_graphdataset(dataset_name):
    args.num_features = 2  # get_num_features(dataset_name)
    graph_lists = [];
    graph_labels = []

    start_time = time.time()
    graph_lists, graph_labels = load_graphs("pts/" + dataset_name + "_k_transition_" + str(args.k_transition) + ".bin")
    end_time = time.time();
    execution_time = end_time - start_time
    print(f" Checking dataset {args.dataset} Execution time: {execution_time:.6f} seconds")

    start_time = time.time()
    k_subgraph_lists = torch.load("pts/" + dataset_name + "_subgraphs_khop_" + str(args.k_transition) + ".pt")  #
    k_subgraph_lists = k_subgraph_lists['set_subgraphs']
    end_time = time.time();
    execution_time = end_time - start_time
    print(f" Loading subgraphs Execution time: {execution_time:.6f} seconds")

    start_time = time.time()
    fg_subgraph_lists = torch.load("pts/" + dataset_name + "_subgraphs_fgs.pt")  #
    fg_subgraph_lists = fg_subgraph_lists['set_subgraphs']
    end_time = time.time();
    execution_time = end_time - start_time
    print(f" Loading FGs Execution time: {execution_time:.6f} seconds")
    print(
        f"len(subgraph_lists): {len(k_subgraph_lists)} | len(fg_subgraph_lists): {len(fg_subgraph_lists)} | len(graph_lists): {len(graph_lists)}")

    samples_all = []
    checking_label = []
    num_node_list = []
    for i in range(len(graph_lists)):
        current_graph = graph_lists[i]
        current_label = graph_labels['glabel'][i]
        checking_label.append(current_label)
        num_node_list.append(current_graph.num_nodes())
        current_subgraphs = k_subgraph_lists[i]
        current_subgraphs_fgs = fg_subgraph_lists[i]

        if len(current_subgraphs) != len(current_subgraphs_fgs):
            print(f"len problem {i}: {len(current_subgraphs)}  != {len(current_subgraphs_fgs)}")
            raise SystemExit()
        pair = (current_graph, current_label, current_subgraphs, current_subgraphs_fgs)
        samples_all.append(pair)
    random.shuffle(samples_all)
    dataset_full = LoadData(samples_all, args.dataset)

    return dataset_full


import pandas as pd


def simile2networkx(smile):
    mol = Chem.MolFromSmiles(smile)
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)
    G = nx.from_numpy_array(adjacency_matrix)
    return G


def generate_graphs_new(dataset):
    if args.dataset in ["ToxCast"]:
        file_name = "original_datasets/ToxCast/toxcast/raw/toxcast_data.csv"
        data = pd.read_csv(file_name)
        d_smile = data['smiles']
        d_class = data.iloc[:, 1:]
    elif args.dataset in ["ClinTox"]:
        file_name = "original_datasets/ClinTox/clintox/raw/clintox.csv"
        data = pd.read_csv(file_name)
        d_smile = data['smiles']
        d_class = data.iloc[:, 1:]
    elif args.dataset in ["Tox21"]:
        file_name = "original_datasets/Tox21/tox21/raw/tox21.csv"
        data = pd.read_csv(file_name)
        d_smile = data['smiles']
        d_class = data.iloc[:, 0:12]
    elif args.dataset in ["SIDER"]:
        file_name = "original_datasets/SIDER/sider/raw/sider.csv"
        data = pd.read_csv(file_name)
        d_smile = data['smiles']
        d_class = data.iloc[:, 1:28]
    elif args.dataset in ["MUV"]:
        file_name = "original_datasets/MUV/muv/raw/muv.csv"
        data = pd.read_csv(file_name)
        d_smile = data['smiles']
        d_class = data.iloc[:, :17]
    else:
        print(f" checking datasets ...")
        raise SystemExit()
    graph_ds = GraphClassificationDataset()
    graph_labels = [];
    set_k_subgraphs = [];
    set_FG_subgraphs = [];
    miss = 0;
    num_mis = 0

    count = 0
    for i, row in data.iterrows():
        if i > 15000:
            break;
        count += 1
        if i % 20 == 0:
            print(f"processing mol {i}")
            time.sleep(0.1)
        smile = d_smile[i]
 
        try:
            G = simile2networkx(smile)
            g, dist_tensor = load_dgl_benzene(G, smile);
            node_ids = g.nodes()
            # print(f"node_ids: {node_ids}")
            mol = get_mol(smile);
            cliques = motif_decomp(mol);
            coords_tensor = get_3D(mol)

            graph_ds.graph_lists.append(g)

            y_tensor = torch.tensor(d_class.iloc[i]).view(1, -1)
 

            graph_labels += y_tensor
 

            k_subgraphs = []
            for individual_node in node_ids:
                sg = dgl.khop_in_subgraph(g, individual_node, k=args.k_transition)[0]
                ids = sg.ndata[dgl.NID]
                sg.ndata['c'] = coords_tensor[ids]

                k_subgraphs.append(sg)
            set_k_subgraphs.append(k_subgraphs)
            # extract FGs
            fg_subgraphs = []

            for ind_node in node_ids:
                check_exsit_in_fgs = 0
                for cliq in cliques:
                    if ind_node in cliq:
                        sg = dgl.node_subgraph(g, cliq)
                        sg.ndata['c'] = coords_tensor[cliq]
                        fg_subgraphs.append(sg)
                        mis_fg = 1;
                        check_exsit_in_fgs = 1
                        continue
                if check_exsit_in_fgs == 0:
                    # print(f"ind_node {ind_node}")
                    sg = dgl.khop_in_subgraph(g, individual_node, k=args.k_transition)[0]
                    ids = sg.ndata[dgl.NID]
                    sg.ndata['c'] = coords_tensor[ids]
                    fg_subgraphs.append(sg)
                    continue
            set_FG_subgraphs.append(fg_subgraphs)
            if len(k_subgraphs) != len(fg_subgraphs):
                print(f" error ======================================================")
                raise SystemExit()
        except:
            miss += 1
            print(f'Missing loading dgl graph: {i} | count {count}')

    graph_labels = torch.stack(graph_labels)
    graph_ds.graph_labels = {"glabel": torch.tensor(graph_labels)}

    print(f"total dgl graph missing: {miss} | num_mis: {num_mis}")
    print(f"set_k_subgraphs: {len(set_k_subgraphs)}")
    print(f"set_FG_subgraphs: {len(set_FG_subgraphs)}")

    torch.save({"set_subgraphs": set_k_subgraphs},
               "pts/" + args.dataset + "_subgraphs_khop_" + str(args.k_transition) + ".pt")
    print(f"_subgraphs_khop_ saved. ...")
    time.sleep(0.2)
    torch.save({"set_subgraphs": set_FG_subgraphs}, "pts/" + args.dataset + "_subgraphs_fgs.pt")
    print(f"_subgraphs_fgs saved. ...")
    save_graphs("pts/" + args.dataset + "_k_transition_" + str(args.k_transition) + ".bin", graph_ds.graph_lists,
                graph_ds.graph_labels)
    print(f"graph dataset saved. ...")


from molecules import MoleculeDataset


def LoadData(samples_all, DATASET_NAME):
    return MoleculeDataset(samples_all, DATASET_NAME)


def generate_graphs(dataset, k_hop):
    graph_ds = GraphClassificationDataset()
    graph_labels = []
    set_subgraphs = []
    trans_logMs = []
    miss = 0
    checking = []

    for i in range(len(dataset)):
        # if i >= 200:
        # 	print(f" testing small dataset")
        # 	break
        if i % 10 == 0:
            print(f'Processing graph_th: {i}')
            time.sleep(0.1)
        data = dataset[i]

        path = "pts/" + args.dataset + "_kstep_" + str(args.k_transition) + ".pt"
        try:
            g = load_dgl_fromPyG(data)
            if not os.path.exists(path):
                M, logM = load_bias(g)
                trans_logM = torch.from_numpy(np.array(logM)).float()
            graph_ds.graph_lists.append(g)
            trans_logMs.append(trans_logM)
            graph_labels += data.y  # not append

            ####adding set subgraphs:
            node_ids = g.nodes()
            all_subgraphs = [dgl.khop_in_subgraph(g, individual_node, k=args.k_transition)[0] for individual_node in
                             node_ids]
            set_subgraphs.append(all_subgraphs)
        except:
            miss += 1
            print(f'Missing loading dgl graph: {i}')
    print(f"total DGL missing: {miss}")

    graph_labels = torch.stack(graph_labels)

    # print(graph_labels)
    graph_ds.graph_labels = {"glabel": torch.tensor(graph_labels)}
    torch.save({"set_subgraphs": set_subgraphs}, "pts/" + args.dataset + "_subgraphs_khop_" + str(k_hop) + ".pt")
    torch.save({"trans_logMs": trans_logMs}, "pts/" + args.dataset + "_M_khop_" + str(k_hop) + ".pt")

    return graph_ds


################ checking
#							num_tasks=dataset.num_tasks
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
        # Get the i^th sample and label
        # return self.graphs[i], self.labels[i], self.trans_logM[i], self.B[i], self.adj[i], self.sim[i], self.phi[i]
        return self.graphs[i], self.labels[i], self.subgraphs[i]
 


def load_bias(g):
    M, logM = getM_logM(g, kstep=args.k_transition)
  

    return M, logM


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiments")

    # SIDER	Tox21	ClinTox	ToxCast MUV
    # python exp_moleculeSTCT.py --dataset ToxCast   --device cuda:2
    parser.add_argument("--dataset", default="ClinTox", help="Dataset")
    parser.add_argument("--model", default="Mainmodel", help="GNN Model")

    parser.add_argument("--run_times", type=int, default=1)

    parser.add_argument("--drop", type=float, default=0.1, help="dropout")
    parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

    # adding args
    parser.add_argument("--device", default="cuda:0", help="GPU ids")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--testmode", type=int, default=0)

    # transfer learning
    parser.add_argument("--pretrained_mode", type=int, default=0)
    parser.add_argument("--domain_adapt", type=int, default=0)
    parser.add_argument("--pretrained_ds", default="pre_trained", help="Loading pretrained model ")
    parser.add_argument("--d_transfer", type=int, default=300)
    parser.add_argument("--layer_relax", type=int, default=0)
    parser.add_argument("--readout_f", default="sum")  # mean set2set sum
    parser.add_argument("--adapt_epoches", type=int, default=50)
    # transfer learning

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--pt_epoches", type=int, default=250)
    parser.add_argument("--ft_epoches", type=int, default=250)
    parser.add_argument("--useAtt", type=int, default=1)
    parser.add_argument("--dims", type=int, default=300, help="hidden dims")
    parser.add_argument("--task", default="graph_classification")
    parser.add_argument("--encoder", default="GIN")
    parser.add_argument("--recons_type", default="adj")
    parser.add_argument("--k_transition", type=int, default=3)
    parser.add_argument("--angstrom", type=float, default = 1.5) 
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--output_path", default="outputs/", help="outputs model")

    parser.add_argument("--pre_training", default="1", help="pre_training or not")
    parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
    parser.add_argument("--file_name", default="outputs_excels.xlsx", help="file_name dataset")

    ################################################################################################
    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    main()