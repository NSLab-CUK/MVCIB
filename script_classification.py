import argparse
import copy
import logging
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import scipy.sparse as sp
import os
import os.path
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset
import dgl.sparse as dglsp
import numpy as np
import torch
from tqdm import tqdm

import dgl
from dgl import LaplacianPE
# from dgl.nn import LaplacianPosEnc
import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from collections import defaultdict
import itertools

from collections import deque

from gnnutils import make_masks, test, add_original_graph, load_webkb, load_planetoid, \
    load_wiki, load_bgp, load_film, load_airports, train_finetuning_class, train_finetuning_cluster, test_cluster

from collections import defaultdict
from collections import deque
from torch.utils.data import DataLoader, ConcatDataset
import random as rnd
# from gnnutils import update_evaluation_value
import warnings

warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")
warnings.filterwarnings("ignore", message="scipy._lib.messagestream.MessageStream size changed")


db_name = 0


def update_evaluation_value(file_path, colume, row, value):
    try:
        df = pd.read_excel(file_path)

        df[colume][row] = value

        df.to_excel(file_path, sheet_name='data', index=False)

        return
    except:
        print("Error when saving results! Save again!")
        time.sleep(3)


def run_node_classification(index_excel, ds_name, output_path, file_name, data_all, num_features, out_size,
                            num_classes, g, adj_org, M, trans_logM, sim, phi, B, degree, k_transition, device,device_2,
                            num_epochs, current_epoch, aug_check,sim_check,phi_check,test_node_degree):
    print("running run_node_classification")

    data = pd.read_excel(file_name)
    index_excel = index_excel

    if data['Mean'][index_excel] != -1:
        acc = data['Mean'][index_excel]
        print(f'Already run_node_classification done in file, index: {index_excel}, mean: {acc}')
    else:
        dataset = data['dataset'][index_excel]
        lr = data['lr'][index_excel]
        dims = data['dims'][index_excel]
        out_size = data['out_size'][index_excel]
        num_layers = data['num_layers'][index_excel]
        k_transition = data['k_transition'][index_excel]
        alfa = data['alfa'][index_excel]
        beta = data['beta'][index_excel]

        # if 1==1:
        # cp_filename = output_path + ds_name + f'_0.001_64_2_3_0.5_0.5.pt'
        # lr  = 1e-3; dims = 64; num_layers = 2
        print(f"Node class process - {index_excel}")

        cp_filename = output_path + f'{dataset}_{lr}_{dims}_{num_layers}_{k_transition}_{alfa}_{beta}.pt'

        if not os.path.exists(cp_filename):
            print(f"run_node_classification: no file: {cp_filename}")
            print(f"run_node_classification: no file: {cp_filename}")
            return None

        runs_acc = []
        for i in tqdm(range(1)):
            print(f'run_node_classification, run time: {i}')
            acc = run_epoch_node_classification(i, data_all, num_features, out_size, num_classes,
                                                g, adj_org, M, trans_logM, sim, phi, B, degree, k_transition,
                                                cp_filename, dims,
                                                num_layers, lr, device, device_2,num_epochs, current_epoch, aug_check,sim_check,phi_check,test_node_degree)
            runs_acc.append(acc)

        runs_acc = np.array(runs_acc) * 100

        update_evaluation_value(file_name, 'Mean', index_excel, runs_acc.mean())
        update_evaluation_value(file_name, 'Variant', index_excel, runs_acc.std())

        final_msg = "Node Classification: Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
        print(final_msg)


def run_epoch_node_classification(i, data, num_features, out_size, num_classes,
                                  g, adj_org, M, trans_logM, sim, phi, B, degree, k_transition, cp_filename, dims,
                                  num_layers, lr, device,device_2, num_epochs, current_epoch,aug_check,sim_check,phi_check,test_node_degree):
    graph_name = ""
    best_val_acc = 0
    best_model = None
    pat = 20
    best_epoch = 0

    # fine tuning & testing
    print('fine tuning ...')

    model = Transformer_class(num_features, out_size, num_classes, hidden_dim=dims,
                              num_layers=num_layers, num_heads=4, graph_name=graph_name,
                              cp_filename=cp_filename, aug_check = aug_check,sim_check = sim_check,phi_check = sim_check) #.to(device)

    dataset_1 = ['cora', 'citeseer', 'Photo','WikiCS']
    for ds in dataset_1:
        if ds in cp_filename:
            model.to(device)
    else:
        if torch.cuda.device_count() > 1:

            id_2 = int(str(device_2).split(":")[1])
            model = torch.nn.DataParallel(model, device_ids=[id_2])

    best_model = model
    # for module_name, module in f_model.named_modules():
    # 	print(f"Transformer_class module_name : {module_name} , value : {module}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print("creating  random mask")
    data = make_masks(data, val_test_ratio=0.2)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # save dataload
    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["test_mask"] = test_mask

    test_check = 0
    for epoch in range(1, num_epochs):
        current_epoch = epoch
        train_loss, train_acc = train_finetuning_class(model, data, train_mask, optimizer, device,device_2,  g=g, adj_org=adj_org,
                                                       M=M, trans_logM=trans_logM, sim=sim, phi=phi, B=B,
                                                       k_transition=k_transition,
                                                       pre_train=0, current_epoch=current_epoch)

        if epoch % 1 == 0:
            # (model, test_data, mask, device, g, trans_logM, sim, phi, k_transition, current_epoch, alfa, beta ):
            valid_acc, valid_f1 = test(model, data, val_mask, device,device_2, g=g, adj_org=adj_org, trans_logM=trans_logM,
                                       sim=sim, phi=phi, B=B, degree=degree,
                                       k_transition=k_transition, current_epoch=current_epoch, test_node_degree=0)

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_model = model
                best_epoch = epoch
                pat = (pat + 1) if (pat < 5) else pat
            else:
                pat -= 1

            if epoch % 1 == 0:
                print(
                    'Epoch: {:02d}, best_epoch: {:02d}, Train Loss: {:0.4f}, Train Acc: {:0.4f}, Val Acc: {:0.4f} '.format(
                        epoch, best_epoch, train_loss, train_acc, valid_acc))
            # logging.info(
            #	'Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:0.4f}, Val Acc: {:0.4f} '.format(epoch, train_loss,
            #																					  train_acc, valid_acc))
            if epoch - best_epoch > 100:
                print("1 validation patience reached ... finish training")
                break
        # if pat < -200:
        # 	print("validation patience reached ... finish training")
        # logging.info("validation patience reached ... finish training")
        # break

    # Testing
    test_check = 1
    test_acc, test_f1 = test(best_model, data, test_mask, device,device_2, g=g, adj_org=adj_org, trans_logM=trans_logM, sim=sim,
                             phi=phi, B=B, degree=degree,
                             k_transition=k_transition, current_epoch=current_epoch, test_node_degree=test_node_degree)
    print('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f}, F1_test: {:0.4f}'.format(
        best_epoch, best_val_acc, test_acc, test_f1))
    # logging.info('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f} '.format(best_epoch, best_val_acc, test_acc))
    return test_acc, test_f1


# (args.index_excel, args.output_path, args.file_name, data, num_features, args.out_size,
# num_classes, g, M, logM, sim ,phi,B, args.k_transition, device, args.run_times_fine, n_edges, current_epoch= epoch)
def run_node_clustering(index_excel, ds_name, output_path, file_name, data_all, num_features, out_size,
                        num_classes, g, M, logM, sim, phi, B, k_transition, device,device_2, num_epochs, adj, d, n_edges,
                        current_epoch, aug_check,sim_check,phi_check):
    data = pd.read_excel(file_name)
    index_excel = index_excel

    if data['acc_cluster'][index_excel] != -1:
        acc = data['acc_cluster'][index_excel]
        print(f'Already run_node_clustering done in file, index: {index_excel}, mean: {acc}')
    else:
        dataset = data['dataset'][index_excel]
        lr = data['lr'][index_excel]
        dims = data['dims'][index_excel]
        out_size = data['out_size'][index_excel]
        num_layers = data['num_layers'][index_excel]
        k_transition = data['k_transition'][index_excel]
        alfa = data['alfa'][index_excel]
        beta = data['beta'][index_excel]

        # if 1==1:
        # cp_filename = output_path + ds_name + f'_0.001_64_2_3_0.5_0.5.pt'
        # lr  = 1e-3; dims = 64; num_layers = 2
        print(f"Node clustering process - {index_excel}")

        cp_filename = output_path + f'{dataset}_{lr}_{dims}_{num_layers}_{k_transition}_{alfa}_{beta}.pt'

        if os.path.isfile(cp_filename) == False:
            print(f"run_node_clustering: no file {cp_filename}")
            return None

        # data_all, num_features, out_size, k_eigenvector, num_classes,  isbgp, g, A_k, D, D_dim, Kindices, M, I,trans_logM  = generate_D_I_M_G()

        runs_acc = []
        for i in tqdm(range(1)):
            print(f'run time: {i}')
            acc, precision, recall, nmi, q, c = run_epoch_node_clustering(i, data_all, num_features, out_size,
                                                                          num_classes,
                                                                          g, M, logM, sim, phi, B, k_transition,
                                                                          cp_filename, dims, num_layers, lr, device,device_2,
                                                                          num_epochs, adj, d, n_edges, current_epoch, aug_check,sim_check,phi_check)
            runs_acc.append(acc)
            time.sleep(1)
        runs_acc = np.array(runs_acc) * 100

        update_evaluation_value(file_name, 'acc_cluster', index_excel, acc)
        update_evaluation_value(file_name, 'nmi', index_excel, nmi)
        update_evaluation_value(file_name, 'q', index_excel, q)
        update_evaluation_value(file_name, 'c', index_excel, c)

        print('acc: {:03f}, precision: {:0.4f}, recall: {:0.4f}, nmi: {:0.4f}, Q: {:0.4f}, , C: {:0.4f}'.format(acc,
                                                                                                                precision,
                                                                                                                recall,
                                                                                                                nmi, q,
                                                                                                                c))


def run_epoch_node_clustering(i, data, num_features, out_size, num_classes, g, M, logM, sim, phi, B, k_transition,
                              cp_filename, dims, num_layers, lr, device,device_2, num_epochs, adj, d, n_edges, current_epoch,aug_check,sim_check,phi_check):
    graph_name = ""
    best_val_acc = 0
    best_model = None
    pat = 20
    best_epoch = 0

    # fine tuning & testing
    print('fine tuning run_epoch_node_clustering...')

    model = Transformer_cluster(num_features, out_size, num_classes, hidden_dim=dims,
                                num_layers=num_layers, num_heads=4, graph_name=graph_name,
                                cp_filename=cp_filename, aug_check=aug_check,sim_check=sim_check,phi_check=phi_check) #.to(device)

    dataset_1 = ['cora', 'citeseer', 'Photo','WikiCS']
    for ds in dataset_1:
        if ds in cp_filename:
            model.to(device)
    else:
        if torch.cuda.device_count() > 1:
            id_2 = int(str(device_2).split(":")[1])
            model = torch.nn.DataParallel(model, device_ids=[id_2])

    best_model = model
    # for module_name, module in f_model.named_modules():
    # 	print(f"Transformer_class module_name : {module_name} , value : {module}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print("creating  random mask")
    data = make_masks(data, val_test_ratio=0.0)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    # save dataload.
    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["test_mask"] = test_mask
    adj = torch.FloatTensor(adj).to(device)
    for epoch in range(1, num_epochs):
        current_epoch = epoch
        train_loss = train_finetuning_cluster(model, data, train_mask, optimizer, device, device_2,g=g,
                                              M=M, trans_logM=logM, sim=sim, phi=phi, B=B, k_transition=k_transition,
                                              pre_train=0, adj=adj, d=d, n_edges=n_edges, current_epoch=current_epoch)

        if epoch % 1 == 0:
            print('Epoch: {:02d}, Train Loss: {:0.4f}'.format(epoch, train_loss))

    # Testing

    acc, precision, recall, nmi, q, c = test_cluster(best_model, data, train_mask, optimizer, device,device_2, g=g,
                                                     M=M, trans_logM=logM, sim=sim, phi=phi, B=B,
                                                     k_transition=k_transition, pre_train=0, adj=adj, d=d,
                                                     n_edges=n_edges, current_epoch=current_epoch)

    # print('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f}, F1_test: {:0.4f}'.format(best_epoch, best_val_acc, acc, precision))
    # logging.info('Best Val Epoch: {:03d}, Best Val Acc: {:0.4f}, Test Acc: {:0.4f} '.format(best_epoch, best_val_acc, acc))
    print('acc: {:03f}, precision: {:0.4f}, recall: {:0.4f}, nmi: {:0.4f}, Q: {:0.4f}, , C: {:0.4f}'.format(acc,
                                                                                                            precision,
                                                                                                            recall, nmi,
                                                                                                            q, c))
    return acc, precision, recall, nmi, q, c
