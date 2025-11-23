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
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='scipy._lib.messagestream.MessageStream')
from gnnutils import make_masks, train, test, add_original_graph, load_webkb, load_planetoid, load_wiki, load_bgp, \
	load_film, load_airports, load_amazon, load_coauthor, load_WikiCS, load_crocodile, load_Cora_ML

from util import get_B_sim_phi, getM_logM, load_dgl, get_A_D, load_dgl_fromPyG

from models import Transformer, Mainmodel, Mainmodel_finetuning

# from script_classification import run_node_classification, run_epoch_node_classification, update_evaluation_value, run_node_clustering

 
from script_classification import run_node_classification, run_node_clustering, update_evaluation_value

np.seterr(divide = 'ignore') 
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
from models import Transformer, Mainmodel, Mainmodel_finetuning, Mainmodel_domainadapt
def run_domain_adaptation(  file_name, pre_train_loader, optimizer, batch_size, device):
	model = Mainmodel_domainadapt(args, args.num_features ,hidden_dim=args.dims,num_layers=args.num_layers,
				num_heads=args.num_heads, k_transition = args.k_transition, num_classes = args.num_classes, cp_filename =file_name , encoder = args.encoder ).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
	best_model = model
	best_loss = 100000000
	best_epoch = 0
	num_adapt = args.adapt_epoches
	for epoch in range(1, num_adapt):
		epoch_train_loss, reconstruction_loss = train_epoch_domainadaptation(model, args, optimizer, device, pre_train_loader, epoch,  1, batch_size)
		if best_loss >= epoch_train_loss:
			best_model = model
			best_epoch = epoch
			best_loss = epoch_train_loss
		if epoch - best_epoch > 20:
			break
		msg = "Epoch:%d	|Best_epoch:%d	|Train_loss:%0.4f	 |reconstruction_loss:%0.4f	" % (epoch, best_epoch, epoch_train_loss,  reconstruction_loss )
		print(msg)
	return best_model, best_epoch
def run_pretraining(model, pre_train_loader, optimizer, batch_size, device):
	best_model = model
	best_loss = 100000000
	for epoch in range(1, args.pt_epoches):
		epoch_train_loss, KL_Loss, contrastive_loss, reconstruction_loss = train_epoch(model, args, optimizer, device, pre_train_loader, epoch,  1, batch_size)
		if best_loss >= epoch_train_loss:
			best_model = model
			best_epoch = epoch
			best_loss = epoch_train_loss
		if epoch - best_epoch > 50:
			break
		# msg1 = "out_degree1: SP %0.4f, Std %0.4f , EO %0.4f, Std %0.4f " % (SP.mean(), SP.std(),EO.mean(), EO.std())
		msg = "Epoch:%d	|Best_epoch:%d	|Train_loss:%0.4f	|KL_Loss:%0.4f	|contrastive_loss:%0.4f	|reconstruction_loss:%0.4f	" \
			% (epoch, best_epoch,epoch_train_loss, KL_Loss, contrastive_loss, reconstruction_loss )
		print(msg)
	return best_model
def run(i, dataset_full, num_features, num_classes ):
	model = Mainmodel(args, num_features,  hidden_dim=args.dims,num_layers=args.num_layers,
				 num_heads=args.num_heads, k_transition = args.k_transition, encoder = args.encoder).to(device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

	best_model = model
	best_loss = 100000000

	trainset, valset, testset = dataset_full.train, dataset_full.val, dataset_full.test
	data_all = dataset_full.data_all
	print("\nTraining Graphs: ", len(trainset))
	print("Validation Graphs: ", len(valset))
	print("Test Graphs: ", len(testset))
	batch_size = args.batch_size
	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn = dataset_full.collate)
	val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn = dataset_full.collate)
	test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn = dataset_full.collate)
	pre_train_loader = DataLoader(data_all, batch_size= batch_size, shuffle=True, collate_fn = dataset_full.collate )	#Pre-training model


	# transfer learning
	file_name_cpt = args.output_path + f'{args.dataset}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
	# if pretained is true, only run pretraining
	if args.pretrained_mode == 1:
		if not os.path.exists(file_name_cpt):
			best_model, best_epoch = run_pretraining(model, pre_train_loader, optimizer, batch_size, device)
			torch.save(best_model,file_name_cpt)
			update_evaluation_value(args.file_name, 'Best_epoch', args.index_excel, best_epoch)
			print(f"\nFinished pre-trained model ...")
		else:
			print(f"\nexists pretrained model, quiting ...")
		raise SystemExit()
	# for domain adapt and fine tunning 
	else:
		ds = args.pretrained_ds
		file_name_cpt = args.output_path + f'{ds}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}.pt'
		if args.domain_adapt == 1:
			file_domain_adapt = args.output_path + f'{ds}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}_{args.dataset}.pt'
			if not os.path.exists(file_domain_adapt):
				print(f"\nDomain adaptation starting ...")
				best_model, best_epoch = run_domain_adaptation( file_name_cpt, pre_train_loader, optimizer, batch_size, device)
				file_name_cpt = args.output_path + f'{ds}_{args.encoder}_{args.dims}_{args.num_layers}_{args.k_transition}_{args.dataset}.pt'
				torch.save(best_model,file_name_cpt)
			file_name_cpt = file_domain_adapt
			print(f"\nDomain adaptation finished ...")
	print(f"\nFine tunning the pre-trained model ...")
	time.sleep(0.1)
	# end transfer learning


	runs_acc = []
	
	for i in tqdm(range(3)):
		print(f'\nrun time: {i}')
		acc = run_epoch_graph_classification(best_model, train_loader, val_loader,test_loader,num_features, file_name_cpt,  batch_size = batch_size )
		runs_acc.append(acc)
		time.sleep(0.1)
	
	runs_acc = np.array(runs_acc) * 100

	final_msg = "Graph classification: Mean %0.4f, Std %0.4f" % (runs_acc.mean(), runs_acc.std())
	print(final_msg)
	return 0
from train_moleculenet import train_epoch, evaluate_network, train_epoch_graph_classification, train_epoch_domainadaptation

def run_epoch_graph_classification(model, train_loader, val_loader,test_loader,num_features, file_name, batch_size):

	model = Mainmodel_finetuning(args, num_features,hidden_dim=args.dims,num_layers=args.num_layers,
				 num_heads=args.num_heads, k_transition = args.k_transition, num_classes = args.num_classes, cp_filename =file_name , encoder = args.encoder ).to(device)
	
	best_model = model
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
	best_loss = 100000000
	t0 = time.time()
	per_epoch_time = []
	epoch_train_AUCs, epoch_val_AUCs, epoch_test_AUCs = [], [], []
	epoch_train_losses, epoch_val_losses = [], []
	for epoch in range(1, args.ft_epoches):
		start = time.time()
		epoch_train_loss, epoch_train_auc, optimizer = train_epoch_graph_classification(args, model, optimizer, device, train_loader, epoch,batch_size)

		epoch_val_loss, epoch_val_auc = evaluate_network(args, model, optimizer, device, val_loader, epoch, batch_size) 	# (model, device, val_loader, epoch)
		_, epoch_test_auc = evaluate_network(args, model, optimizer, device, test_loader, epoch, batch_size)				# (model, device, test_loader, epoch)
		epoch_train_losses.append(epoch_train_loss)
		epoch_val_losses.append(epoch_val_loss)
		epoch_train_AUCs.append(epoch_train_auc)
		epoch_val_AUCs.append(epoch_val_auc)
		epoch_test_AUCs.append(epoch_test_auc)
		if best_loss >= epoch_train_loss:
			best_model = model; best_epoch = epoch; best_loss = epoch_train_loss
		if epoch - best_epoch > 50:
			print(f"Finish epoch - best_epoch > 100")
			break
		if epoch % 1 == 0:
			print(f'Epoch: {epoch}	|Best_epoch: {best_epoch}	|Train_loss: {np.round( epoch_train_loss, 6)}	|Val_loss: {np.round(epoch_val_loss, 6)}	| Train_auc: {np.round(epoch_train_auc,6)}	| Val_auc: {np.round(epoch_val_auc, 6) }	| epoch_test_auc: {np.round(epoch_test_auc, 6) } ')
		per_epoch_time.append(time.time()-start)
		# Stop training after params['max_time'] hours
		if time.time()-t0 > 48 * 3600:
			print('-' * 89)
			print("Max_time for training elapsed")
			break
	_, test_acc = evaluate_network(args, best_model, optimizer, device, test_loader, epoch, batch_size)# (best_model, device, test_loader, epoch)
	index = epoch_val_AUCs.index(max(epoch_val_AUCs))
	test_auc = epoch_test_AUCs[index]
	train_auc = epoch_train_AUCs[index]
	print("Train AUC: {:.4f}".format(train_auc))
	print("Test AUC: {:.4f}".format(test_auc))
	print("Convergence Time (Epochs): {:.4f}".format(epoch))
	print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
	print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
	return test_acc

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
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
	if args.dataset in [ "BACE", "BBBP", "SIDER",  "Tox21", "ToxCast", "ClinTox"]:
		dataset = MoleculeNet(root='original_datasets/' + args.dataset, name = args.dataset)
		args.num_classes = dataset.num_classes #dataset.num_tasks 
		args.num_features = dataset.num_features
		num_features = args.num_features
		num_classes = args.num_classes
	else:
		raise NotImplementedError
		raise SystemExit()
	path = "pts/"+ args.dataset + "_k_transition_" + str(args.k_transition) + ".bin"
	if not os.path.exists(path):
		print(f"# 1. Constructing pts...")
		graph_ds  = generate_graphs(dataset, args.k_transition)
		save_graphs("pts/"+ args.dataset + "_k_transition_" + str(args.k_transition) + ".bin" , graph_ds.graph_lists, graph_ds.graph_labels)

		print(f"# 2. Loading graphs and labels...")
		graph_lists, graph_labels = load_graphs("pts/"+ args.dataset + "_k_transition_" + str(args.k_transition) + ".bin" )
	else:
		print(f"# 3 Loading graphs and labels...")
		graph_lists, graph_labels = load_graphs("pts/"+ args.dataset + "_k_transition_" + str(args.k_transition) + ".bin" )
	
	print(f"DEGUB generated graph process")
 

	print(f"# 4 Loading subgraphs ...")
	subgraph_lists = torch.load("pts/"+ args.dataset + "_subgraphs_khop_" + str(args.k_transition) + ".pt") # 
	subgraph_lists = subgraph_lists['set_subgraphs']
	print(f"len(graph_lists): {len(graph_lists)}|	len(subgraph_lists): {len(subgraph_lists)}")
	
	print(f"# 5 Loading transition matrices...")
	dic  = torch.load("pts/"+ args.dataset + "_M_khop_"  + str(args.k_transition) + ".pt")
	trans_logMs = dic['trans_logMs']
	
	if args.testmode ==1:
		print(f"Only generating dataset ...")
		raise SystemExit()
	
	samples_all = []
	num_features = args.num_features
	print(f'num_features: {num_features}, args.num_classes: {args.num_classes} ')
	samples_all = []
	checking_label = []
	num_node_list = []
	for i in range(len(graph_lists)):
		# if i >= 200: 
		# 	print(f" testing small dataset")
		# 	break
		current_graph = graph_lists[i]
		current_label = graph_labels['glabel'][i]
		checking_label.append(current_label)
		num_node_list.append(current_graph.num_nodes())
		current_subgraphs = subgraph_lists[i]
		current_trans_logM =  trans_logMs[i]

		pair = (current_graph, current_label, current_subgraphs, current_trans_logM)
		samples_all.append(pair)
	random.shuffle(samples_all)


	dataset_full = LoadData(samples_all, args.dataset)

	runs_acc = []

	for i in tqdm(range(args.run_times)):
		acc = run(i, dataset_full, num_features, num_classes )
		runs_acc.append(acc)

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
		if i % 10 ==0:
			print(f'Processing graph_th: {i}')
			time.sleep(0.1)
		data = dataset[i]

		path = "pts/" + args.dataset + "_kstep_" + str(args.k_transition) + ".pt"
		try:
			g = load_dgl_fromPyG(data)
			if not os.path.exists(path):
				M, logM  = load_bias(g)
				trans_logM = torch.from_numpy(np.array(logM)).float()
			graph_ds.graph_lists.append(g)
			trans_logMs.append(trans_logM)
			graph_labels+= data.y # not append

			####adding set subgraphs:
			node_ids = g.nodes()
			all_subgraphs = [dgl.khop_in_subgraph(g, individual_node, k= 1)[0] for individual_node in node_ids]
			set_subgraphs.append(all_subgraphs)
		except:
			miss+=1
			print(f'Missing loading dgl graph: {i}')
	print(f"total DGL missing: {miss}")

	graph_labels = torch.stack(graph_labels)
	
	# print(graph_labels)
	graph_ds.graph_labels = {"glabel": torch.tensor(graph_labels)}
	torch.save({"set_subgraphs": set_subgraphs}, "pts/"+ args.dataset + "_subgraphs_khop_"+ str(k_hop) +".pt")
	torch.save({"trans_logMs": trans_logMs}, "pts/"+ args.dataset + "_M_khop_" + str(k_hop)  + ".pt")

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
		#return self.graphs[i], self.labels[i], self.trans_logM[i], self.B[i], self.adj[i], self.sim[i], self.phi[i]
		return self.graphs[i], self.labels[i], self.subgraphs[i]
 


def load_bias(g):
	M, logM = getM_logM(g, kstep=args.k_transition)
 

	return M, logM


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Experiments")

	#
	parser.add_argument("--dataset", default="BACE", help="Dataset")
	parser.add_argument("--model", default="Mainmodel", help="GNN Model")

	parser.add_argument("--run_times", type=int, default=1)

	parser.add_argument("--drop", type=float, default=0.1, help="dropout")
	parser.add_argument("--custom_masks", default=True, action='store_true', help="custom train/val/test masks")

	# adding args
	parser.add_argument("--device", default="cuda:0", help="GPU ids")

	parser.add_argument("--batch_size", type=int, default= 4)
	parser.add_argument("--testmode", type=int, default= 0)

	#transfer learning
	parser.add_argument("--pretrained_mode", type=int, default= 0)
	parser.add_argument("--domain_adapt", type=int, default= 0)
	parser.add_argument("--pretrained_ds", default="pre_trained", help="Loading pretrained model ")
	parser.add_argument("--d_transfer", type=int, default= 32)
	parser.add_argument("--layer_relax", type=int, default= 0)
	parser.add_argument("--readout_f", default="sum" )	# mean set2set sum
	parser.add_argument("--adapt_epoches", type=int, default= 50)
	#transfer learning
	
	parser.add_argument("--lr", type=float, default = 1e-3, help="learning rate")
	parser.add_argument("--pt_epoches", type=int, default= 5)
	parser.add_argument("--ft_epoches", type=int, default= 5)
	parser.add_argument("--useAtt", type=int, default= 1)
	parser.add_argument("--dims", type=int, default=64, help="hidden dims")
	parser.add_argument("--task", default="graph_regression" )
	parser.add_argument("--encoder", default="GCN" )	
	parser.add_argument("--recons_type", default="adj" )
	parser.add_argument("--k_transition", type=int, default = 3) 
	parser.add_argument("--angstrom", type=float, default = 1.5) 
	parser.add_argument("--num_layers", type=int, default = 4)
	parser.add_argument("--num_heads", type=int, default = 4)
	parser.add_argument("--output_path", default="outputs/", help="outputs model")

	parser.add_argument("--pre_training", default = "1", help="pre_training or not")
	parser.add_argument("--index_excel", type=int, default="-1", help="index_excel")
	parser.add_argument("--file_name", default="outputs_excels.xlsx", help="file_name dataset")

	################################################################################################
	args = parser.parse_args()
	print(args)
	device = torch.device(args.device)
	main()