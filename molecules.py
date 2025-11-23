import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        
        with open(data_dir + "/%s.pickle" % self.split,"rb") as f:
            self.data = pickle.load(f)

        if self.num_graphs in [10000, 1000]:
            # loading the sampled indices from file ./zinc_molecules/<split>.index
            with open(data_dir + "/%s.index" % self.split,"r") as f:
                data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
                self.data = [ self.data[i] for i in data_idx[0] ]

            assert len(self.data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
        

        
        self.graph_lists = []
        self.graph_labels = []
        self.n_samples = len(self.data)
        self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        return self.graph_lists[idx], self.graph_labels[idx]
    
    
class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        self.num_atom_type = 28 
        self.num_bond_type = 4 
        
        data_dir='./data/molecules'
        
        if self.name == 'ZINC-full':
            data_dir='./data/molecules/zinc_full'
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=220011)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=24445)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=5000)
        else:            
            self.train = MoleculeDGL(data_dir, 'train', num_graphs=10000)
            self.val = MoleculeDGL(data_dir, 'val', num_graphs=1000)
            self.test = MoleculeDGL(data_dir, 'test', num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time()-t0))
        


def self_loop(g):
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass    
    
    return full_g



def laplacian_positional_encoding(g, pos_enc_dim):
    A = g.adjacency_matrix(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g

def wl_positional_encoding(g):
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


# ogbg-molbace
# ogbg-molbbbp
# ogbg-molclintox
# ogbg-molmuv
# ogbg-molpcba
# ogbg-molsider
# ogbg-moltox21
# ogbg-moltoxcast
# ogbg-molhiv
# ogbg-molesol
# ogbg-molfreesolv
# ogbg-mollipo
# ogbg-molchembl
# ogbg-ppa
# ogbg-code2
class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self,dataset, name):
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        if self.name == 'pre_training':
            self.data_all = dataset
        elif self.name == 'FreeSolv':         # 642       #classes 1
            self.train = dataset[:380]
            self.val = dataset[380:500]
            self.test = dataset[500:]
            self.data_all = dataset
        elif self.name == 'ESOL':           # 1128       #classes 1
            self.train = dataset[:650]
            self.test = dataset[650:850]
            self.val = dataset[850:]
            self.data_all = dataset  
        elif self.name == 'SIDER': # 1300
 
            self.train = dataset[:900]
            self.test = dataset[900: ]
            self.val = dataset[900:]
            self.data_all = dataset  
        elif self.name == 'ToxCast': #8566
            self.train = dataset[:5400]
            self.test = dataset[5400:7200]
            self.val = dataset[7200:]
            self.data_all = dataset 
        elif self.name == 'ClinTox':# 1469 79.47 1.32
            self.train = dataset[:1200]
            self.test = dataset[1200: ]
            self.val = dataset[1200:]
            self.data_all = dataset 
        elif self.name == 'Tox21': # 7778
            self.train = dataset[:4800]
            self.test = dataset[4800:6400]
            self.val = dataset[6400: ]
            self.data_all = dataset        
        elif self.name == 'BACE':  # 1513
            self.train = dataset[:900]
            self.test = dataset[900:1200]
            self.val = dataset[1200: ]
            self.data_all = dataset  
        elif self.name == 'BBBP': #1998
            self.train = dataset[:1200]
            self.test = dataset[1200:1600]
            self.val = dataset[1600:]
            self.data_all = dataset  
        ############################################
        elif self.name == 'PROTEINS':       #1113
            self.train = dataset[:700]
            self.val = dataset[700:900]
            self.test = dataset[900: ]
            self.data_all = dataset
        elif self.name == 'Mutagenicity':       #1113
            self.train = dataset[:2800]
            self.test= dataset[2800:3600]
            self.val  = dataset[3600: ]
            self.data_all = dataset
        elif self.name == 'ENZYMES':        #599
            self.train = dataset[:480]
            self.test = dataset[480:540]
            self.val = dataset[540:]
            self.data_all = dataset
        elif self.name == 'NCI1':  # 4110
            self.train = dataset[:2400]
            self.test = dataset[2400:3200]
            self.val = dataset[3200:]
            self.data_all = dataset
        elif self.name == 'NCI109':
            self.train = dataset[:2400]
            self.test = dataset[2400:3200]
            self.val = dataset[3200:]
            self.data_all = dataset
        elif self.name == 'Lipo':
            self.train = dataset[:2400]
            self.test = dataset[2400:3200]
            self.val = dataset[3200:]
            self.data_all = dataset
            ############################################Name
        elif self.name == 'ZINC':   
            self.train = dataset[:10000]
            self.test = dataset[10000:11000]
            self.val = dataset[11000:]
 
            self.data_all = dataset
        elif self.name == 'MUV':           # 93,087    ~24.2  ~52.6    9   17
            self.train = dataset[:80000]
            self.test = dataset[80000 :90000]
            self.val = dataset[90000 :]
            self.data_all = dataset
        elif self.name == 'QM9':       # 130830
            self.train = dataset[:40000]
            self.test = dataset[40000: 50000]
            self.val  = dataset[50000: 60000]
            self.data_all = dataset
        elif self.name == 'ogbg-molpcba': # Graphs 437,929   #Tasks 128
            self.train = dataset[:240000]
            self.test = dataset[240000:320000]
            self.val = dataset[320000: ]
            self.data_all = dataset
        elif self.name == 'ogbg-molhiv':     #Graphs 39386  41,127	    #Tasks  1
            self.train = dataset[:32000]
            self.test = dataset[32000:36000 ]
            self.val = dataset[36000:]
            self.data_all = dataset
        elif self.name == 'BENZENE':       #   BENZENE		  
            self.train = dataset[:8000]
            self.test= dataset[8000: 10000  ]
            self.val  = dataset[10000: ]
 
            self.data_all = dataset   
        elif self.name == 'Fluoride_carbonyl': # Fluoride_carbonyl
            self.train = dataset[:1200]
            self.test= dataset[1200:  ]
            self.val  = dataset[1200: ]
 
            self.data_all = dataset   
        elif self.name == 'Alkane_carbonyl': # Alkane_carbonyl        A4327
            self.train = dataset[:1200]
            self.test= dataset[1200:  ]
            self.val  = dataset[1200: ]
 
            self.data_all = dataset  

        if self.name != 'pre_training':
            print('train, val test,  sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))

    def collate(self, samples):
 
        graphs, labels, subgraphs , trans_logM = map(list, zip(*samples))
 
        labels = torch.stack(labels)
 
        batched_graph = dgl.batch(graphs)
 
        return batched_graph, labels, subgraphs, trans_logM
