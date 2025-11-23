import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from itertools import chain
import copy, torch, dgl
from rdkit import Chem
from rdkit.Chem import AllChem


import torch
from torch.utils.data import Dataset
# from mol_tree import MolTree
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Disable warnings
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')

def sanitize(mol):
	try:
		smiles = get_smiles(mol)
		mol = get_mol(smiles)
	except Exception as e:
		return None
	return mol
def copy_atom(atom):
	new_atom = Chem.Atom(atom.GetSymbol())
	new_atom.SetFormalCharge(atom.GetFormalCharge())
	new_atom.SetAtomMapNum(atom.GetAtomMapNum())
	return new_atom
def copy_edit_mol(mol):
	new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
	for atom in mol.GetAtoms():
		new_atom = copy_atom(atom)
		new_mol.AddAtom(new_atom)
	for bond in mol.GetBonds():
		a1 = bond.GetBeginAtom().GetIdx()
		a2 = bond.GetEndAtom().GetIdx()
		bt = bond.GetBondType()
		new_mol.AddBond(a1, a2, bt)
	return new_mol

def get_clique_mol(mol, atoms):
	# get the fragment of clique
	smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
	new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
	new_mol = copy_edit_mol(new_mol).GetMol()
	new_mol = sanitize(new_mol)  # We assume this is not None
	return new_mol
 
def get_mol(smiles):
	mol = Chem.MolFromSmiles(smiles)
	if mol is None:
		return None
	# Chem.Kekulize(mol)
	return mol
def motif_decomp(mol):
	n_atoms = mol.GetNumAtoms()
	#print(f"n_atoms: {n_atoms}");
	if n_atoms == 1:
		return [[0]]

	cliques = []  
	breaks = []
	for bond in mol.GetBonds():
		a1 = bond.GetBeginAtom().GetIdx()
		a2 = bond.GetEndAtom().GetIdx()
		cliques.append([a1, a2])  
   
	res = list(BRICS.FindBRICSBonds(mol)) 
	
	if len(res) != 0:
		for bond in res:
			if [bond[0][0], bond[0][1]] in cliques:
				cliques.remove([bond[0][0], bond[0][1]])
			else:
				cliques.remove([bond[0][1], bond[0][0]])
			cliques.append([bond[0][0]])
			cliques.append([bond[0][1]]) 
	pre_cliques =  cliques

	# merge cliques
	for c in range(len(cliques) - 1):
		if c >= len(cliques):
			break
		for k in range(c + 1, len(cliques)):
			if k >= len(cliques):
				break
			if len(set(cliques[c]) & set(cliques[k])) > 0: 
				cliques[c] = list(set(cliques[c]) | set(cliques[k]))
				cliques[k] = []
		cliques = [c for c in cliques if len(c) > 0]
	cliques = [c for c in cliques if n_atoms> len(c) > 0]

	#num_cli = len(cliques)
	if len(cliques) ==0:
		cliques = pre_cliques
	return cliques
 


def mol_to_graph_data_obj_simple(mol):
	"""
	Converts rdkit mol object to graph Data object required by the pytorch
	geometric package. NB: Uses simplified atom and bond features, and represent
	as indices
	:param mol: rdkit mol object
	:return: graph data object with the attributes: x, edge_index, edge_attr
	"""
	# atoms
	num_atom_features = 2   # atom type,  chirality tag
	atom_features_list = []
	for atom in mol.GetAtoms():
		atom_feature = [allowable_features['possible_atomic_num_list'].index(
			atom.GetAtomicNum())] + [allowable_features[
			'possible_chirality_list'].index(atom.GetChiralTag())]
		atom_features_list.append(atom_feature)
	x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
 
	return x
 
allowable_features = {
	'possible_atomic_num_list' : list(range(1, 119)),
	'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
	'possible_chirality_list' : [
		Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
		Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
		Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
		Chem.rdchem.ChiralType.CHI_OTHER
	],
	'possible_hybridization_list' : [
		Chem.rdchem.HybridizationType.S,
		Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
		Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
		Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
	],
	'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
	'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
	'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
	'possible_bonds' : [
		Chem.rdchem.BondType.SINGLE,
		Chem.rdchem.BondType.DOUBLE,
		Chem.rdchem.BondType.TRIPLE,
		Chem.rdchem.BondType.AROMATIC
	],
	'possible_bond_dirs' : [ # only for double bond stereo information
		Chem.rdchem.BondDir.NONE,
		Chem.rdchem.BondDir.ENDUPRIGHT,
		Chem.rdchem.BondDir.ENDDOWNRIGHT
	]
}
def getDist(dgl_g, coords_tensor):
	Adj = dgl_g.adj(); Adj = Adj.to_dense()	
	num_nodes = dgl_g.num_nodes()
	D = torch.zeros((num_nodes, 100))
	for src in range(num_nodes):
		for dst in range(num_nodes):
			if Adj[src, dst] > 0:
				D[src][dst] = torch.norm(coords_tensor[src] - coords_tensor[dst] )
	return D
def load_dgl_benzene(nx_g, smile ):
	# print('loading dgl...')
	# count = 0
	edge_idx1 = [] ; edge_idx2 = []
	for e in nx_g.edges:
		edge_idx1.append(e[0])
		edge_idx2.append(e[1])
	g = dgl.graph((edge_idx1, edge_idx2))
	g = dgl.to_bidirected(g)

	rdkit_mol = AllChem.MolFromSmiles(smile) 
	g.ndata['x'] = mol_to_graph_data_obj_simple(rdkit_mol)
	g.ndata['c'] = get_3D(rdkit_mol)
	d = getDist(g, g.ndata['c'])
	g.ndata['d'] = d
	noise = torch.randn(g.ndata['c'].size())
	g.ndata['n'] = noise
 
	return g, d

 

def GetProbTranMat(Ak, num_node):
	num_node, num_node2 = Ak.shape
	if (num_node != num_node2):
		print('M must be a square matrix!')
	Ak_sum = np.sum(Ak, axis=0).reshape(1, -1)
	Ak_sum = np.repeat(Ak_sum, num_node, axis=0)
	log  = np.log(1. / num_node)
	probTranMat = np.log(np.divide(Ak, Ak_sum)) - log
	probTranMat[probTranMat < 0] = 0;  # set zero for negative and -inf elements
	probTranMat[np.isnan(probTranMat)] = 0;  # set zero for nan elements (the isolated nodes)
	return probTranMat


def getM_logM(dgl_g, kstep=3):
	tran_M = []
	tran_logM = []
 
	Adj = dgl_g.adj()
	Adj = Adj.to_dense()	
	num_nodes = dgl_g.num_nodes()
	Ak = np.matrix(np.identity(num_nodes))
	for i in range(kstep):
		Ak = np.dot(Ak, Adj)
		tran_M.append(Ak)
		probTranMat = GetProbTranMat(Ak, num_nodes)
		tran_logM.append(probTranMat)
	return tran_M, tran_logM


def get_distance(deg_A, deg_B):
	damp = 1 / (deg_A * deg_B)  # -1
	return damp   

def get_B_sim_phi(nx_g, tran_M, num_nodes, n_class, X, kstep=5):
 
	count = 0
	B = np.zeros((num_nodes, num_nodes)) #= np.zeros((num_nodes, num_nodes))
	colour = np.zeros((num_nodes, num_nodes))
	phi = np.zeros((num_nodes, num_nodes, 1))
	sim = np.zeros((num_nodes, num_nodes, kstep))

	trans_check = tran_M[kstep - 1]
	not_adj = tran_M[0]

	kmeans = KMeans(n_clusters= 2 , init='k-means++', max_iter=10, n_init=10, random_state=0)

	y_kmeans = kmeans.fit_predict(X)
	count = 0
	count_1 = 0
	for src in nx_g.nodes():
 
		for dst in nx_g.nodes():

			if src == dst:
				continue

			if not_adj[src, dst] > 0:
				continue

			if colour[src, dst] == 1 or colour[src, dst] == 1:
				continue
			if trans_check[src, dst] > 0.001:
 

				src_d = nx_g.degree(src)
				dst_d = nx_g.degree(dst)

				if np.abs(src_d - dst_d) > 1:
					continue

				if y_kmeans[src] != y_kmeans[dst]:
					continue
				else:
					count_1 += 1
					d = get_distance(src_d, dst_d)
					# B i, j
					B[src, dst] = d
					B[dst, src] = d
					# phi i,j
					if phi[src, dst] == 0:
						phi[src, dst] = d
						phi[dst, src] = d

			colour[src, dst] = 1
			colour[dst, src] = 1
		B[src, src] = 0
		count += 1
 

	sim = compute_sim(tran_M, num_nodes, k_step=kstep)

	return B, sim, phi


def compute_sim(tran_M, num_nodes, k_step=5):
	sim = np.zeros((num_nodes, num_nodes, k_step))
	trans_check = tran_M[k_step - 1]

	for step in range(k_step):
 
		colour = np.zeros((num_nodes, num_nodes))
		trans_k = copy.deepcopy(tran_M[step])
		trans_k[trans_k >= 0.001] = 1
		trans_k[trans_k < 0.001] = 0;
		trans_k = np.array(trans_k)

		row_sums = trans_k.sum(axis=1)
		trans_mul = trans_k @ trans_k.T
		for i in range(num_nodes):

			for j in range(i + 1, num_nodes):
				if trans_check[i, j] < 0.0001:
					continue
				if colour[i, j] == 1 or colour[j, i] == 1:
					continue
				score = np.round(trans_mul[i, j] / (row_sums[i] + row_sums[j] - trans_mul[i, j]), 4)
				if score < 0.001:
					score = 0
				sim[i, j, step] = score
				sim[j, i, step] = score

				colour[i, j] = 1
				colour[j, i] = 1
	return sim


def get_A_D(nx_g, num_nodes):
	num_edges = nx_g.number_of_edges()
	# d= np.zeros((num_nodes, num_nodes))
	d = np.zeros((num_nodes))

	Adj = np.zeros((num_nodes, num_nodes))

	for src in nx_g.nodes():
		src_degree = nx_g.degree(src)
		d[src] = src_degree
		for dst in nx_g.nodes():
 
			if nx_g.has_edge(src, dst):
				Adj[src][dst] = 1
 
	return Adj, d, num_edges


def load_dgl(nx_g, x ):
	#print('loading dgl...')
	edge_idx1 = []
	edge_idx2 = []
	for e in nx_g.edges:
		edge_idx1.append(e[0])
		edge_idx2.append(e[1])
 

	g = dgl.graph((edge_idx1, edge_idx2))
	g = dgl.to_bidirected(g)
 
	g.ndata['x'] = x
 
	return g

# def get_3D(mol):
# 	# if AllChem.EmbedMolecule(mol) != 0:
# 	# 	print("3D embedding failed for the molecule")
# 	AllChem.UFFOptimizeMolecule(mol)
# 	conformer = mol.GetConformer()
# 	coords = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
# 	coords_tensor = torch.tensor(coords, dtype=torch.float32)  # Node feature tensor
# 	return coords_tensor
def get_3D(mol):
	if AllChem.EmbedMolecule(mol) != 0:
		#print("3D embedding failed for the molecule")
		raise SystemExit()
	AllChem.UFFOptimizeMolecule(mol)
	conformer = mol.GetConformer()
	coords = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
	coords_tensor = torch.tensor(coords, dtype=torch.float32)  # Node feature tensor
 
	return coords_tensor

def load_dgl_fromPyG(data):
 
	edge_idx1 = []
	edge_idx2 = []
	edge_index = data.edge_index
	edge_idx1 =edge_index[0]
	edge_idx2 =edge_index[1]
 

	g = dgl.graph((edge_idx1, edge_idx2))
	g = dgl.to_bidirected(g)
 
	g.ndata['x'] = data.x
 
	return g


def load_dgl_fromPyG_pcqm4mv2(data):
 
	arr = np.asarray(data)
 
	label = data[1]
	x = arr[0]['node_feat']
	x = torch.tensor(x)
	edge_index = arr[0]['edge_index']
 

	edge_idx1 = []
	edge_idx2 = []
	edge_index = edge_index
	edge_idx1 = edge_index[0]
	edge_idx2 = edge_index[1]
 

	g = dgl.graph((edge_idx1, edge_idx2))
	g = dgl.to_bidirected(g)
 
	g.ndata['x'] = x
 
	return g

