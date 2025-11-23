# Multi-View Conditional Information Bottleneck <br> (S-CGIB)
Multi-View Conditional Information Bottleneck (MVCIB) is a novel architecture for pre-training Graph Neural Networks in 2D and 3D molecular structures and developed by [NS Lab, CUK](https://nslab-cuk.github.io/) based on pure [PyTorch](https://github.com/pytorch/pytorch) backend. 

<p align=center>
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.8.8-3776AB?logo=python&style=flat-square" alt="Python">
  </a>    
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.4-FF6F00?logo=pytorch&style=flat-square" alt="pytorch">
  </a>    
  <img src="https://custom-icon-badges.demolab.com/github/last-commit/NSLab-CUK/S-CGIB?logo=history&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/languages/code-size/NSLab-CUK/S-CGIB?logo=file-code&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/issues-pr-closed/NSLab-CUK/S-CGIB?color=purple&logo=git-pull-request&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/v/tag/NSLab-CUK/S-CGIB?logo=tag&logoColor=white&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/stars/NSLab-CUK/S-CGIB?logo=star&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/issues-raw/NSLab-CUK/S-CGIB?logo=issue&style=flat-square"/>
  <img src="https://custom-icon-badges.demolab.com/github/license/NSLab-CUK/S-CGIB?logo=law&style=flat-square"/>
</p>

<br>


## 1. Overview

We aim to build a pre-trained Graph Neural Network (GNN) model on 2D and 3D molecular structures. Recent pre-training strategies for molecular graphs have attempted to use 2D and 3D molecular views as both inputs and self-supervised signals, primarily aligning graph-level representations. However, existing studies remain limited in addressing two main challenges of multi-view molecular learning: (1) discovering shared information between two views while diminishing view-specific information and (2) identifying and aligning important substructures, e.g., functional groups, which are crucial for enhancing cross-view consistency and model expressiveness. To solve these challenges, we propose a Multi-View Conditional Information Bottleneck framework, called MVCIB, for pre-training graph neural networks on 2D and 3D molecular structures in a self-supervised setting. Our idea is to discover the shared information while minimizing irrelevant features from each view under the MVCIB principle, which uses one view as a contextual condition to guide the representation learning of its counterpart. To enhance semantic and structural consistency across views, we utilize key substructures, e.g., functional groups and ego-networks, as anchors between the two views. Then, we propose a cross-attention mechanism that captures fine-grained correlations between the substructures to achieve subgraph alignment across views. Extensive experiments in four molecular domains demonstrated that MVCIB consistently outperforms baselines in both predictive performance and interpretability. Moreover, MVCIB achieved the 3d Weisfeiler-Lehman expressiveness power to distinguish not only non-isomorphic graphs but also different 3D geometries that share identical 2D connectivity, such as isomers.

<br>

<p align="center">
  <img src="./Figures/model_architecture.jpg" alt="Graph Transformer Architecture" width="800">
  <br>
  <b></b> The overall architecture of MVCIB.
</p>


## 2. Reproducibility

### Datasets 

We conducted experiments across four different chemical domains: Physiology (BBBP, Tox21, ToxCast,SIDER, ClinTox, and MUV), Physical Chemistry (ESOL, FreeSolv, and Lipo), Biophysics (mol-HIV and BACE), Quantum Mechanics (QM9).
For the pre-training dataset, we considered unlabeled molecules from the ChEMBL database.


### Requirements and Environment Setup

The source code was developed in Python 3.8.8. MVCIB is built using Torch-geometric 2.3.1 and DGL 1.1.0. Please refer to the official websites for installation and setup.
All the requirements are included in the ```environment.yml``` file.

```
# Conda installation

# Install python environment

conda env create -f environment.yml 
```
The source code contains both Self-Supervised Pre-training and Fine-tuning phases. 
We also provide our pre-trained model, named pre_trained_GIN_300_4_2.pt, in the folder outputs/. 
The pre-processed data for each dataset is stored in pts/ folder, which contains inputs prepared for direct use in the model.

### Self-supervised pre-training

#### pre-training
```
# Use the following command to run the pretrain task, the output will generate the pre-trained files in the folder outputs/.
python exp_pretraining.py --encoder GIN --k_transition 2 --device cuda:0
```

### Hyperparameters

The following options can be passed to the below commands for fine-tuning the model:

```--encoder:``` The graph encoder. For example: ```--encoder GIN```

```--lr:``` Learning rate for fine-tuning the model. For example: ```--lr 0.001```

```--dims:``` The dimension of hidden vectors. For example: ```--dims 64```

```--num_layers:``` Number of layers for model training. For example: ```--num_layers 5``` 

```--k_transition:``` The size of 2D molecular subgraphs. For example: ```--k_transition 3```

```--angstrom:``` The size of 3D molecular subgraphs. For example: ```--angstrom 1.5```

```--pretrained_ds:``` The file name of the pre-trained model. For example: ```--pretrained_ds pre_trained```

```--ft_epoches:```Number of epochs for fine-tuning the pre-trained model. For example: ```--ft_epoches 50```.

```--batch_size:``` The size of a batch. For example: ```--batch_size 128```.

```--device:``` The GPU id. For example: ```--device 0```.

### How to fine-tune MVCIB on downstream datasets

The following commands will run the fine-tuning the **MVCIB** on different datasets.
The model performance will be sent to the command console.

#### For BBBP and BACE Datasets
```
python exp_moleculenetBACE_BBBP.py  
``` 
#### For FreeSolv,  ESOL, and Lipo Datasets
```
python exp_molsolv.py  --dataset ESOL 
``` 
#### For MUV, SIDER, Tox21, ClinTox, and ToxCast Datasets
```
python exp_moleculeSTCT.py  --dataset Tox21  
``` 
#### For ogbg-molhiv Dataset
```
python exp_molhiv.py   
``` 

#### For the QM9 Dataset with different properties (Table 2)
```
python exp_molqm9.py  --target_index 0 

# where the target_index presents which target property in the QM9 dataset to predict, indexed from 0 to 6 as shown in Table 2.
``` 




## 3. Reference

:page_with_curl: Paper [on arXiv](https://arxiv.org/): 
* [![arXiv](https://img.shields.io/badge/arXiv-2412.15589-b31b1b?style=flat-square&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2412.15589) 

:pencil: Blog post [on Network Science Lab](https://nslab-cuk.github.io/2024/12/19/SCGIB/): 
* [![Web](https://img.shields.io/badge/NS@CUK-Post-0C2E86?style=flat-square&logo=jekyll&logoColor=FFFFFF)](https://nslab-cuk.github.io/2024/12/19/SCGIB/)


## 4. Citing MVCIB

Please cite our [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33891) if you find *MVCIB* useful in your work:
```
@misc{hoang2024pretraininggraphneuralnetworks,
      title={Pre-training Graph Neural Networks on Molecules by Using Subgraph-Conditioned Graph Information Bottleneck}, 
      author={Van Thuy Hoang and O-Joun Lee},
      year={2024},
      eprint={2412.15589},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.15589}, 
}
```

Please take a look at our unified graph transformer model, [**UGT**](https://github.com/NSLab-CUK/Unified-Graph-Transformer), which can preserve local and globl graph structure, and community-aware graph transformer model, [**CGT**](https://github.com/NSLab-CUK/Community-aware-Graph-Transformer), which can mitigate degree bias problem of message passing mechanism, together. 


## 5. Contributors

<a href="https://github.com/NSLab-CUK/S-CGIB/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NSLab-CUK/S-CGIB" />
</a>



<br>

***

<a href="https://nslab-cuk.github.io/"><img src="https://github.com/NSLab-CUK/NSLab-CUK/raw/main/Logo_Dual_Wide.png"/></a>

***

