import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

"""
Evaluation functions from OGB.
https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py
"""


def Fidelity(scores, targets, pos, neg):
    num = targets.size()[0]
    targets = targets.squeeze(dim=-1)
 
    scores = scores.argmax(dim=1)

    pos = pos.argmax(dim=1)
    neg = neg.argmax(dim=1)
    
    p_ground = np.array(scores.detach().cpu().to(float) == targets.detach().cpu().to(float)).astype(int)
    p_pos = np.array(pos.detach().cpu().to(float) == targets.detach().cpu().to(float)).astype(int)
    p_neg = np.array(neg.detach().cpu().to(float) == targets.detach().cpu().to(float)).astype(int)
    acc_pos = p_pos.sum()
    acc_neg = p_neg.sum()

    neg0 = (neg ==0).sum()
    neg1 = (neg ==1).sum()
    print(f"neg0: {neg0}/{(targets ==0).sum()}| neg1: {neg1}/{(targets ==1).sum()} | all: {len(p_ground)}")


    pos_fidelity = np.abs(p_ground - p_neg ).sum()/num
    neg_fidelity = np.abs(p_ground - p_pos ).sum()/num
    
    pos_fidelity = np.round(pos_fidelity, 4); neg_fidelity = np.round(neg_fidelity, 4)
 
    
    return pos_fidelity, neg_fidelity, acc_pos, acc_neg
def eval_rocauc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''

    rocauc_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}


def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')
    return sum(ap_list) / len(ap_list)
    #return {'ap': sum(ap_list) / len(ap_list)}


def eval_rmse(y_true, y_pred):
    '''
        compute RMSE score averaged across tasks
    '''
    rmse_list = []

    for i in range(y_true.shape[1]):
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rmse_list.append(np.sqrt(
            ((y_true[is_labeled, i] - y_pred[is_labeled, i]) ** 2).mean()))

    return {'rmse': sum(rmse_list) / len(rmse_list)}


def eval_acc(y_true, y_pred):
    acc_list = []

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return {'acc': sum(acc_list) / len(acc_list)}


def eval_F1(seq_ref, seq_pred):
    # '''
    #     compute F1 score averaged over samples
    # '''

    precision_list = []
    recall_list = []
    f1_list = []

    for l, p in zip(seq_ref, seq_pred):
        label = set(l)
        prediction = set(p)
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {'precision': np.average(precision_list),
            'recall': np.average(recall_list),
            'F1': np.average(f1_list)}


def rmse(scores, targets):
    eps = 1e-6
    criterion = nn.MSELoss()
    rmse = torch.sqrt(criterion(scores, targets) + eps)
    rmse = rmse.detach().item()
    # rmse = torch.sqrt(torch.mean((scores- targets )**2))
    # #MAE = F.l1_loss(scores, targets)
    # rmse = rmse.detach().item()/ float(scores.shape[0])
    return rmse


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE


def accuracy_TU(scores, targets):
    targets = targets.squeeze(dim=-1)
    scores = scores.argmax(dim=1)
    # print(f'accuracy_TU: scores: {scores}, type: {type(scores)}')
    # print(f'accuracy_TU: targets: {targets}, type: {type(targets)}')

    acc = (scores.to(float) == targets.to(float)).sum().item()
    # print(f"accuracy_TU debug targets: {targets} ")
    # print(f"accuracy_TU debug scores: {scores} ")
    # raise SystemExit()
    # acc2 = torch.sum(scores==targets)
    # print(f'accuracy_TU: acc: {acc}')
    # print(f'accuracy_TU: acc2: {acc2}')
    return acc


# def accuracy_TU(scores, targets):
#     scores = scores.detach().cpu().argmax(dim=1)
#     acc = (scores == targets).float().sum().item()
#     return acc


def accuracy_MNIST_CIFAR(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    return acc


def accuracy_CITATION_GRAPH(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores == targets).float().sum().item()
    acc = acc / len(targets)
    return acc


def accuracy_SBM(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax(torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy(), axis=1)
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc


def binary_f1_score(scores, targets):
    """Computes the F1 score using scikit-learn for binary class labels. 

    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.argmax(dim=1).cpu().numpy()
    return f1_score(y_true, y_pred, average='binary')


def accuracy_VOC(scores, targets):
    scores = scores.detach().argmax(dim=1).cpu()
    targets = targets.cpu().detach().numpy()
    acc = f1_score(scores, targets, average='weighted')
    return acc


# clustering
from sklearn.metrics.cluster import contingency_matrix


def _compute_counts(y_true, y_pred):  # TODO(tsitsulin): add docstring pylint: disable=missing-function-docstring
    contingency = contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)
    diff_class_true = contingency.sum(axis=1) - same_class_true
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()
    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (total - 1) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, false_negatives, true_negatives


def modularity(adjacency, clusters):
    """Computes graph modularity.
    Args:
      adjacency: Input graph in terms of its sparse adjacency matrix.
      clusters: An (n,) int cluster vector.

    Returns:
      The value of graph modularity.
      https://en.wikipedia.org/wiki/Modularity_(networks)
    """
    degrees = adjacency.sum(axis=0).A1
    n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
    result = 0
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
        degrees_submatrix = degrees[cluster_indices]
        result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix) ** 2) / n_edges
    return result / n_edges


def precision(y_true, y_pred):
    true_positives, false_positives, _, _ = _compute_counts(y_true, y_pred)
    return true_positives / (true_positives + false_positives)


def recall(y_true, y_pred):
    true_positives, _, false_negatives, _ = _compute_counts(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)


def accuracy_score(y_true, y_pred):
    true_positives, false_positives, false_negatives, true_negatives = _compute_counts(
        y_true, y_pred)
    return (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)


def conductance(adjacency, clusters):  # TODO(tsitsulin): add docstring pylint: disable=missing-function-docstring
    inter = 0
    intra = 0
    cluster_idx = np.zeros(adjacency.shape[0], dtype=bool)
    for cluster_id in np.unique(clusters):
        cluster_idx[:] = 0
        cluster_idx[np.where(clusters == cluster_id)[0]] = 1
        adj_submatrix = adjacency[cluster_idx, :]
        inter += np.sum(adj_submatrix[:, cluster_idx])
        intra += np.sum(adj_submatrix[:, ~cluster_idx])
    return intra / (inter + intra)
