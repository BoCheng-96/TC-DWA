from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment

# def cluster_accuracy(y_true,
#                      y_predicted,
#                      cluster_number: Optional[int] = None):
#     """
#     Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
#     determine reassignments.
#
#     :param y_true: list of true cluster numbers, an integer array 0-indexed
#     :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
#     :param cluster_number: number of clusters, if None then calculated from input
#     :return: reassignment dictionary, clustering accuracy
#     """
#     if cluster_number is None:
#         cluster_number = (max(y_predicted.max(), y_true.max()) + 1
#                           )  # assume labels are 0-indexed
#     count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
#     for i in range(y_predicted.size):
#         count_matrix[y_predicted[i], y_true[i]] += 1
#
#     row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
#     reassignment = dict(zip(row_ind, col_ind))
#     accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
#     return reassignment, accuracy


def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    row_ind, col_ind = ind

    return None, w[row_ind, col_ind].sum() * 1.0 / y_pred.size

    # return None, sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    # row_ind, col_ind = linear_assignment(w.max() - w)
    #
    # return None, w[row_ind, col_ind].sum() * 1.0 / y_pred.size