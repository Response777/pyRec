""" SVD based matrix completion
1. Impute with row/col mean, and then
2. do SVD
"""
from ..utils.transform import sparse_to_dense

import numpy as np
import numba as nb
from scipy import linalg

def shrink(A, rank):
    U, S, V = linalg.svd(A, full_matrices=False)
    S[rank:] = 0
    return np.linalg.multi_dot([U, np.diag(S), V])


class IterSVD:
    def __init__(self, conf):
        self.num_user = conf["num_user"]
        self.num_item = conf["num_item"]
        self.rank = conf["rank"]

    def fit(self, dev_set, val_set, num_epoch):
        X_dev = sparse_to_dense(dev_set, self.num_user, self.num_item)
        X_val = sparse_to_dense(val_set, self.num_user, self.num_item)
        dev_mask = ~np.isnan(X_dev)
        val_mask = ~np.isnan(X_val)

        col_mean = np.nanmean(X_dev, axis=0)
        indices = np.where(np.isnan(X_dev))
        X_dev[indices] = np.take(col_mean, indices[1])
        for i in range(num_epoch):
            X_dev[~dev_mask] = shrink(X_dev, self.rank)[~dev_mask]
            tot_val = np.sqrt(np.square(X_val - X_dev)[val_mask].sum() / val_mask.sum())
            print(f"Epoch {i}: val = {tot_val:.5f}")
        self.P = X_dev

    def predict(self, row, col):
        return self.P[row, col]

    def predict_matrix(self):
        return self.P

