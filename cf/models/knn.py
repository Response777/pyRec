from ..utils.transform import sparse_to_dense
from ..models import BaselinePredictor

import numpy as np
import numba as nb
from scipy.sparse import csr_matrix


def cosine_sim(dense):
    dense = np.nan_to_num(dense)
    sim_mat = np.dot(dense.T, dense)
    sum_mat = np.sqrt(np.square(dense.T).sum(axis=1))
    return sim_mat / (np.dot(sum_mat[:, np.newaxis], sum_mat[np.newaxis, :]))

def pearson_sim(dense):
    return cosine_sim(dense - np.nanmean(dense, axis=1)[:, np.newaxis])
    
def baseline_sim(dense):
    model = BaselinePredictor({
        "num_user": dense.shape[0],
        "num_item": dense.shape[1],
        "reg_b": 0.10,
        "lr": 0.02
    })
    model.fit(dense, None, 3)
    return cosine_sim(dense - model.predict_matrix())

sim_metrics = {
    "cosine": cosine_sim,
    "pearson": pearson_sim,
    "baseline": baseline_sim,
}

class KNN:
    def __init__(self, conf):
        self.num_user = conf["num_user"]
        self.num_item = conf["num_item"]
        self.K = conf["K"]
        self.sim_func = sim_metrics[conf["similarity"]]

    def fit(self, dev_set, val_set, _):
        self.dense = sparse_to_dense(dev_set, self.num_user, self.num_item)
        self.sim = self.sim_func(self.dense)
        self.ind = np.argsort(-self.sim, axis=1)
        self.ind = self.ind[:, 1:] # exclude itself
        self.cache = self.dense.copy()
        self.cache[:, :] = np.nan

        mask = csr_matrix(~np.isnan(self.dense).T.astype(int))
        U = (mask * mask.T).toarray()
        self.sim *= (U - 1) / (U - 1 + 100) 
        print(f"dev err: {self.validate(dev_set)}, val err: {self.validate(val_set)}")

    def validate(self, dataset):
        tot = 0.
        for (_, i, j, r) in dataset.itertuples():
            err = (r - self.predict(i, j))
            tot += (err * err)
        return np.sqrt(tot / len(dataset))
      
    def predict(self, row, col):
        if np.isnan(self.cache[row, col]):
            rating = self.dense[row, self.ind[col]]
            weight = self.sim[col][self.ind[col]]
            weight = weight[~np.isnan(rating)]
            rating = rating[~np.isnan(rating)]
            if self.K != None and len(weight) > self.K:
                weight = weight[:self.K]
                rating = rating[:self.K]
            self.cache[row, col] = (weight * rating).sum() / np.abs(weight).sum()
        return self.cache[row, col]
      
