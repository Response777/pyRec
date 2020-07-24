import numpy as np
import numba as nb
from ..utils.transform import sparse_to_dense


def predict(dense, sim, ind, row, col, K):
    knn = []
    for idx in ind[col]:
        if len(knn) < K and not np.isnan(dense[row, idx]):
            knn.append(idx)
    return (sim[col][knn] * dense[row, knn]).sum() / np.abs(sim[col][knn]).sum()

def sim_matrix(dense):
    sim_mat = np.dot(dense.T, dense)
    sum_mat = np.square(dense.T).sum(axis=1)
    return sim_mat / (np.dot(sum_mat[:, np.newaxis], sum_mat[np.newaxis, :]))

class KNN:
    def __init__(self, conf):
        self.num_user = conf["num_user"]
        self.num_item = conf["num_item"]
        self.K = conf["K"]

    def fit(self, dev_set, val_set, _):
        self.dense = sparse_to_dense(dev_set, self.num_user, self.num_item)
        self.sim = sim_matrix(np.nan_to_num(self.dense))
        self.ind = np.argsort(-self.sim, axis=1)
        print(f"dev err: {self.validate(dev_set)}, val err: {self.validate(val_set)}")

    def validate(self, dataset):
        tot = 0.
        for (_, i, j, r) in dataset.itertuples():
            err = (r - predict(self.dense, self.sim, self.ind, i, j, self.K))
            tot += (err * err)
        return np.sqrt(tot / len(dataset))
      
    def predict(self, row, col):
        knn = list(filter(lambda x: not np.isnan(self.dense[row, x]), self.ind[col]))
        if len(knn) > self.K:
            knn = knn[:self.K]
        return (self.sim[col][knn] * self.dense[row][knn]).sum() / np.abs(self.sim[col][knn]).sum()
       
