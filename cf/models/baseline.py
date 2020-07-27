from ..utils.transform import sparse_to_dense

import numpy as np
import numba as nb


@nb.jit()
def step(err, b_u, b_i, i, j, lr, reg_b):
    b_u[i] += lr * (err - reg_b * b_u[i])
    b_i[j] += lr * (err - reg_b * b_i[j])


class BaselinePredictor:
    def __init__(self, conf):
        self.b_i = np.random.rand(conf["num_item"])
        self.b_u = np.random.rand(conf["num_user"])

        self.reg_b = conf["reg_b"]
        self.lr = conf["lr"]

    def fit(self, dev_set, val_set, num_epoch):
        X_dev = sparse_to_dense(dev_set, len(self.b_u), len(self.b_i))
        X_val = sparse_to_dense(val_set, len(self.b_u), len(self.b_i))
        dev_mask = ~np.isnan(X_dev)
        val_mask = ~np.isnan(X_val)
        for epoch in range(num_epoch):
            tot_dev = 0
            for (_, i, j, r) in dev_set.itertuples():
                err = (r - self.predict(i, j))
                tot_dev += err * err
                step(err, self.b_u, self.b_i, i, j, self.lr, self.reg_b)

            self.lr = max(self.lr*0.97, 1e-4)
            predict = self.predict_matrix()
            tot_dev = np.sqrt(np.square(X_dev - predict)[dev_mask].sum() / dev_mask.sum())
            tot_val = np.sqrt(np.square(X_val - predict)[val_mask].sum() / val_mask.sum())
            print(f"Epoch {epoch}: dev = {tot_dev}, val = {tot_val}")

    def predict(self, row, col):
        return self.b_u[row] + self.b_i[col]

    def predict_matrix(self):
        return self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
