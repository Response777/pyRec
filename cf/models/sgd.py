from ..utils.transform import sparse_to_dense

import numpy as np
import numba as nb


@nb.jit()
def step(err, W_u, W_i, b_u, b_i, i, j, lr, reg_w, reg_b):
    u, v = W_u[i], W_i[j]

    W_u[i] += lr * (err * v - reg_w * u)
    W_i[j] += lr * (err * u - reg_w * v)

    b_u[i] += lr * (err - reg_b * b_u[i])
    b_i[j] += lr * (err - reg_b * b_i[j])


class SGD:
    def __init__(self, conf):
        self.W_i = np.random.rand(conf["num_item"],
                                  conf["num_dim"])
        self.W_u = np.random.rand(conf["num_user"],
                                  conf["num_dim"])
        self.b_i = np.random.rand(conf["num_item"])
        self.b_u = np.random.rand(conf["num_user"])

        self.reg_w = conf["reg_w"]
        self.reg_b = conf["reg_b"]
        self.lr = conf["lr"]

    def fit(self, dev_set, val_set, num_epoch):
        X_dev = sparse_to_dense(dev_set, len(self.W_u), len(self.W_i))
        X_val = sparse_to_dense(val_set, len(self.W_u), len(self.W_i))
        dev_mask = ~np.isnan(X_dev)
        val_mask = ~np.isnan(X_val)
        for epoch in range(num_epoch):
            tot_dev = 0
            for (_, i, j, r) in dev_set.itertuples():
                err = (r - self.predict(i, j))
                tot_dev += err * err
                step(err, self.W_u, self.W_i, self.b_u, self.b_i, i, j, self.lr, self.reg_w, self.reg_b)

            self.lr = max(self.lr*0.97, 1e-4)
            predict = self.predict_matrix()
            tot_dev = np.sqrt(np.square(X_dev - predict)[dev_mask].sum() / dev_mask.sum())
            tot_val = np.sqrt(np.square(X_val - predict)[val_mask].sum() / val_mask.sum())
            print(f"Epoch {epoch}: dev = {tot_dev}, val = {tot_val}")

    def predict(self, row, col):
        return np.dot(self.W_u[row], self.W_i[col]) + self.b_u[row] + self.b_i[col]

    def predict_matrix(self):
        return np.dot(self.W_u, self.W_i.T) + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
