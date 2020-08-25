from ..utils.transform import sparse_to_dense

import numpy as np
import numba as nb


@nb.jit
def step_U(i, U, V_precomp, mask, b_precomp, reg_w):
    k = U.shape[1]
    A = V_precomp[mask[i], :, :].sum(axis=0) + reg_w * mask[i].sum() * np.eye(k)
    if mask[i].sum() == 0: return
    U[i] = np.linalg.solve(A, b_precomp[i])


@nb.jit
def step_V(j, V, U_precomp, mask, b_precomp, reg_w):
    k = V.shape[1]
    A = U_precomp[mask[:, j], :, :].sum(axis=0) + reg_w * mask[:, j].sum() * np.eye(k)
    if mask[:, j].sum() == 0: return
    V[j] = np.linalg.solve(A, b_precomp[j])


class ALS:
    def __init__(self, conf):
        self.W_i = np.random.rand(conf["num_item"],
                                  conf["num_dim"])
        self.W_u = np.random.rand(conf["num_user"],
                                  conf["num_dim"])
        self.b_i = np.random.rand(conf["num_item"])
        self.b_u = np.random.rand(conf["num_user"])

        self.reg_w = conf["reg_w"]
        self.reg_b = conf["reg_b"]

    def fit(self, dev_set, val_set, num_epoch):
        X_dev = sparse_to_dense(dev_set, len(self.W_u), len(self.W_i))
        X_val = sparse_to_dense(val_set, len(self.W_u), len(self.W_i))
        dev_mask = ~np.isnan(X_dev)
        val_mask = ~np.isnan(X_val)
        X_dev[~dev_mask] = 0.0
        for epoch in range(num_epoch):
            normalized_X = X_dev - self.b_u[:, np.newaxis] - self.b_i[np.newaxis, :]
            normalized_X[~dev_mask] = 0

            V_precomp = self.W_i[:, :, np.newaxis] @ self.W_i[:, np.newaxis, :]
            b_precomp = normalized_X @ self.W_i
            for i in range(len(self.W_u)):
                step_U(i, self.W_u, V_precomp, dev_mask, b_precomp, self.reg_w)

            U_precomp = self.W_u[:, :, np.newaxis] @ self.W_u[:, np.newaxis, :]
            b_precomp = normalized_X.T @ self.W_u
            for j in range(len(self.W_i)):
                step_V(j, self.W_i, U_precomp, dev_mask, b_precomp, self.reg_w)

            X_bu = X_dev - self.b_i[np.newaxis, :] - self.W_u @ self.W_i.T
            X_bu[~dev_mask] = 0
            self.b_u = X_bu.sum(axis=1) / (dev_mask.sum(axis=1) * (1 + self.reg_b))
            self.b_u = np.nan_to_num(self.b_u)

            X_bi = X_dev - self.b_u[:, np.newaxis] - self.W_u @ self.W_i.T
            X_bi[~dev_mask] = 0
            self.b_i = X_bi.sum(axis=0) / (dev_mask.sum(axis=0) * (1 + self.reg_b))
            self.b_i = np.nan_to_num(self.b_i)

            predict = self.predict_matrix()
            tot_dev = np.sqrt(np.square(X_dev - predict)[dev_mask].sum() / dev_mask.sum())
            tot_val = np.sqrt(np.square(X_val - predict)[val_mask].sum() / val_mask.sum())
            print(f"Epoch {epoch}: dev = {tot_dev:.5f}, test = {tot_val:.5f}")

    def predict(self, row, col):
        return np.dot(self.W_u[row], self.W_i[col]) + self.b_u[row] + self.b_i[col]

    def predict_matrix(self):
        return self.W_u @ self.W_i.T + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
