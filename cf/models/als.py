import numpy as np
import numba as nb
from ..utils.transform import sparse_to_dense

@nb.jit
def step_U(i, U, V_precomp, mask, b_precomp, lambda1):
    k = U.shape[1]
    A = V_precomp[mask[i], :, :].sum(axis=0) + lambda1 * mask[i].sum() * np.eye(k)
    U[i] = np.linalg.solve(A, b_precomp[i])


@nb.jit
def step_V(j, V, U_precomp, mask, b_precomp, lambda1):
    k = V.shape[1]
    A = U_precomp[mask[:, j], :, :].sum(axis=0) + lambda1 * mask[:, j].sum() * np.eye(k)
    V[j] = np.linalg.solve(A, b_precomp[j])


class ALS:
    def __init__(self, conf):
        self.W_i = np.random.rand(conf["num_item"],
                                  conf["num_dim"])
        self.W_u = np.random.rand(conf["num_user"],
                                  conf["num_dim"])
        self.reg_w = conf["reg_w"]

    def fit(self, dev_set, val_set, num_epoch):
        X_dev = sparse_to_dense(dev_set, len(self.W_u), len(self.W_i))
        X_val = sparse_to_dense(val_set, len(self.W_u), len(self.W_i))
        dev_mask = ~np.isnan(X_dev)
        val_mask = ~np.isnan(X_val)
        X_dev[~dev_mask] = 0.0
        X_val[~val_mask] = 0.0
        for epoch in range(num_epoch):
            V_precomp = np.matmul(
                self.W_i[:, :, np.newaxis],
                self.W_i[:, np.newaxis, :]
            )
            b_precomp = np.matmul(X_dev, self.W_i)
            for i in range(len(self.W_u)):
                step_U(i, self.W_u, V_precomp, dev_mask,
                       b_precomp, self.reg_w)

            U_precomp = np.matmul(
                self.W_u[:, :, np.newaxis],
                self.W_u[:, np.newaxis, :]
            )
            b_precomp = np.matmul(X_dev.T, self.W_u)
            for j in range(len(self.W_i)):
                step_V(j, self.W_i, U_precomp, dev_mask,
                       b_precomp, self.reg_w)

            predict = self.predict_matrix()
            tot_dev = np.sqrt(np.square(X_dev - predict)[dev_mask].sum() / dev_mask.sum())
            tot_val = np.sqrt(np.square(X_val - predict)[val_mask].sum() / val_mask.sum())
            print(f"Epoch {epoch}: dev = {tot_dev:.5f}, test = {tot_val:.5f}")

    def predict_matrix(self):
        return np.matmul(self.W_u, self.W_i.T)
