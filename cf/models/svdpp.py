from ..utils.transform import sparse_to_dense

import numpy as np
import numba as nb


@nb.jit
def update_factor(err, Wu, Wi, i, j, reg_w, lr, implicit):
    u, v = Wu[i], Wi[j]
    Wu[i] += lr * (err * v - reg_w * u)
    Wi[j] += lr * (err * (u + implicit) - reg_w * v)

@nb.jit
def update_bias(err, Bu, Bi, i, j, reg_b, lr):
    Bu[i] += lr * (err - reg_b * Bu[i])
    Bi[j] += lr * (err - reg_b * Bi[j])

@nb.jit
def update_implicit_factor(err, v, reg_w, Y, Ru, Ru_norm, lr):
    for j in Ru:
        Y[j] += lr * (err * Ru_norm * v - reg_w * Y[j])


class SVDpp:
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

        self.Y = np.zeros((conf["num_item"],
                           conf["num_dim"]))

    def fit(self, dev_set, val_set, num_epoch):
        R = list(map(lambda x: np.where(~np.isnan(x))[0],
                     sparse_to_dense(dev_set,
                                     len(self.b_u),
                                     len(self.b_i))))
        for epoch in range(num_epoch):
            tot_dev = 0
            for (_, i, j, r) in dev_set.itertuples():
                Ru = R[i]
                Ru_norm = np.power(len(Ru), -0.5)
                Ysum = self.Y[Ru].sum(axis=0)
                implicit_term = Ru_norm * Ysum

                err = r - self.predict(i, j, implicit_term)
                tot_dev += err * err

                update_factor(err, self.W_u, self.W_i, 
                              i, j, self.reg_w, self.lr,
                              implicit_term)
                update_bias(err, self.b_u, self.b_i, 
                            i, j, self.reg_b, self.lr)
                update_implicit_factor(err, self.W_i[j], 
                        self.reg_w, self.Y, Ru, Ru_norm,
                        self.lr)
            self.lr = max(self.lr*0.97, 1e-4)

            tot_dev = np.sqrt(tot_dev / len(dev_set))
            tot_val = 0
            for (_, i, j, r) in val_set.itertuples():
                Ru = R[i]
                Ru_norm = np.power(len(Ru), -0.5)
                Ysum = self.Y[Ru].sum(axis=0)
                implicit_term = Ru_norm * Ysum

                err = r - self.predict(i, j, implicit_term)
                tot_val += err * err
            tot_val = np.sqrt(tot_val / len(val_set))
            print(f"Epoch {epoch}: dev = {tot_dev}, val = {tot_val}")

    def predict(self, row, col, implicit_term):
        return np.dot(self.W_u[row] + implicit_term,
                      self.W_i[col]) + self.b_u[row] + self.b_i[col]


    def predict_all(self, R):
        Ysum = np.asarray(list(map(lambda Ru: self.Y[Ru].sum(axis=0), R)))
        coeff = np.asarray(list(map(lambda Ru: np.power(len(Ru), -0.5), R)))
        return np.dot(self.W_u + coeff[:, np.newaxis] * Ysum, self.W_i.T) + self.b_i.T + self.b_u
