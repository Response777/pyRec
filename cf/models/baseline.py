from ..utils.transform import sparse_to_dense

import numpy as np
import numba as nb
import pandas as pd


@nb.jit()
def step(err, b_u, b_i, i, j, lr, reg_b):
    b_u[i] += lr * (err - reg_b * b_u[i])
    b_i[j] += lr * (err - reg_b * b_i[j])

class BaselinePredictor:
    def __init__(self, conf):
        self.b_u = np.random.rand(conf["num_user"])
        self.b_i = np.random.rand(conf["num_item"])

        self.reg_b = conf["reg_b"]
        self.lr = conf["lr"]

    def build_val_callback(self, val_set):
        if val_set is None:
            return lambda *args: None
        X_val = sparse_to_dense(val_set, len(self.b_u), len(self.b_i))
        val_mask = ~np.isnan(X_val)
        def score(predict):
            return np.sqrt(np.square(X_val - predict)[val_mask].sum() / val_mask.sum())
        return score

    def build_iterator(self, dataset):
        if isinstance(dataset, pd.DataFrame):
            def df_iter():
                for (_, i, j, r) in dataset.itertuples():
                    yield (i, j, r)
            return df_iter
        elif isinstance(dataset, np.ndarray):
            indices = np.where(~np.isnan(dataset))
            def np_iter():
                for (i, j) in zip(*indices):
                    yield (i, j, dataset[i, j])
            return np_iter
        else:
            raise NotImplementedError

    def fit(self, dev_set, val_set, num_epoch):
        val_callback = self.build_val_callback(val_set)
        iterator = self.build_iterator(dev_set)
        for epoch in range(num_epoch):
            tot_dev = 0
            cnt = 0
            for (i, j, r) in iterator():
                err = (r - self.predict(i, j))
                tot_dev += err * err
                cnt += 1
                step(err, self.b_u, self.b_i, i, j, self.lr, self.reg_b)

            self.lr = max(self.lr*0.97, 1e-4)
            predict = self.predict_matrix()
            tot_dev = np.sqrt(tot_dev / cnt)
            tot_val = val_callback(predict)
            print(f"Epoch {epoch}: dev = {tot_dev}, val = {tot_val}")

    def predict(self, row, col):
        return self.b_u[row] + self.b_i[col]

    def predict_matrix(self):
        return self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :]
