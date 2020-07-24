import numpy as np
import numba as nb


@nb.jit()
def step(err, W_u, W_i, b_i, b_u, i, j, lr, reg_w, reg_b):
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
        for epoch in range(num_epoch):
            tot_dev = 0
            for (_, i, j, r) in dev_set.itertuples():
                err = (r - self.predict(i, j))
                tot_dev += err * err
                step(err, self.W_u, self.W_i, self.b_i, self.b_u, i, j, self.lr, self.reg_w, self.reg_b)

            self.lr = max(self.lr*0.97, 1e-4)
            tot_dev = np.sqrt(tot_dev / len(dev_set))
            tot_val = np.sqrt(sum(map(lambda x: (x[3] - self.predict(x[1], x[2]))**2, val_set.itertuples())) / len(val_set))
            print(f"Epoch {epoch}: dev = {tot_dev}, val = {tot_val}")

    def predict(self, row, col):
        return np.dot(self.W_u[row], self.W_i[col]) + self.b_u[row] + self.b_i[col]
