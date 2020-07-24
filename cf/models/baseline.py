from . import rms_error

import numpy as np
import numba as nb


@nb.jit()
def step(X, item_b, user_b, i, j, lr, lambda2):
    err = X[i, j] - item_b[i] - user_b[j]
    item_b[i] += lr * (err - lambda2 * item_b[i])
    user_b[j] += lr * (err - lambda2 * user_b[j])


class BaselinePredictor:
    def __init__(self, num_item, num_user, lambda2):
        self.num_item = num_item
        self.num_user = num_user
        self.lambda2 = lambda2

        self.initialize()

    def initialize(self):
        self.item_b = np.random.rand(self.num_item)
        self.user_b = np.random.rand(self.num_user)

    def fit(self, X, train_mask, test_mask, lr, num_epoch):
        indices = np.asarray(np.where(train_mask)).T
        for epoch in range(num_epoch):
            for (i, j) in indices:
                step(X, self.item_b, self.user_b, i, j, lr, self.lambda2)

            lr = max(lr*0.97, 1e-4)

            predict = self.predict()
            if test_mask is None:
                print("Epoch {}: dev err = {:.5f}".format(
                    epoch, rms_error(X, predict, train_mask)))
            else:
                print("Epoch {}: dev err = {:.5f}, test err = {:.5f}".format(
                    epoch, rms_error(X, predict, train_mask), rms_error(X, predict, test_mask)))

    def predict(self):
        return self.item_b[:, np.newaxis] + self.user_b[np.newaxis, :]
