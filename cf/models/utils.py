import numpy as np

def rms_error(X, X_, mask):
    return np.sqrt(np.square(X - X_)[mask].sum() / mask.sum())

def batch(data, batch_sz):
    length = len(data)
    for idx in range(0, length, batch_sz):
        yield data[idx:min(idx+batch_sz, length)]

def split_train_test(mask, train_ratio, seed=0):
    np.random.seed(seed)

    train_mask = np.random.rand(*mask.shape) < train_ratio
    test_mask = ~train_mask

    train_mask &= mask
    test_mask  &= mask
    return train_mask, test_mask

