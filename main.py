import os
import json
import numba as nb
import numpy as np
import pandas as pd

import cf.models
from cf.utils import flags
FLAGS = flags.FLAGS

# define the parameters
flags.DEFINE_bool("submit", False, "")
flags.DEFINE_string("conf", "confs/svd-sgd.json", "config file")

def generate_prediction(model, dataset, mean, path):
    dataset = dataset.copy()
    pred = np.zeros(len(dataset))
    for (idx, i, j, r) in dataset.itertuples():
        pred[idx] = model.predict(i, j) + mean
    dataset.Prediction = pred
    dataset.to_csv(path, index=False)

def generate_submission(model, mean, path):
    sample = pd.read_csv("submissions/sampleSubmission.csv")
    pred = np.zeros(len(sample))
    for (i, idx) in enumerate(sample['Id']):
        r, c = idx.split('_')
        r, c = (int(r[1:]) - 1, int(c[1:]) - 1)
        pred[i] = model.predict(r, c) + mean
    sample.Prediction = pred
    sample.to_csv(path, index=False)

if __name__ == '__main__':
    if not FLAGS.submit:
        dev_set = pd.read_csv("datasets/dev.csv")
        val_set = pd.read_csv("datasets/val.csv")
    else:
        dev_set = pd.read_csv("datasets/raw.csv")
        val_set = pd.read_csv("datasets/val.csv")

    mean = dev_set.Prediction.mean()
    dev_set.Prediction = (dev_set.Prediction - mean)
    val_set.Prediction = (val_set.Prediction - mean)

    with open(FLAGS.conf, "r") as f:
        conf = json.load(f)
    model = getattr(cf.models, conf["model"])(conf)
    model.fit(dev_set, val_set, conf["num_epoch"])

    if FLAGS.submit:
        generate_submission(model, mean, f"predictions/submit-{conf['name']}.csv")
    else:
        generate_prediction(model, val_set, mean, f"predictions/val-{conf['name']}.csv")
