import os
import json
import numba as nb
import numpy as np
import pandas as pd

import cf.models
from cf.utils import flags
FLAGS = flags.FLAGS

# define the parameters
flags.DEFINE_string("conf", "confs/ml-100k/svd-als.json", "config file")

if __name__ == '__main__':
    dev_set = pd.read_csv("datasets/dev.csv")
    val_set = pd.read_csv("datasets/val.csv")

    mean = dev_set.Prediction.mean()
    dev_set.Prediction = (dev_set.Prediction - mean)
    val_set.Prediction = (val_set.Prediction - mean)

    with open(FLAGS.conf, "r") as f:
        conf = json.load(f)
    model = getattr(cf.models, conf["model"])(conf)

    from IPython import embed
    model.fit(dev_set, val_set, conf["num_epoch"])
