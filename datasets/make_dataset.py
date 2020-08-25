import pandas as pd
import numpy as np

from cf.utils import flags
FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "ml-1m", "dataset")
flags.DEFINE_float("dev_ratio", 0.80, "dev/(dev+val)")

if __name__ == "__main__":
    columns = ["row", "col", "Prediction", "timestamp"]
    if FLAGS.dataset == "ml-100k":
        df = pd.read_csv("ml-100k/u.data", header=None, sep='\t')
        df.columns = columns
        df = df.drop(columns=["timestamp"])
    elif FLAGS.dataset == "ml-1m":
        df = pd.read_csv("ml-1m/ratings.dat", header=None, sep='::')
        df.columns = columns
        df = df.drop(columns=["timestamp"])

    df.row -= 1
    df.col -= 1
    mask = np.random.rand(len(df)) < FLAGS.dev_ratio
    df[mask].to_csv("dev.csv", index=False)
    df[~mask].to_csv("val.csv", index=False)
