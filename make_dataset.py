import pandas as pd
import numpy as np

from cf.utils import flags
FLAGS = flags.FLAGS

flags.DEFINE_float("dev_ratio", 0.90, "dev/(dev+val)")

if __name__ == "__main__":
    df = pd.read_csv("datasets/raw.csv")
    np.random.seed(42)
    mask = np.random.rand(len(df)) < FLAGS.dev_ratio
    df[mask].to_csv("datasets/dev.csv", index=False)
    df[~mask].to_csv("datasets/val.csv", index=False)
