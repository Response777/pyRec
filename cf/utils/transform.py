import numpy as np
def sparse_to_dense(df, n_rows, n_cols):
    # Pivot may miss rows / cols
    # df = df.pivot(index="row", columns="col", values="Prediction")
    a = np.full((n_rows, n_cols), np.nan)
    for (_, i, j, v) in df.itertuples(): a[i, j] = v
    return a

