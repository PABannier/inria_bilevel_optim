import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    data_path = Path("data/train/X.csv.gz")
    X = pd.read_csv(data_path, compression="gzip")

    subj_idx = np.unique(X["subject"])[0]
    print(subj_idx)

    X_used = X[X["subject"] == subj_idx]
    X_used = X_used.drop(["subject"], axis=1)

    stds = np.empty(X.shape[1])

    for idx in range(len(stds)):
        x = X_used.iloc[idx].values
        stds[idx] = np.std(x[np.abs(x) < np.median(np.abs(x))])

    print(np.mean(stds))
