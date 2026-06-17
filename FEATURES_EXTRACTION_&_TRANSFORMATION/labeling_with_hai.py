import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("final_data_imputed.csv")

df["Fold_Change_HAI"] = (
    df["hai_peak"]
    /
    (df["hai_baseline"] + 1e-8)
)

df["Label_HAI"] = (
    df["Fold_Change_HAI"] >= 4
).astype(int)


df["Age Reported"] = pd.to_numeric(df["Age Reported"], errors="coerce")

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"].astype(str))

df["Virus"] = df["Virus"].str.replace("Wisonsin", "Wisconsin")
df = pd.get_dummies(df, columns=["Virus"], dtype=int)

df = df.drop(columns=['hai_baseline', 'hai_peak', 'Fold_Change_HAI'], errors='ignore')

df.to_csv("./final_data/dataset_virus_encoded.csv", index=False)
