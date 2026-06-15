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

df = pd.read_csv("dataset_imputado_biologico.csv")

df["Fold_Change_HAI"] = (
    df["hai_peak"]
    /
    (df["hai_baseline"] + 1e-8)
)

df["Label_HAI"] = (
    df["Fold_Change_HAI"] >= 4
).astype(int)

print("\nDistribuição HAI:")
print(df["Label_HAI"].value_counts())

# 1. Garantir que a idade seja numérica
df["Age Reported"] = pd.to_numeric(df["Age Reported"], errors="coerce")

# 2. Transformar Sexo em 0 e 1
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"].astype(str))

def agrupar_cepas(nome_virus):
    nome_virus = str(nome_virus)
    
    # A/California/7/2009 é a cepa pandêmica clássica do H1N1
    if "California" in nome_virus:
        return "H1N1"
        
    # Perth e Victoria nesse seu dataset representam as cepas de H3N2
    elif "Perth" in nome_virus or "Victoria" in nome_virus:
        return "H3N2"
        
    # Qualquer vírus que comece com 'B/' ou tenha essas cidades é do Tipo B
    elif nome_virus.startswith("B/") or "Brisbane" in nome_virus or "Wisconsin" in nome_virus or "Massachusetts" in nome_virus:
        return "Tipo_B"
        
    else:
        # Fallback de segurança (caso tenha algum outro vírus não mapeado)
        return nome_virus

# Aplica a função para substituir a coluna "Virus" original pelos grupos
df["Virus"] = df["Virus"].apply(agrupar_cepas)
print(df["Virus"].unique())

df["Virus"] = df["Virus"].str.replace("Wisonsin", "Wisconsin")
# 3. Transformar Virus e Cohort usando One-Hot Encoding (Cria colunas binárias)
df = pd.get_dummies(df, columns=["Virus"], dtype=int)

df.to_csv("dataset_virus_grouped.csv", index=False)
