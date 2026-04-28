import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados (Simulando o df_effect que você já tem)
# O seu CSV contém colunas como 'B.memory', 'CD8.GrzmB', 'NK', 'live.cd8', etc. 
df = pd.read_csv('./Dataset/Public_dt/FCS_PUBLIC_no_standard.csv')

# Selecionar apenas as colunas numéricas (features imunológicas)
features_cols = ['B.memory', 'B.plasma', 'CD8.GrzmB', 'Macro', 'Mo0', 'NK', 'cd4.treg', 'live.cd4tconv', 'live.cd8']
df_effect = df[features_cols].copy()

# --- ABORDAGEM 1: TESTE ESTATÍSTICO MULTIVARIADO (Score Composto) ---
# Em vez de testar um por um, criamos um Score Z Combinado. 
# Isso resume a "intensidade" da resposta imune em uma única label.

scaler = StandardScaler()
df_zscore = pd.DataFrame(scaler.fit_transform(df_effect), columns=features_cols)
# Exemplo hipotético (ajuste conforme a biologia da doença estudada):
# Células associadas à proteção (+) e associadas à supressão/falha (-)
colunas_protetoras = ['B.memory', 'CD8.GrzmB']
colunas_supressoras = ['cd4.treg'] # Exemplo: Tregs e Monócitos clássicos altos podem indicar pior prognóstico

# Calcula o score somando as protetoras e subtraindo as supressoras
df['Score_Biologico'] = df_zscore[colunas_protetoras].sum(axis=1) - df_zscore[colunas_supressoras].sum(axis=1)

# Agora você pode usar percentis desse score para criar a label
threshold = df['Score_Biologico'].quantile(0.75)
df['Target'] = np.where(df['Score_Biologico'] > threshold, 1, 0)

# Calculamos a média dos Z-scores de todas as células para cada participante
# Isso considera todas as features simultaneamente.
# df['Global_Response_Score'] = df_zscore.mean(axis=1)

# # Criamos a label baseada em significância (ex: acima do percentil 75 é 'Alta Proteção')
# threshold = df['Global_Response_Score'].quantile(0.75)
# df['Target'] = np.where(df['Global_Response_Score'] > threshold, 1, 0)

# # --- ABORDAGEM 2: CLUSTERING (K-MEANS) ---
# # O algoritmo agrupa os IDs (como SUB120417.404) por similaridade em todas as colunas 

# kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
# df['Cluster_Label'] = kmeans.fit_predict(df_zscore)

# # Analisar a média de cada marcador por cluster
# analise_clusters = df.groupby('Cluster_Label')[features_cols].mean()
# print(analise_clusters)

# # --- VISUALIZAÇÃO (PCA) ---
# # Como são muitas features, o PCA ajuda a ver os grupos em 2D
# pca = PCA(n_components=2)
# components = pca.fit_transform(df_zscore)
# df['PCA1'] = components[:, 0]
# df['PCA2'] = components[:, 1]

# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster_Label', style='Label_Estatistica', s=100)
# plt.title('Comparação: Label por Cluster vs Label por Score Estatístico')
# plt.savefig('pca_comparison.png')

# # Exibir os resultados finais
# print(df[['Participant ID', 'Global_Response_Score', 'Label_Estatistica', 'Cluster_Label']].head())

df.to_csv('FCS_PUBLIC_labeled.csv', index=False)