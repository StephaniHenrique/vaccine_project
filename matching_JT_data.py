import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df_lungs = pd.read_csv('./Dataset/Private_dt/Before_processing/JT_lungs.csv')
df_spleen = pd.read_csv('./Dataset/Private_dt/Before_processing/JT_spleen.csv')
df_julia = pd.read_csv('./Dataset/Private_dt/Before_processing/JM-experiments.csv')

print("Lungs Dataset:")
print(df_lungs.head())
print("\nSpleen Dataset:")
print(df_spleen.head())

map_lungs = {
    'Strain': 'Experiment',
    'route of administration': 'Treatment',
    'tissues': 'Tissue',
    'Mouse number ': 'Mouse',
    'Mouse dose': 'Timepoint',  # Usamos a dose como proxy para o tempo, já que não temos uma coluna de tempo explícita
    ' Target': 'Target',
    '% T conv (/ live CD45)': 'live.cd4tconv',
    '% Treg (/ CD4+ T cells)': 'cd4.treg',
    '% CD8 (/ live CD45)': 'live.cd8',
    '% GrnzB  (/ CD8 T cells)': 'CD8.GrzmB', 
    '% Dendritc cell (/Live CD45+)': 'DC',
    '% monocytes (/ live CD45)': 'Mo0',
    '% memory (B cells)': 'B.memory'
}

map_spleen = {
    'Strain': 'Experiment',
    'route of administration': 'Treatment',
    'tissues': 'Tissue',
    'Mouse number ': 'Mouse',
    'Mouse dose': 'Timepoint',  # Usamos a dose como proxy para o tempo, já que não temos uma coluna de tempo explícita
    ' Target': 'Target',
    'Tconv (/ CD4 T cells)': 'live.cd4tconv',  # Frequência, mesmo sem o símbolo de %
    '% Treg (/CD4+ T cells)': 'cd4.treg',
    '% CD8 (/live CD45+ cells)': 'live.cd8',
    'GrnzB (/ CD8 T cells)': 'CD8.GrzmB',        # Frequência
    '%dendritic cell (/live CD45+)': 'DC',
    'monocytes (/live CD45+)': 'Mo0',            # Frequência
    '% memory (/B cells)': 'B.memory',
    '% Plasma cells (/B cells)': 'B.plasma',
    '%macrophages (/ liveCD45+)': 'Macro',
    'neutrophil (/ live CD45+)': 'Neutro',       # Frequência
    '% NK (/ live CD45+)': 'NK'
}

df_lungs_harmonized = df_lungs[list(map_lungs.keys())].rename(columns=map_lungs)
df_spleen_harmonized = df_spleen[list(map_spleen.keys())].rename(columns=map_spleen)

julia_columns = df_julia.columns.tolist()

missing_features_lungs = [col for col in julia_columns if col not in df_lungs_harmonized.columns]   
missing_features_spleen = [col for col in julia_columns if col not in df_spleen_harmonized.columns]

print("\nMissing features in Lungs dataset compared to Julia dataset:")
print(missing_features_lungs)
print("\nMissing features in Spleen dataset compared to Julia dataset:")
print(missing_features_spleen)

for col in julia_columns:
    if col not in df_lungs_harmonized.columns:
        df_lungs_harmonized[col] = pd.NA
    if col not in df_spleen_harmonized.columns:
        df_spleen_harmonized[col] = pd.NA

# Reordenar as colunas para o mesmo padrão da Julia
df_lungs_harmonized = df_lungs_harmonized[julia_columns]
df_spleen_harmonized = df_spleen_harmonized[julia_columns]

df_final = pd.concat([df_lungs_harmonized, df_spleen_harmonized], ignore_index=True)

# Visualizar o resultado
print("Colunas integradas com sucesso. Shape do dataset final:", df_final.shape)
print(df_final.head())
df_final.to_csv('Lungs_Spleen_harmonized.csv', index=False)

group_cols = ['Experiment', 'Treatment', 'Tissue', 'Mouse']

meta_cols = ['Experiment', 'Timepoint', 'Treatment', 'Tissue', 'Mouse', 'Target']
feature_cols = [col for col in df_final.columns if col not in meta_cols]

baseline_rep = 'PBS'
df_pre = df_final[df_final['Timepoint'] == baseline_rep]
df_peak = df_final[df_final['Timepoint'] != baseline_rep]

df_combined = pd.merge(df_pre, df_peak, on=group_cols, suffixes=('_pre', '_peak'))
print("Combined dataset:")
print(df_combined.head())

df_effect = df_combined[group_cols + ['Target_peak', 'Timepoint_pre', 'Timepoint_peak']].copy()
df_effect = df_effect.rename(columns={'Target_peak': 'Target'})

for col in feature_cols:
    val_pre = pd.to_numeric(df_combined[f'{col}_pre'], errors='coerce')
    val_peak = pd.to_numeric(df_combined[f'{col}_peak'], errors='coerce')

    # OPÇÃO 1: Subtração Simples (Descomente a linha abaixo se preferir usar subtração)
    # df_effect[col] = val_peak - val_pre
    
    # OPÇÃO 2: Log2 Fold Change (Recomendado para dados biológicos)
    # Adicionamos um número muito pequeno (1e-5) para evitar divisão por zero ou log(0)
    df_effect[col] = np.log2((val_peak + 1e-5) / (val_pre + 1e-5))

# scaler = StandardScaler()
# df_effect[feature_cols] = scaler.fit_transform(df_effect[feature_cols])

print("Dataset com efeitos calculados:")
print(df_effect.head())
df_effect.to_csv('JT_no_standard.csv', index=False)