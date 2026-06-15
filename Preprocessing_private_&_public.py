import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar os datasets
df_train = pd.read_csv('./Dataset/Private_dt/no_standard/JM_no_standard.csv')
df_val = pd.read_csv('./Dataset/Private_dt/no_standard/JM_validation_no_standard.csv')
df_public = pd.read_csv('./FCS_PUBLIC_efeito_calculado.csv')

# =========================================================================
# DICIONÁRIO DE MAPEAMENTO BIOLÓGICO (Ratinho -> Humano)
# Mapeamento 1:1 sugerido com base na imunologia das populações celulares.
# Modifique os valores da direita caso queira ajustar a correspondência.
# =========================================================================
mouse_to_human_map = {
    # --- Células CD4 T ---
    "cd4tconv.cd44": "activated_cd4+_helper_t_cells",           # CD44 high = CD4 ativado/memória
    
    # --- Células CD8 T ---
    "CD8.GrzmB": "activated_cd8+_t_cells",                     # Granzima B = CD8 ativado de resposta aguda
    "CD8.CD44hi": "memory_cd8+_t_cells_(other)",                # CD44 high = CD8 de memória em camundongos
    "live.cd8": "cd8+_t_cells_(other)",                         # CD8 total basal
    
    # --- Monócitos ---
    "mo0.CD11cneg": "classical_monocytes",                      # CD11c- em ratos representa monócitos clássicos
    "mo0.CD11c": "non-classical_monocytes",                     # CD11c+ em ratos representa monócitos não-clássicos
    
    # --- Células Dendríticas ---
    "DC": "activated_pdcs",                                     # Dendríticas como proxy de ativação de pDCs
    
    # --- Células B e Plasmócitos ---
    "B.naive.FO": "unspecified_b_cells",                        # B Folicular Naive = B basal
    "B.GC": "activated_unspecified_b_cells",                    # Centro Germinativo = B fortemente ativado
    "B.plasma": "plasmablasts_/_plasma_cells"                   # Células plasmáticas / Plasmócitos
}

# Aplicar o mapeamento para padronizar os nomes das colunas dos ratinhos
df_train = df_train.rename(columns=mouse_to_human_map)
df_val = df_val.rename(columns=mouse_to_human_map)

# Definição de metadados (colunas que não entram no modelo)
meta_cols = ['Participant ID', 'Experiment', 'Treatment', 'Tissue', 'Mouse', 'Target', 
             'Mouse_pre', 'Mouse_peak', 'Timepoint_pre', 'Timepoint_peak', 'Label_HAI', 'Cluster']

# 2. Função de Limpeza e Imputação
def clean_and_impute_features(df, meta_cols, threshold=0.5):
    features = [c for c in df.columns if c not in meta_cols]
    df_features = df[features]
    
    # Apagar colunas com mais de 50% de nulos
    missing_pct = df_features.isnull().mean()
    cols_to_keep = missing_pct[missing_pct <= threshold].index
    df_features = df_features[cols_to_keep]
    
    # Imputar com a média
    df_features = df_features.fillna(df_features.mean(numeric_only=True))
    
    df_meta = df[[c for c in df.columns if c in meta_cols]]
    return pd.concat([df_meta, df_features], axis=1)

df_train_cleaned = clean_and_impute_features(df_train, meta_cols)
df_val_cleaned = clean_and_impute_features(df_val, meta_cols)
df_public_cleaned = clean_and_impute_features(df_public, meta_cols)

# 3. Identificar as features comuns agora que os nomes combinam
features_train = [c for c in df_train_cleaned.columns if c not in meta_cols]
features_public = [c for c in df_public_cleaned.columns if c not in meta_cols]
common_features = list(set(features_train).intersection(set(features_public)))

print(f"Features em comum encontradas pós-mapeamento ({len(common_features)}):")
print(common_features)

# 4. Criar os datasets finais com features comuns + Identificadores solicitados
mapping = {'no': 0, 'maybe': 1, 'yes': 1, 0.0: 0, 1.0: 1}

df_train_final = df_train_cleaned[common_features + ['Target']].copy()
df_val_final = df_val_cleaned[common_features + ['Target']].copy()

# Mantendo Participant ID, Label_HAI e Cluster no dataset público conforme solicitado
df_public_final = df_public_cleaned[['Participant ID', 'Label_HAI', 'Cluster'] + common_features].copy()

df_train_final['Target'] = df_train_final['Target'].map(mapping)
df_val_final['Target'] = df_val_final['Target'].map(mapping)

# 5. Preparação para a Padronização (Z-Score)
df_train_final["Role_id"] = 0
df_val_final["Role_id"] = 1

df_standardize = pd.concat([df_train_final, df_val_final], ignore_index=True)
scaler = StandardScaler()

# Ajustar (FIT) apenas nos dados de treino para evitar vazamento
mask_train = df_standardize['Role_id'].isin([0, 2])
scaler.fit(df_standardize.loc[mask_train, common_features])

# Aplicar a transformação (TRANSFORM) nos datasets de Ratos
df_standardize[common_features] = scaler.transform(df_standardize[common_features])

df_JM_train_final_standard = df_standardize[df_standardize['Role_id'] == 0].drop('Role_id', axis=1)
df_JM_val_final_standard = df_standardize[df_standardize['Role_id'] == 1].drop('Role_id', axis=1)

# Aplicar o mesmo transform nos dados Humanos (Públicos)# 5. Aplicar o mesmo transform nos dados Humanos (Públicos)
# Lembrando que o scaler só altera as colunas de common_features, deixando ID, Label_HAI e Cluster intactos
df_public_final[common_features] = scaler.transform(df_public_final[common_features])

# =========================================================================
# CRIAÇÃO DAS DUAS VERSÕES DO DATASET PÚBLICO (Alvos Diferentes)
# =========================================================================

# VERSÃO 1: Target é a resposta de anticorpos (Label_HAI)
df_public_hai = df_public_final[common_features].copy()
df_public_hai['Target'] = df_public_final['Label_HAI']

# VERSÃO 2: Target é o agrupamento celular do paciente (Cluster)
df_public_cluster = df_public_final[common_features].copy()
df_public_cluster['Target'] = df_public_final['Cluster']


# =========================================================================
# 6. GRAVAÇÃO DOS ARQUIVOS FINAIS
# =========================================================================
# Salvando os dados dos ratinhos
df_JM_train_final_standard.to_csv('JM_TRAIN_final_standard.csv', index=False)
df_JM_val_final_standard.to_csv('JM_VAL_final_standard.csv', index=False)
pd.concat([df_JM_train_final_standard, df_JM_val_final_standard], ignore_index=True).to_csv('JM_combined_standard.csv', index=False)

# Salvando as duas versões do dataset Humano Público para seus testes
df_public_hai.to_csv('FCS_PUBLIC_target_hai_standard.csv', index=False)
df_public_cluster.to_csv('FCS_PUBLIC_target_cluster_standard.csv', index=False)

print("Processamento concluído! Arquivos de teste gerados:")
print(" - 'FCS_PUBLIC_target_hai_standard.csv' (Prever Resposta HAI)")
print(" - 'FCS_PUBLIC_target_cluster_standard.csv' (Prever Pertencimento ao Cluster)")