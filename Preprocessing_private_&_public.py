import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Carregar os datasets
df_train = pd.read_csv('./Dataset/Private_dt/no_standard/JM_no_standard.csv')
df_val = pd.read_csv('./Dataset/Private_dt/no_standard/JM_validation_no_standard.csv')
df_jt_train = pd.read_csv('./Dataset/Private_dt/no_standard/JT_no_standard.csv')   
df_jt_val = pd.read_csv('./Dataset/Private_dt/no_standard/JT_validation_no_standard.csv')
df_public = pd.read_csv('./Dataset/Public_dt/no_standard/FCS_PUBLIC_no_standard.csv')

# Definição de metadados (colunas que não entram no modelo)
meta_cols = ['Participant ID', 'Experiment', 'Treatment', 'Tissue', 'Mouse', 'Target', 
             'Mouse_pre', 'Mouse_peak', 'Timepoint_pre', 'Timepoint_peak']

def clean_and_impute_features(df, meta_cols, threshold=0.5):
    # Identifica colunas que são features
    features = [c for c in df.columns if c not in meta_cols]
    
    # Seleciona apenas as features para análise de nulos
    df_features = df[features]
    
    # 1. Apagar colunas que têm MAIS de 50% de linhas vazias
    missing_pct = df_features.isnull().mean()
    cols_to_keep = missing_pct[missing_pct <= threshold].index
    df_features = df_features[cols_to_keep]
    
    # 2. Imputar (preencher) valores nas colunas que restaram
    # Usando a média para colunas numéricas
    df_features = df_features.fillna(df_features.mean(numeric_only=True))
    
    # Recombina com os metadados
    df_meta = df[[c for c in df.columns if c in meta_cols]]
    return pd.concat([df_meta, df_features], axis=1)

# Aplicar a limpeza e imputação em ambos
df_train_cleaned = clean_and_impute_features(df_train, meta_cols)
df_val_cleaned = clean_and_impute_features(df_val, meta_cols)
df_jt_train_cleaned = clean_and_impute_features(df_jt_train, meta_cols)
df_jt_val_cleaned = clean_and_impute_features(df_jt_val, meta_cols)
df_public_cleaned = clean_and_impute_features(df_public, meta_cols)

# 3. Encontrar as features comuns após a limpeza e imputação
features_train = [c for c in df_train_cleaned.columns if c not in meta_cols]
features_train_jt = [c for c in df_jt_train_cleaned.columns if c not in meta_cols]
features_public = [c for c in df_public_cleaned.columns if c not in meta_cols]
common_features = list(set(features_train).intersection(set(features_train_jt)).intersection(set(features_public))) #Lista sem meta dados incluindo target

print(f"Features em comum encontradas ({len(common_features)}):")
print(common_features)

# 4. Criar os datasets finais com features comuns + Target
# Mapeamento do Target para numérico (Ajuste conforme seus labels)
mapping = {'no': 0, 'maybe': 1, 'yes': 1, 0.0: 0, 1.0: 1}

df_train_final = df_train_cleaned[common_features + ['Target']].copy()
df_val_final = df_val_cleaned[common_features + ['Target']].copy()
df_JT_train_final = df_jt_train_cleaned[common_features + ['Target']].copy()
df_JT_val_final = df_jt_val_cleaned[common_features + ['Target']].copy()
df_public_final = df_public_cleaned[['Participant ID']+common_features].copy()

df_train_final['Target'] = df_train_final['Target'].map(mapping)
df_val_final['Target'] = df_val_final['Target'].map(mapping)
df_JT_train_final['Target'] = df_JT_train_final['Target'].map(mapping)
df_JT_val_final['Target'] = df_JT_val_final['Target'].map(mapping)
#df_public_final['Target'] = df_public_final['Target'].map(mapping)

# # Remover linhas onde o Target ficou vazio após o mapeamento
# df_train_final = df_train_final.dropna(subset=['Target'])
# df_val_final = df_val_final.dropna(subset=['Target'])

# Salvar os datasets harmonizados para o treinamento
# df_train_final.to_csv('JM_TRAIN_final.csv', index=False)
# df_val_final.to_csv('JM_VAL_final.csv', index=False)
# ... (Seu código acima dessa linha está ótimo!)

df_train_final["Role_id"] = 0
df_val_final["Role_id"] = 1
df_JT_train_final["Role_id"] = 2
df_JT_val_final["Role_id"] = 3
#df_public_final["Role_id"] = 4

df_standardize = pd.concat([df_train_final, df_val_final, df_JT_train_final, df_JT_val_final], ignore_index=True)

scaler = StandardScaler()

# 1. CORREÇÃO DO VAZAMENTO DE DADOS: FIT apenas nos dados de TREINO (Role_id 0 e 2)
mask_train = df_standardize['Role_id'].isin([0, 2])
scaler.fit(df_standardize.loc[mask_train, common_features])

# 2. TRANSFORM em todos os dados do DataFrame combinado
df_standardize[common_features] = scaler.transform(df_standardize[common_features])

# Re-separando os datasets
df_JM_train_final_standard = df_standardize[df_standardize['Role_id'] == 0].drop('Role_id', axis=1)
df_JM_val_final_standard = df_standardize[df_standardize['Role_id'] == 1].drop('Role_id', axis=1)
df_JT_train_final_standard = df_standardize[df_standardize['Role_id'] == 2].drop('Role_id', axis=1)
df_JT_val_final_standard = df_standardize[df_standardize['Role_id'] == 3].drop('Role_id', axis=1)

# 3. CORREÇÃO DO DATASET PÚBLICO: Apenas Transform (sem fit!), mantendo como DataFrame
df_public_final[common_features] = scaler.transform(df_public_final[common_features])
# Agora df_public_final tem as features padronizadas e ainda possui a coluna 'Target'

# 4. CORREÇÃO DA GRAVAÇÃO DE ARQUIVOS: Não atribuir a variáveis
df_JM_train_final_standard.to_csv('JM_TRAIN_final_standard.csv', index=False)
df_JM_val_final_standard.to_csv('JM_VAL_final_standard.csv', index=False)
df_JT_train_final_standard.to_csv('JT_TRAIN_final_standard.csv', index=False)
df_JT_val_final_standard.to_csv('JT_VAL_final_standard.csv', index=False)

pd.concat([df_JM_train_final_standard, df_JT_train_final_standard], ignore_index=True).to_csv('TRAIN_combined_standard.csv', index=False)
pd.concat([df_JM_val_final_standard, df_JT_val_final_standard], ignore_index=True).to_csv('VAL_combined_standard.csv', index=False)

pd.concat([df_JM_train_final_standard, df_JM_val_final_standard], ignore_index=True).to_csv('JM_combined_standard.csv', index=False)
pd.concat([df_JT_train_final_standard, df_JT_val_final_standard], ignore_index=True).to_csv('JT_combined_standard.csv', index=False)

pd.concat([df_JM_train_final_standard, df_JT_train_final_standard, df_JM_val_final_standard, df_JT_val_final_standard], ignore_index=True).to_csv('TRAIN_combo_j&j_standard.csv', index=False)

# Salvando o público sem erros
df_public_final.to_csv('FCS_PUBLIC_final_standard.csv', index=False)