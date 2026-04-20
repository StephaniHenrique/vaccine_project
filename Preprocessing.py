import pandas as pd
import numpy as np

# Carregar os datasets
df_train = pd.read_csv('./JM-train.csv')
df_val = pd.read_csv('./JM-validation.csv')

# Definição de metadados (colunas que não entram no modelo)
meta_cols = ['Experiment', 'Treatment', 'Tissue', 'Mouse', 'Target', 
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

# 3. Encontrar as features comuns após a limpeza e imputação
features_train = [c for c in df_train_cleaned.columns if c not in meta_cols]
features_val = [c for c in df_val_cleaned.columns if c not in meta_cols]
common_features = list(set(features_train).intersection(set(features_val)))

print(f"Features em comum encontradas ({len(common_features)}):")
print(common_features)

# 4. Criar os datasets finais com features comuns + Target
# Mapeamento do Target para numérico (Ajuste conforme seus labels)
mapping = {'no': 0, 'maybe': 1, 'yes': 1, 0.0: 0, 1.0: 1}

df_train_final = df_train_cleaned[common_features + ['Target']].copy()
df_val_final = df_val_cleaned[common_features + ['Target']].copy()

df_train_final['Target'] = df_train_final['Target'].map(mapping)
df_val_final['Target'] = df_val_final['Target'].map(mapping)

# Remover linhas onde o Target ficou vazio após o mapeamento
df_train_final = df_train_final.dropna(subset=['Target'])
df_val_final = df_val_final.dropna(subset=['Target'])

# Salvar os datasets harmonizados para o treinamento
df_train_final.to_csv('JM_TRAIN_final.csv', index=False)
df_val_final.to_csv('JM_VAL_final.csv', index=False)

print("\nProcessamento concluído.")
print(f"Shape Treino: {df_train_final.shape}")
print(f"Shape Validação: {df_val_final.shape}")

# dataset = './JM-experiments_combinacoes_normalizadas.csv'
# input_missing_values = False
# scaler_option = 2 #0 -> no Scaling, 1 -> minmax, 2 -> standard

# df_raw_data = pd.read_csv(f'./Dataset/Private_dt/{dataset}.csv')

# df_raw_data.isnull().sum().sort_values(ascending=False)[:]

# if input_missing_values == False:
#     df_raw_data = df_raw_data.dropna(axis=0)

# df_raw_data.isnull().sum().sort_values(ascending=False)[:]


# dataset_features = df_raw_data.drop('target', axis=1)
# X = dataset_features.to_numpy()

# dataset_targets = df_raw_data.loc[:, 'target']
# y = dataset_targets.to_numpy()

# print("Features (X):")
# print(X)
# print("\nTargets (y):")
# print(y)

# print("\nShapes:")
# print("X shape:", X.shape)
# print("y shape:", y.shape)

# print("\nTypes:")
# print("X type:", type(X))
# print("y type:", type(y))

# inputer = None

# if input_missing_values:
#     print('oi')
#     inputer = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan, max_iter=100, tol=0.001, n_nearest_features=None, initial_strategy='mean', imputation_order='ascending', random_state=random_state)
#     X = inputer.fit_transform(X)

# if scaler_option == 1:
#     scaler = MinMaxScaler()
# elif scaler_option == 2:
#     scaler = StandardScaler()

# if scaler_option > 0:
#     X = scaler.fit_transform(X)

# features_indexes = df_raw_data.columns[0:-1]
# print(features_indexes)

# df_processed_data['target'] = y