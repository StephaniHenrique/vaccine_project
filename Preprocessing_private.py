import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar os datasets
df_JM_train = pd.read_csv('./Dataset/Private_dt/no_standard/JM_no_standard.csv')
df_JM_val = pd.read_csv('./Dataset/Private_dt/no_standard/JM_validation_no_standard.csv')
df_JT_train = pd.read_csv('./Dataset/Private_dt/no_standard/JT_no_standard.csv')
df_JT_val = pd.read_csv('./Dataset/Private_dt/no_standard/JT_validation_no_standard.csv')


mouse_to_human_map = {
    "cd4tconv.cd44": "activated_cd4+_helper_t_cells",      
    "CD8.GrzmB": "activated_cd8+_t_cells",                     
    "CD8.CD44hi": "memory_cd8+_t_cells_(other)",                
    "live.cd8": "cd8+_t_cells_(other)",      
    "mo0.CD11cneg": "classical_monocytes",                     
    "mo0.CD11c": "non-classical_monocytes",   
    "DC": "activated_pdcs",      
    "B.naive.FO": "unspecified_b_cells",                       
    "B.GC": "activated_unspecified_b_cells",                  
    "B.plasma": "plasmablasts_/_plasma_cells"                  }

# #APplying mapping
# df_train = df_train.rename(columns=mouse_to_human_map)
# df_val = df_val.rename(columns=mouse_to_human_map)

#All possile meta cols
meta_cols = ['Participant ID', 'Experiment', 'Treatment', 'Tissue', 'Mouse', 'Target', 
             'Mouse_pre', 'Mouse_peak', 'Timepoint_pre', 'Timepoint_peak', 'Label_HAI', 'Cluster']

def clean_and_impute_features(df, meta_cols, threshold=0.5):
    features = [c for c in df.columns if c not in meta_cols]
    df_features = df[features]
    
    missing_pct = df_features.isnull().mean()
    cols_to_keep = missing_pct[missing_pct <= threshold].index
    df_features = df_features[cols_to_keep]
    
    #imputation
    df_features = df_features.fillna(df_features.mean(numeric_only=True))
    
    df_meta = df[[c for c in df.columns if c in meta_cols]]
    return pd.concat([df_meta, df_features], axis=1)

df_JM_train_cleaned = clean_and_impute_features(df_JM_train, meta_cols)
df_JM_val_cleaned = clean_and_impute_features(df_JM_val, meta_cols)
df_JT_train_cleaned = clean_and_impute_features(df_JT_train, meta_cols)
df_JT_val_cleaned = clean_and_impute_features(df_JT_val, meta_cols)
# df_public_cleaned = clean_and_impute_features(df_public, meta_cols)

#FEATURES IN COMMON
features_JM = [c for c in df_JM_train_cleaned.columns if c not in meta_cols]
features_JT = [c for c in df_JT_train_cleaned.columns if c not in meta_cols]
common_features = list(set(features_JM).intersection(set(features_JT)))

print({len(common_features)})
print(common_features)

mapping = {'no': 0, 'maybe': 1, 'yes': 1, 0.0: 0, 1.0: 1}

df_JM_train_final = df_JM_train_cleaned[common_features + ['Target']].copy()
df_JM_val_final = df_JM_val_cleaned[common_features + ['Target']].copy()
df_JT_train_final = df_JT_train_cleaned[common_features + ['Target']].copy()
df_JT_val_final = df_JT_val_cleaned[common_features + ['Target']].copy()

# df_public_final = df_public_cleaned[['Participant ID', 'Label_HAI'] + common_features].copy()

df_JM_train_final['Target'] = df_JM_train_final['Target'].map(mapping)
df_JM_val_final['Target'] = df_JM_val_final['Target'].map(mapping)
df_JT_train_final['Target'] = df_JT_train_final['Target'].map(mapping)
df_JT_val_final['Target'] = df_JT_val_final['Target'].map(mapping)


df_standardize_JM = pd.concat([df_JM_train_final, df_JM_val_final], ignore_index=True)
df_standardize_JT = pd.concat([df_JT_train_final, df_JT_val_final], ignore_index=True)
scaler = StandardScaler()

#FITTING ONLY IN JM DATA
scaler.fit(df_standardize_JT[common_features])

#applying transformation
df_standardize_JM[common_features] = scaler.transform(df_standardize_JM[common_features])
df_standardize_JT[common_features] = scaler.transform(df_standardize_JT[common_features])


df_standardize_JM.to_csv('JM_final_standard_val.csv', index=False)
df_standardize_JT.to_csv('JT_final_standard_train.csv', index=False)
