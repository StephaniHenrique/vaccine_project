import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar os datasets
df_public_301 = pd.read_csv('./study_301_effect_encoded.csv')
df_public_multiple = pd.read_csv('./FCS_PUBLIC_effect_encoded_virus.csv')


#All possile meta cols
meta_cols = ['Participant ID', 'Gender', 'Label_HAI', 'Virus_A/California/7/2009', 'Virus_A/Perth/16/2009', 'Virus_A/Perth/19/2009', 'Virus_A/Victoria/361/2011', 'Virus_B/Brisbane/60/2008', 'Virus_B/Massachusetts/2/2012', 'Virus_B/Wisconsin/01/2010']

fc_features = [c for c in df_public_301.columns if c not in meta_cols]
fc_features_multiple = [c for c in df_public_multiple.columns if c not in meta_cols]

df_standardize_301 = df_public_301.copy()
df_standardize_multiple = df_public_multiple.copy()

scaler = StandardScaler()
scaler_multiple = StandardScaler()

df_standardize_301[fc_features] = scaler.fit_transform(df_standardize_301[fc_features])
df_standardize_multiple[fc_features_multiple] = scaler_multiple.fit_transform(df_standardize_multiple[fc_features_multiple])

df_standardize_301.to_csv('301_final_standard.csv', index=False)
df_standardize_multiple.to_csv('multiple_final_standard.csv', index=False)