import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

df = pd.read_csv('./data/dataset_merged.csv')

#filtering columns based on missing data
missing_percent = df.isnull().mean()
columns_to_keep = []

all_cols = df.columns.tolist()

exclude_cols = [
    'Participant ID', 'Virus', 'hai_baseline', 'day_baseline', 'hai_peak', 
    'day_peak', 'time_to_peak', 'hai_rate', 'hai_velocity', 'Cohort', 
    'Phenotype', 'Gender', 'Age Reported', 'Age Unit', 'Age Event', 
    'Ethnicity', 'Race', 'Species', 'Description'
]

metadata_cols = [
    'Participant ID', 'Virus', 'Gender', 'Age Reported', 'hai_baseline', 'hai_peak'
]

feature_cols = [c for c in all_cols if c not in exclude_cols]

for col in feature_cols:
    if missing_percent[col] < 0.65: 
            columns_to_keep.append(col)

df_filtered_cols = df[columns_to_keep].copy()

surviving_core_cols = [c for c in feature_cols if c in df_filtered_cols.columns]

#If a patient has less than 70% of the core features, we will drop it from the dataset, as it will be too noisy for imputation
min_valid_core_features = len(surviving_core_cols) * 0.70
df_filtered_rows = df_filtered_cols.dropna(thresh=min_valid_core_features, subset=surviving_core_cols).copy()

surviving_idx = df_filtered_rows.index # to get demographic data
print(f"Original_data: {df.shape}")
print(f"after filtering: {df_filtered_rows.shape}")

total_missing = df_filtered_rows.isnull().sum().sum()

print(f"Total missing values: {total_missing}")

#spliting numeric and metadata for imputation
df_numeric = df_filtered_rows.copy()
df_metadata = df.loc[surviving_idx, metadata_cols].copy()

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(df_numeric)
df_scaled = pd.DataFrame(numeric_scaled, columns=df_numeric.columns, index=df_numeric.index)

print('starting imputation...')

imputer = IterativeImputer(
    estimator=ExtraTreesRegressor(n_estimators=10, random_state=42, n_jobs=-1),
    max_iter=15, 
    n_nearest_features=15,
    imputation_order='random',
    random_state=42, 
    verbose=2
)

df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)

#coming back to the original scale
df_imputed_numeric = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df_imputed_scaled.columns, index=df_imputed_scaled.index)

df_imputed_numeric[df_imputed_numeric < 0] = 0

df_final = pd.concat([df_metadata, df_imputed_numeric], axis=1)

df_final.to_csv('final_data_imputed.csv', index=False)