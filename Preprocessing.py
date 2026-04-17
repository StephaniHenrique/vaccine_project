import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import IterativeImputer


dataset = 'JM_experiments'
input_missing_values = False
scaler_option = 2 #0 -> no Scaling, 1 -> minmax, 2 -> standard

df_raw_data = pd.read_csv(f'./Dataset/Private_dt/{dataset}.csv')

df_raw_data.isnull().sum().sort_values(ascending=False)[:]

if input_missing_values == False:
    df_raw_data = df_raw_data.dropna(axis=0)

df_raw_data.isnull().sum().sort_values(ascending=False)[:]


dataset_features = df_raw_data.drop('target', axis=1)
X = dataset_features.to_numpy()

dataset_targets = df_raw_data.loc[:, 'target']
y = dataset_targets.to_numpy()

print("Features (X):")
print(X)
print("\nTargets (y):")
print(y)

print("\nShapes:")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTypes:")
print("X type:", type(X))
print("y type:", type(y))

inputer = None

if input_missing_values:
    print('oi')
    inputer = IterativeImputer(estimator=BayesianRidge(), missing_values=np.nan, max_iter=100, tol=0.001, n_nearest_features=None, initial_strategy='mean', imputation_order='ascending', random_state=random_state)
    X = inputer.fit_transform(X)

if scaler_option == 1:
    scaler = MinMaxScaler()
elif scaler_option == 2:
    scaler = StandardScaler()

if scaler_option > 0:
    X = scaler.fit_transform(X)

features_indexes = df_raw_data.columns[0:-1]
print(features_indexes)

df_processed_data['target'] = y