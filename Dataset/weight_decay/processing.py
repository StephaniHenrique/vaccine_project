import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)
import os

def prepare_ml_datasets(input_csv, imputation_strategy='interpolate', encoding_strategy='factorized', 
                      reduce_3=False, curve=False, limit_to_first_5_days=True, limit_to_first_12_days=False, include_pbs=True, possible_vectors=None):
  
    valid_encodings = ['standard', 'factorized', 'grouped_3']
    if encoding_strategy not in valid_encodings:
        raise ValueError(f"Invalid encoding strategy. Choose from: {valid_encodings}")


    df = pd.read_csv(input_csv)
    
    mask_vac = df['vector'].str.contains('VAC', case=False, na=False)
    mask_pbs_control = df['vector'].str.contains('PBS', case=False, na=False)
    
    if include_pbs:
        df = df[mask_vac | mask_pbs_control].copy()
    else:
        mask_active_only = mask_vac & ~df['vector'].str.upper().str.startswith('PBS')
        df = df[mask_active_only].copy()
        
    df['mouse number'] = df['mouse number'].fillna(0).astype(int)

    df['vector'] = df['vector'].str.replace(r'^(PBS-challenge-|PBS-)', '', case=False, regex=True)
   
    df['Group'] = (df['Strain'].astype(str) + '_' + 
                   df['vector'].astype(str))
    
    df_vac = df.copy()
    metadata = df_vac[['mouse number', 'Group', 'exp', 'vector']].copy()
    
    dpv_cols, dpc_cols, discarded_late_dpv_cols, discarded_late_dpc_cols = [], [], [], []
    for col in df_vac.columns:
        col_lower = col.lower()
        if re.match(r'^d\d+', col_lower):
            if 'pc' in col_lower:
                discarded_late_dpc_cols.append(col)
                dpc_cols.append(col)
            else:
                if limit_to_first_12_days:
                    day_num_match = re.search(r'\d+', col_lower)
                    if day_num_match and int(day_num_match.group()) > 12:
                        discarded_late_dpv_cols.append(col)
                        continue
                dpv_cols.append(col)

    if limit_to_first_5_days and discarded_late_dpc_cols:
        df_vac = df_vac.drop(columns=discarded_late_dpc_cols)

    if limit_to_first_12_days and discarded_late_dpv_cols:
        df_vac = df_vac.drop(columns=discarded_late_dpv_cols)


    if imputation_strategy == 'ffill_bfill' and dpv_cols:
        df_vac[dpv_cols] = df_vac[dpv_cols].ffill(axis=1).bfill(axis=1)
    elif imputation_strategy == 'interpolate' and dpv_cols:
        df_vac[dpv_cols] = df_vac[dpv_cols].interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1)
    elif imputation_strategy == 'percentage_change' and dpv_cols:
        d0_candidates = [c for c in dpv_cols if '0' in str(c)]
        if not d0_candidates:
            raise KeyError(f"ERROR: Could not find Day 0 baseline column among: {dpv_cols}")
        d0_col = d0_candidates[0]
        df_vac[dpv_cols] = df_vac[dpv_cols].div(df_vac[d0_col], axis=0) * 100
        df_vac[dpv_cols] = df_vac[dpv_cols].interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1)
        df_vac = df_vac.drop(columns=[d0_col])
        dpv_cols.remove(d0_col)
    elif imputation_strategy == 'per_group' and dpv_cols:
        # Group by vector types (e.g., VAC-H5/NP, VAC-H5, PBS) and mice group and interpolate within those subsets
        group_cols = ['vector', 'mouse number']
        df_vac[dpv_cols] = df_vac.groupby(group_cols)[dpv_cols].transform(
            lambda grp: grp.interpolate(method='linear', axis=0).ffill(axis=0).bfill(axis=0)
        )
        # Fallback to horizontal interpolation if an entire group-vector combination is completely missing a timepoint
        df_vac[dpv_cols] = df_vac[dpv_cols].interpolate(method='linear', axis=1).ffill(axis=1).bfill(axis=1)
    elif imputation_strategy == 'knn_trajectory' and dpv_cols:
        imputer = KNNImputer(n_neighbors=3, weights='distance')
        df_vac[dpv_cols] = imputer.fit_transform(df_vac[dpv_cols])
    
    if reduce_3 and dpv_cols:
        dpv_cols_sorted = sorted(dpv_cols, key=lambda c: int(re.findall(r'\d+', c)[0]) if re.findall(r'\d+', c) else 0)
        downsampled_dpv_cols = dpv_cols_sorted[::3]
        discarded_dpv_cols = [c for c in dpv_cols if c not in downsampled_dpv_cols]
        df_vac = df_vac.drop(columns=discarded_dpv_cols)
        dpv_cols = downsampled_dpv_cols
    elif curve and dpv_cols:
        trajectory_df = pd.DataFrame(index=df_vac.index)
        time_points = np.arange(len(dpv_cols))
        trajectory_df['traj_slope'] = df_vac[dpv_cols].apply(lambda r: np.polyfit(time_points, r.values, 1)[0], axis=1)
        trajectory_df['traj_min_peak'] = df_vac[dpv_cols].min(axis=1)
        trajectory_df['traj_auc'] = df_vac[dpv_cols].sum(axis=1)
        trajectory_df['traj_variance'] = df_vac[dpv_cols].std(axis=1)
        df_vac = df_vac.drop(columns=dpv_cols)
        df_vac = pd.concat([df_vac, trajectory_df], axis=1)

    # Encoding Strategies
    if encoding_strategy == 'factorized':
        df_vac['vector_is_control'] = df_vac['vector'].str.upper().str.startswith('PBS').astype(int)
        df_vac['vector_target'] = df_vac['vector'].str.replace(r'^PBS-', '', case=False, regex=True)
        df_vac = pd.get_dummies(df_vac, columns=['vector_target'], prefix='target')
        df_vac = df_vac.drop(columns=['vector'])
    elif encoding_strategy == 'grouped_3':
        df_vac['vector_grouped'] = df_vac['vector'].str.replace(r'^PBS-', '', case=False, regex=True)
        if possible_vectors is not None:
            df_vac['vector_grouped'] = pd.Categorical(
                df_vac['vector_grouped'], 
                categories=possible_vectors
            )
        df_vac = pd.get_dummies(df_vac, columns=['vector_grouped'], prefix='vector')
        df_vac = df_vac.drop(columns=['vector'])
    else:
        if possible_vectors is not None:
            df_vac['vector'] = pd.Categorical(df_vac['vector'], categories=possible_vectors)
        df_vac = pd.get_dummies(df_vac, columns=['vector'], prefix='vector')

    df_vac = pd.get_dummies(df_vac, columns=['Inoculation'], prefix='inoc')
    bool_cols = df_vac.select_dtypes(include='bool').columns
    df_vac[bool_cols] = df_vac[bool_cols].astype(int)

    y = df_vac['Survival'].values
    base_drop_cols = ['exp', 'Strain', 'mouse number']
  
    X = df_vac.drop(columns=base_drop_cols, errors='ignore').fillna(0)
    df_vac = df_vac.drop(columns=base_drop_cols, errors='ignore').fillna(0)
    return X, y, metadata, df_vac

if __name__ == "__main__":
    input_csv = "./Vector_outcomes_Filtered_Subset_h5np_nopbs.csv"
    list_imputation = ['percentage_change']
    list_encoding = ['grouped_3']
    augmentations = ['ros', 'smote', 'adasyn']
    reduce_3 = False
    curve = False

    all_vectors = ['VAC-H5/NP', 'VAC-H5', 'VAC-NP', 'RAB-H5/NP']

    for imputation in tqdm(list_imputation, desc="Imputation"):
        for encoding in list_encoding:
            print(f"\n\n--- Processing with Imputation: '{imputation}' | Encoding: '{encoding}' ---")
                
            X, y, metadata, df = prepare_ml_datasets(
                input_csv, 
                imputation_strategy=imputation, encoding_strategy=encoding, reduce_3 = reduce_3, curve=curve, possible_vectors=all_vectors)

            df.to_csv(f"./H5NP_NOPBS_PROCESSED_NOEXP.csv", index=False)
             