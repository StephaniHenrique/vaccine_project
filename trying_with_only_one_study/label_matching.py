import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler, KBinsDiscretizer

#Studies that I choose after preprocessing.py
selected_studies = [113, 296, 301, 311, 312, 314, 364, 478, 514, 519]

input_file = 'datasets_per_study'
out_file = 'datasets_processed_ARCSINH'
os.makedirs(out_file, exist_ok=True)


for study in selected_studies:
    file_path = f"{input_file}/study_{study}.csv"
    
    if not os.path.exists(file_path):
        print(f"FIle not found: {file_path}")
        continue
        
    df = pd.read_csv(file_path)
    
    #Labeling with HAI
    if 'HAI_Peak' in df.columns and 'HAI_Baseline' in df.columns:
        df["Fold_Change_HAI"] = df["HAI_Peak"] / (df["HAI_Baseline"] + 1e-8)
        df["Label_HAI"] = (df["Fold_Change_HAI"] >= 4).astype(int)
    else:
        print(f"study {study} has no HAI DATA!!!!!")
        continue
        
    #Keeping age and gender
    df["Age Reported"] = pd.to_numeric(df["Age Reported"], errors="coerce")
    le = LabelEncoder()
    df["Gender_Encoded"] = le.fit_transform(df["Gender"].astype(str))
    
    
    #Calculating the FC effect
    ignored_columns = [
        'Participant ID', 'Age Reported', 'Gender', 'Race', 'Cohort', 
        'Study Time Collected', 'Study Time Collected Unit', 'Phenotype', 
        'Age Unit', 'Age Event', 'Ethnicity', 'Species', 'Description', 
        'Virus', 'HAI_Baseline', 'HAI_Peak', 'HAI_Rate', 'Study_ID',
         'Gender_Encoded', 'Fold_Change_HAI', 'Label_HAI'
    ]

    features_fc = [col for col in df.columns if col not in ignored_columns]

    metadata_keep = [
        'Participant ID', 'Virus', 'Label_HAI', 
        'Age Reported', 'Gender_Encoded', 'Fold_Change_HAI', 
        'HAI_Baseline', 'HAI_Peak'
    ]

    df_day0 = df[df['Study Time Collected'] == 0]
    df_day7 = df[df['Study Time Collected'] == 7]
    
    #Matching patient data from different days
    df_matched = pd.merge(
        df_day0, 
        df_day7, 
        on=['Participant ID'] + [col for col in metadata_keep if col not in ['Participant ID']], 
        suffixes=('_baseline', '_peak')
    )
    
    if df_matched.empty:
        print(f"study {study}: THere's no patients with day 0/7")
        continue

    
    df_final = df_matched[['Participant ID', 'Virus', 'Label_HAI']].copy()
    

    cofator = 5  
    arcsinh_transformer = FunctionTransformer(
        func=lambda x: np.arcsinh(x / cofator),
        validate=False
    )

    df_matched[features_fc] = arcsinh_transformer.transform(df_matched[features_fc])
    num_bins = 10
    discretizador_multiple = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
    df_matched[features_fc] = discretizador_multiple.fit_transform(df_matched[features_fc])



    #Calculate effect
    for feature in features_fc:
        col_base = f"{feature}_baseline"
        col_peak = f"{feature}_peak"
        
        if col_base in df_matched.columns and col_peak in df_matched.columns:
            val_pre = df_matched[col_base]
            val_peak = df_matched[col_peak]
            
            #Log2 Fold Change
            df_final_effected[feature] = val_peak - val_pre
            # df_final_effected[feature] = np.log2((val_peak + 1e-5) / (val_pre + 1e-5))

    df_final_effected = pd.get_dummies(
        df_final_effected, 
        columns=['Virus'], 
        prefix='Virus', 
        dtype=int
    )


    df_final_effected.to_csv('{out_file}/study_{study}_effect_encoded_ARCSINH.csv', index=False)
