import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

#ARRUMAAAR
#Studies that I choose after preprocessing.py
selected_studies = [113, 296, 301, 311, 312, 314, 364, 478, 514, 519]

input_file = 'datasets_per_study'
out_file = 'datasets_LR_processed'
os.makedirs(out_file, exist_ok=True)


for study in selected_studies:
    file_path = f"{input_file}/study_{study}.csv"
    
    if not os.path.exists(file_path):
        print(f"FIle not found: {file_path}")
        continue
        
    df = pd.read_csv(file_path)
        
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
         'Gender_Encoded']

    features_fc = [col for col in df.columns if col not in ignored_columns]

    metadata_keep = [
        'Participant ID', 'Virus',
        'Age Reported', 'Gender_Encoded',
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
      
    df_final_effected = df_matched[['Participant ID', 'Virus', 'HAI_Peak', 'HAI_Baseline']].copy()

    # CORREÇÃO 2: Tudo abaixo foi indentado para ficar dentro do loop 'for study in selected_studies:'
    for feature in features_fc:
        col_base = f"{feature}_baseline"
        col_peak = f"{feature}_peak"

        if col_base in df_matched.columns and col_peak in df_matched.columns:
            df_final_effected[feature] = np.log2(
                (df_matched[col_peak] + 1e-5) / (df_matched[col_base] + 1e-5)
            )

    df_final_effected = pd.get_dummies(
        df_final_effected,
        columns=['Virus'],
        prefix='Virus',
        dtype=int
    )

    df_final_effected.to_csv(f"{out_file}/study_{study}_effect_encoded_LR.csv", index=False)