import pandas as pd
import os


df = pd.read_csv('./dataset_merged.csv')

#Removing duplicate columns if needed
column_y = [col for col in df.columns if col.endswith('_y')]
df = df.drop(columns=column_y)
df.columns = df.columns.str.replace('_x$', '', regex=True)

#Extracting ID study
df['Study_ID'] = df['Participant ID'].astype(str).str.split('.').str[1]

#Creating diferent datasets for each study, so we don't have missing values
output_file= 'datasets_per_study'
os.makedirs(output_file, exist_ok=True)

#listing all of them
studies = df['Study_ID'].dropna().unique()

clean_data = {}

for study in studies:
    df_study = df[df['Study_ID'] == study].copy()
    
    #cleaning 100% empty rows
    df_study = df_study.dropna(axis=1, how='all')
    
    #removing missing values
    df_study_zero_na = df_study.dropna()
    
    if not df_study_zero_na.empty:
        file_path = f"{output_file}/study_{study}.csv"
        df_study_zero_na.to_csv(file_path, index=False)
        clean_data[study] = df_study_zero_na
        
        print(f"study {study}: {df_study_zero_na.shape[0]} samples and {df_study_zero_na.shape[1]} features")
        
        cohorts = df_study_zero_na['Cohort'].unique()
        print(f"---> Cohorts in this study: {', '.join(map(str, cohorts))}")
        