import pandas as pd

df = pd.read_parquet('./Dataset/Public_dt/Before_preprocessing/fcs_analyzed_result.parquet')

df["Population Name Reported"].value_counts().to_csv("population_name_reported_counts.csv", index=True)
df["Population Definition Reported"].value_counts().to_csv("population_definition_reported_counts.csv", index=True)

df['Study_ID'] = df['Participant ID'].astype(str).str.split('.').str[1]
df['Patient_ID'] = df['Participant ID']

df_grouped = df.groupby(
    ['Population Definition Reported', 'Cell Number Unit'], 
    dropna=True
).agg(
    Count=('Study_ID', 'size'),               
    Unique_Studies=('Study_ID', 'nunique'),   
    Unique_Patients=('Patient_ID', 'nunique'), 
    Unique_Timepoints=('Study Time Collected', 'nunique'), 
    Timepoints_List=('Study Time Collected', lambda x: ', '.join(x.dropna().unique().astype(str))),
    Study_IDs_List=('Study_ID', lambda x: ', '.join(x.unique().astype(str)))
).reset_index()


df_grouped = df_grouped[
    (df_grouped['Count'] > 50) & 
    (df_grouped['Unique_Patients'] > 1) & 
    (df_grouped['Unique_Timepoints'] > 1)
]

df_grouped = df_grouped.sort_values(by='Count', ascending=False)

df_grouped.to_csv("population_definition_and_unit_counts.csv", index=False)

df_grouped["Cell Number Unit"].value_counts().to_csv("cell_number_unit_counts.csv", index=True)

all_ids_combined = ', '.join(df_grouped['Study_IDs_List'].dropna())
final_ids_list = sorted(list(set([id_.strip() for id_ in all_ids_combined.split(',') if id_.strip()])))

with open("list_study_ids.txt", "w") as f:
    f.write("\n".join(final_ids_list))