import pandas as pd
import numpy as np

#transforming FC data 
df_fc = pd.read_excel('.original_data/fcs_analyzed_result.xlsx')
df_filtered = df_fc[df_fc['Study Time Collected'].isin([0, 7])] #Getting only day 0 and 7

idx_column= [ #Columns to separate samples 
    'Participant ID', 'Age Reported', 'Gender', 'Race', 'Cohort', 
    'Study Time Collected', 'Study Time Collected Unit'
]

duplicates = (
    df_filtered
    .groupby(idx_column+ ['Population Name Reported'])
    .size()
    .reset_index(name='n')
)

print('Duplicate data:')
print(duplicates[duplicates['n'] > 1])

df_horizontal = df_filtered.pivot_table(
    index=colunas_indice, 
    columns='Population Name Reported',     
    values='Population Cell Number',
    aggfunc='mean'                          
).reset_index()
df_horizontal.columns.name = None

#Merging with demographic and HAI data
df_demographics = pd.read_excel('.original_data/demographics.xlsx')
extra_columns_demo = [col for col in df_demographics.columns if col not in df_horizontal.columns or col == 'Participant ID'] #Removing features that already exist in FC data

df_com_demo = pd.merge(
    df_horizontal, 
    df_demographics[extra_columns_demo], 
    on='Participant ID', 
    how='left'
)

#Adding HAI
df_hai_raw = pd.read_excel('.original_data/hai.xlsx') 

#TO match HAI with FC data, some patients need to repeat the FC data to different virus tested in HAI
col_virus = 'Virus'                  
col_timepoint = 'Study Time Collected'   
col_value = 'Value Preferred'                 

# Ordenar por Paciente, Vírus e Tempo (para garantir que o Baseline seja a 1ª linha)
df_hai_raw = df_hai_raw.sort_values(by=['Participant ID', col_virus, col_timepoint])

def cal_hai_metrics(virus):
    #the lowest day represent the "baseline data"
    baseline_val = virus.iloc[0][value_col]
    
    #THe maximum value is the peak
    future_days = virus.iloc[1:]
    if not future_days.empty:
        peak_val = future_days[value_col].max()
    else:
        peak_val = baseline_val #If there's not different days
        
    #Using the threshold indentified before
    if pd.notna(baseline_val) and baseline_val > 0:
        rate = peak_val / baseline_val
    else:
        rate = np.nan
        
    return pd.Series({
        'HAI_Baseline': baseline_val,
        'HAI_Peak': peak_val,
        'HAI_Rate': rate
    })

#Applying function
df_hai_processado = df_hai_raw.groupby(['Participant ID', col_virus]).apply(cal_hai_metrics).reset_index()

#Merging data 
df_final = pd.merge(
    df_com_demo, 
    df_hai_processado, 
    on='Participant ID', 
    how='left'
)

df_final.to_csv('./dataset_merged.csv', index=False)