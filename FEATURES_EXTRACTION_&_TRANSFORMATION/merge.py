import pandas as pd
import numpy as np

def merge_data(df_path, demo_path, hai_path, output_path):
    df_features = pd.read_csv(df_path)
    
    df_demo = pd.read_excel(demo_path)
    df_hai = pd.read_excel(hai_path)

    df_features['Participant ID'] = df_features['Participant ID'].astype(str).str.strip().str.upper()
    df_demo['Participant ID'] = df_demo['Participant ID'].astype(str).str.strip().str.upper()
    df_hai['Participant ID'] = df_hai['Participant ID'].astype(str).str.strip().str.upper()

    #checking if everything is numeric
    df_hai['Study Time Collected'] = pd.to_numeric(df_hai['Study Time Collected'], errors='coerce')
    df_hai['Value Preferred'] = pd.to_numeric(df_hai['Value Preferred'], errors='coerce')
    
    #removing empty values
    df_hai_valid = df_hai.dropna(subset=['Study Time Collected', 'Value Preferred']).copy()

    #ordering by participant, virus and time, to make sure the baseline is always before the peak
    df_hai_valid = df_hai_valid.sort_values(by=['Participant ID', 'Virus', 'Study Time Collected'])

    #getting the baseline
    df_baseline = df_hai_valid.groupby(['Participant ID', 'Virus']).first().reset_index()
    df_baseline = df_baseline[['Participant ID', 'Virus', 'Value Preferred', 'Study Time Collected']]
    df_baseline = df_baseline.rename(columns={'Value Preferred': 'hai_baseline', 'Study Time Collected': 'day_baseline'})

    #getting peak
    df_hai_valid = df_hai_valid.merge(df_baseline[['Participant ID', 'Virus', 'day_baseline']], on=['Participant ID', 'Virus'])

    #peak needs to be after baseline
    df_hai_pos_vacc = df_hai_valid[df_hai_valid['Study Time Collected'] > df_hai_valid['day_baseline']].copy()

    idx_peak = df_hai_pos_vacc.groupby(['Participant ID', 'Virus'])['Value Preferred'].idxmax()
    df_peak = df_hai_pos_vacc.loc[idx_peak, ['Participant ID', 'Virus', 'Value Preferred', 'Study Time Collected']]
    df_peak = df_peak.rename(columns={'Value Preferred': 'hai_peak', 'Study Time Collected': 'day_peak'})

    df_hai_wide = pd.merge(df_baseline, df_peak, on=['Participant ID', 'Virus'], how='left')

    df_hai_wide['time_to_peak'] = df_hai_wide['day_peak'] - df_hai_wide['day_baseline']

    #FOld change HAI
    df_hai_wide['hai_rate'] = np.where(
        df_hai_wide['hai_baseline'] == 0, 
        np.nan, 
        df_hai_wide['hai_peak'] / df_hai_wide['hai_baseline']
    )

    df_hai_wide['hai_velocity'] = (df_hai_wide['hai_peak'] - df_hai_wide['hai_baseline']) / df_hai_wide['time_to_peak']

    #cleaning repeated columns between features and demographics, to avoid confusion when merging
    overlap_col = ['Age Reported', 'Gender', 'Race', 'Cohort', 'Age Unit', 'Age Event', 'Ethnicity', 'Species']
    remove_col = [c for c in overlap_col if c in df_features.columns] #Checking if the demo features are also in the fc features
    df_features_clean = df_features.drop(columns=remove_col)

    df_base = pd.merge(df_demo, df_features_clean, on='Participant ID', how='left')
    df_final = pd.merge(df_hai_wide, df_base, on='Participant ID', how='left')

    df_final.to_csv(output_path, index=False)
    return df_final


df_path = './data/dataset_features_per_day.csv'
demo_path = './data/demographics_2026-05-14_15-43-57.xlsx'
hai_path = './data/hai_2026-05-14_15-44-31.xlsx'
output_path = './data/dataset_merged.csv'

dataset_completo = merge_data(df_path, demo_path, hai_path, output_path)