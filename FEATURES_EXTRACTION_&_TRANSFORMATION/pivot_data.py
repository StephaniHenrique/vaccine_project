import pandas as pd
import numpy as np

def pivoting_data(data_path, mapping_path, output_path):
    
    df_FC = pd.read_excel(data_path)
    df_map = pd.read_csv(mapping_path)

    #tested in smaller subset
    df_map_melted = df_map.melt(id_vars=['Generic_Feature'], var_name='Study_ID', value_name='Population_Definition')
    df_map_melted = df_map_melted.dropna(subset=['Population_Definition'])
    df_map_melted = df_map_melted[df_map_melted['Population_Definition'] != 'None']

    map_dict = {
        (str(row['Study_ID']).strip(), str(row['Population_Definition']).strip()): row['Generic_Feature']
        for _, row in df_map_melted.iterrows()
    }

    print(map_dict)
    #example of how the mapping should work
    # map_dict = {
    #     ('301', 'CD4+ T cells') : 'CD4_T_cells',
    #     ('301', 'CD8+ T cells') : 'CD8_T_cells',
    #     ('301', 'B cells') : 'B_cells',
    #     ('301', 'NK cells') : 'NK_cells',...
    # }

    def mapping_feature(row):
        participant_id = str(row['Participant ID'])
        study_id = participant_id.split('.')[-1].strip() if '.' in participant_id else ''
        
        #getting the original marker
        pop_def = str(row['Population Definition Reported']).strip()
        return map_dict.get((study_id, pop_def), None)

    df_FC['Generic_Feature'] = df_FC.apply(mapping_feature, axis=1) #Applying line by line
    df = df_FC.dropna(subset=['Generic_Feature']).copy() #dropping rows that couldn't be mapped to any generic feature

    identity_col = ['Participant ID', 'Age Reported', 'Gender', 'Race', 'Cohort']
    df_agg = df.groupby(identity_col + ['Study Time Collected', 'Generic_Feature'])['Population Cell Number'].mean().reset_index()

    #getting day 0 and 7
    df_final = df_agg[df_agg['Study Time Collected'].isin([0, 7])].copy()

    def classify_date(row):
        feature = str(row['Generic_Feature']).strip().lower().replace(" ", "_")
        if row['Study Time Collected'] == 0:
            return f"{feature}_baseline"
        elif row['Study Time Collected'] == 7:
            return f"{feature}_peak"

    df_final['Feature_Time'] = df_final.apply(classify_date, axis=1)

    df_wide = pd.pivot_table(
        df_final,
        index=identity_col,
        columns='Feature_Time',
        values='Population Cell Number',
        aggfunc='mean'
    ).reset_index()

    df_wide.columns.name = None
    df_wide.to_csv(output_path, index=False)
   
    return df_wide

#I CREATED A FEATURE MAP FOR STUDY BECAUSE SOME DIFFERENT CELL POPULATIONS WERE MAPPED TO THE SAME GENERIC FEATURE, SO I NEEDED TO CHOOSE THE BEST ONE TO REPRESENT THE FEATURE IN THE MODEL. THIS FILE IS "FEATURE_MAP_PER_STUDY.CSV"
#AFTER THIS, AFTER THE SELECTION OF BEST FEATURES TO REPRESENT EACH GENERIC FEATURE PER STUDY, I USED THIS MAP (SELECTED_FEATURES.CSV) TO PIVOT THE TABLE

data_path = './data/fcs_analyzed_result.xlsx'
mapping_path = './data/selected_features.csv'
output_path = './data/dataset_features_per_day.csv'

dataset_final = pivoting_data(data_path, mapping_path, output_path)