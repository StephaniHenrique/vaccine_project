#matching data 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./Dataset/Private_dt/JM_SD.csv') #change for train if u need

baseline_timepoint = 'Pre'
peak_timepoint = '4dpc' 

#matching by Treatment, Tissue and Target
group_cols = ['Treatment', 'Tissue', 'Protection']

meta_cols = ['Experiment', 'Timepoint', 'Treatment', 'Tissue', 'Mouse', 'Protection', 'Age_prime', 'Age_challenge', 'Body_score', 'Weight']
feature_cols = [col for col in df.columns if col not in meta_cols]

df_pre = df[df['Timepoint'] == baseline_timepoint]
df_peak = df[df['Timepoint'] == peak_timepoint]

df_combined = pd.merge(df_pre, df_peak, on=group_cols, suffixes=('_pre', '_peak'))

df_effect = df_combined[group_cols + ['Mouse_pre', 'Mouse_peak', 'Age_prime_peak', 'Age_challenge_peak', 'Body_score_peak', 'Weight_peak']].copy()

for col in feature_cols:
    val_pre = df_combined[f'{col}_pre']
    val_peak = df_combined[f'{col}_peak']
    
    # df_effect[col] = val_peak - val_pre
    df_effect[col] = np.log2((val_peak + 1e-5) / (val_pre + 1e-5))


print(len(df_effect))
print(df_effect.head())

df_effect.to_csv('JM_SD_no_standard.csv', index=False)#change for train if u need