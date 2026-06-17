import pandas as pd 
import numpy as np

df = pd.read_excel('./demographics.xlsx')

df['study_id'] = df['Participant ID'].astype(str).str.split('.').str[1]

study_count = df['study_id'].value_counts()

print("--- quantity of patients per study! ---")
print(study_count)