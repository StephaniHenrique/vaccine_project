import pandas as pd
import numpy as np

df = pd.read_csv('./dataset_virus_encoded.csv')

meta_cols = [
    col for col in df.columns 
    if not col.endswith('_baseline') and not col.endswith('_peak')
]

print(meta_cols)

df_effect = df[meta_cols].copy() #initializating

baseline_cols = [col for col in df.columns if col.endswith('_baseline')]
peak_cols = [col for col in df.columns if col.endswith('_peak')]
print(len(baseline_cols))
print(len(peak_cols))
#Passing through each feature to calculate the effect
for base_col in baseline_cols:
    feature_name = base_col.replace('_baseline', '')
    peak_col = f"{feature_name}_peak"
    
    if peak_col in df.columns:
        val_pre = df[base_col]
        val_peak = df[peak_col]
        df_effect[feature_name] = val_peak - val_pre
        #Log2 Fold Change 
        # df_effect[feature_name] = np.log2((val_peak + 1e-5) / (val_pre + 1e-5))
    else:
        print(f"Warning: Peak column for {feature_name} not found. Skipping.")


print(f"Samples processed: {len(df_effect)}")
print(len(df_effect.columns))

# 4. Salvar o novo dataset pronto para os modelos
df_effect.to_csv('MULTIPLE_EFFECT_ARCSINH_no_standard.csv', index=False)