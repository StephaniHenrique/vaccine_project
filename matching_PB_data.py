import pandas as pd
import numpy as np

# 1. Carregar os dados
df = pd.read_csv('./dataset_virus_grouped.csv')

# Lista de colunas para remoção (REMOVIDO o 'Label_HAI' para que ele seja mantido)
colunas_remover = [
    "hai_baseline", "day_baseline", "hai_peak", "day_peak",
    "time_to_peak", "hai_rate", "hai_velocity", "Fold_Change_HAI",
    # "Label_HAI",  <-- Linha removida daqui
    "Phenotype",  "Age Unit",
    "Age Event", "Ethnicity", "Race", "Species", "Cohort"
]

# 2. Separar Metadados
# O 'Label_HAI' e o 'Cluster' entrarão aqui automaticamente por não estarem na lista acima
meta_cols = [
    col for col in df.columns 
    if not col.endswith('_baseline') and not col.endswith('_peak') and col not in colunas_remover
]

# Criamos o DataFrame final contendo ID, Label_HAI e Cluster
df_effect = df[meta_cols].copy()

# Encontrar todas as colunas de baseline de células
baseline_cols = [col for col in df.columns if col.endswith('_baseline') and col not in colunas_remover]

# 3. Calcular a Diferença / Efeito (Log2 Fold Change)
for base_col in baseline_cols:
    feature_name = base_col.replace('_baseline', '')
    peak_col = f"{feature_name}_peak"
    
    if peak_col in df.columns:
        val_pre = df[base_col]
        val_peak = df[peak_col]
        
        # Log2 Fold Change 
        df_effect[feature_name] = np.log2((val_peak + 1e-5) / (val_pre + 1e-5))

# Checar o resultado
print(f"Total de registros processados: {len(df_effect)}")
print(f"Colunas finais no dataset: {df_effect.columns.tolist()}")

# 4. Salvar o novo dataset pronto para os modelos
df_effect.to_csv('FCS_PUBLIC_effect_encoded_grouped_virus.csv', index=False)