import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
df = pd.read_csv('JM-experiments.csv')

# 2. Definir as variáveis
# Confirme qual é a string exata para 4 dias no seu dataset (ex: '4dpc', 'Day4', etc)
baseline_timepoint = 'Pre'
peak_timepoint = '4dpc' 

# Colunas que DEVEM ser iguais para cruzar os ratos
group_cols = ['Treatment', 'Tissue', 'Target']

# Identificar as colunas que são as características das células (features)
meta_cols = ['Experiment', 'Timepoint', 'Treatment', 'Tissue', 'Mouse', 'Target']
feature_cols = [col for col in df.columns if col not in meta_cols]

# 3. Filtrar os dados (apenas Pre e Pico)
df_pre = df[df['Timepoint'] == baseline_timepoint]
df_peak = df[df['Timepoint'] == peak_timepoint]

# 4. Combinar os dados (Produto Cartesiano por grupo)
# O "merge" vai cruzar todos os ratos Pre com todos os ratos Pico que tenham o mesmo Treatment, Tissue e Target.
df_combined = pd.merge(df_pre, df_peak, on=group_cols, suffixes=('_pre', '_peak'))

# 5. Criar o DataFrame final e calcular o efeito
df_effect = df_combined[group_cols + ['Mouse_pre', 'Mouse_peak']].copy()

for col in feature_cols:
    val_pre = df_combined[f'{col}_pre']
    val_peak = df_combined[f'{col}_peak']
    
    # OPÇÃO 1: Subtração Simples (Descomente a linha abaixo se preferir usar subtração)
    # df_effect[col] = val_peak - val_pre
    
    # OPÇÃO 2: Log2 Fold Change (Recomendado para dados biológicos)
    # Adicionamos um número muito pequeno (1e-5) para evitar divisão por zero ou log(0)
    df_effect[col] = np.log2((val_peak + 1e-5) / (val_pre + 1e-5))

# 6. Normalização (Z-score)
# Vamos normalizar as colunas numéricas para que todas fiquem na mesma escala (média 0, desvio padrão 1)
scaler = StandardScaler()
df_effect[feature_cols] = scaler.fit_transform(df_effect[feature_cols])

# Checar o resultado
print(f"Total de combinações geradas: {len(df_effect)}")
print(df_effect.head())

# Salvar o novo dataset pronto para ser usado nos seus modelos
df_effect.to_csv('JM-experiments_combinacoes_normalizadas.csv', index=False)