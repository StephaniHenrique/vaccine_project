#Gerando os dados combinados para o dataset publico, considerando a diferença entre os tempos (similar ao que fizemos para o dataset privado)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
df = pd.read_csv('./Dataset/Public_dt/fcs_pivot.csv')

# 2. Definir as variáveis
# Confirme qual é a string exata para 4 dias no seu dataset (ex: '4dpc', 'Day4', etc)
baseline_timepoint = 0
peak_timepoint = 7

# Colunas que DEVEM ser iguais para cruzar os ratos
group_cols = ['Participant ID']

# Identificar as colunas que são as características das células (features)
meta_cols = ['Participant ID', 'Study Time Collected', 'Cell Number Unit'] 
feature_cols = [col for col in df.columns if col not in meta_cols]

df = df[df['Cell Number Unit'].isin(['percentile'])].copy()

# 3. Filtrar os dados (apenas Pre e Pico)
df_pre = df[df['Study Time Collected'] == baseline_timepoint]
df_peak = df[df['Study Time Collected'] == peak_timepoint]

# 4. Combinar os dados (Produto Cartesiano por grupo)
# O "merge" vai cruzar todos os ratos Pre com todos os ratos Pico que tenham o mesmo Treatment, Tissue e Target.
df_combined = pd.merge(df_pre, df_peak, on=group_cols, suffixes=('_pre', '_peak'))

# 5. Criar o DataFrame final e calcular o efeito
df_effect = df_combined[group_cols].copy()

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
# scaler = StandardScaler()
# df_effect[feature_cols] = scaler.fit_transform(df_effect[feature_cols])

# Checar o resultado
print(f"Total de combinações geradas: {len(df_effect)}")
print(df_effect.head())

# Salvar o novo dataset pronto para ser usado nos seus modelos
df_effect.to_csv('FCS_PUBLIC_no_standard.csv', index=False)