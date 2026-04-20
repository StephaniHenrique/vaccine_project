import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
# Substitua o caminho caso necessário
df = pd.read_csv('./Dataset/Private_dt/JM-experiments.csv')

# 2. Definir as variáveis de tempo e grupos
baseline_timepoint = 'Pre'
peak_timepoint = '4dpc' 

# Colunas que DEVEM ser iguais para cruzar os ratos
group_cols = ['Treatment', 'Tissue', 'Target']

# Identificar colunas de metadados e features
meta_cols = ['Experiment', 'Timepoint', 'Treatment', 'Tissue', 'Mouse', 'Target']
feature_cols = [col for col in df.columns if col not in meta_cols]

# 3. Filtrar os dados originais
df_pre = df[df['Timepoint'] == baseline_timepoint].copy()
df_peak = df[df['Timepoint'] == peak_timepoint].copy()

# -----------------------------------------------------------------------------
# PREVENÇÃO DE DATA LEAKAGE: Criação de Blocos Isolados
# -----------------------------------------------------------------------------
n_blocks = 5  # Número de partições (deve ser igual ou compatível com o k_hout do Model.py)

def assign_isolated_blocks(df_subset):
    """
    Atribui cada rato (Mouse) a um bloco específico de 0 a (n_blocks-1).
    Ratos no mesmo bloco podem cruzar-se, mas nunca com ratos de outros blocos.
    """
    mice = df_subset['Mouse'].unique()
    np.random.seed(42) # Semente fixa para garantir reprodutibilidade
    np.random.shuffle(mice)
    
    # Divide os ratos nos blocos
    blocks = np.array_split(mice, n_blocks)
    mouse_to_block = {}
    
    for block_id, block_mice in enumerate(blocks):
        for mouse in block_mice:
            mouse_to_block[mouse] = block_id
            
    return df_subset['Mouse'].map(mouse_to_block)

# Aplicar os blocos separadamente para Pre e Pico, mantendo os agrupamentos
df_pre['Block_ID'] = df_pre.groupby(group_cols, group_keys=False).apply(assign_isolated_blocks)
df_peak['Block_ID'] = df_peak.groupby(group_cols, group_keys=False).apply(assign_isolated_blocks)

# 4. Combinar os dados (Produto Cartesiano RESTRITO AO BLOCO)
# O "merge" agora exige que os ratos estejam no mesmo Block_ID para se cruzarem
df_combined = pd.merge(df_pre, df_peak, on=group_cols + ['Block_ID'], suffixes=('_pre', '_peak'))

# 5. Criar o DataFrame final e calcular o efeito (Log2 Fold Change)
df_effect = df_combined[group_cols + ['Mouse_pre', 'Mouse_peak', 'Block_ID']].copy()

for col in feature_cols:
    val_pre = df_combined[f'{col}_pre']
    val_peak = df_combined[f'{col}_peak']
    
    # Adicionamos 1e-5 para evitar erro matemático de divisão por zero ou log(0)
    df_effect[col] = np.log2((val_peak + 1e-5) / (val_pre + 1e-5))

# 6. Normalização (Z-score)
scaler = StandardScaler()
df_effect[feature_cols] = scaler.fit_transform(df_effect[feature_cols])

# 7. Criar a coluna Target (Alvo) para o ML
# Ajuste esta lógica de acordo com o que você quer que o modelo preveja!
# Exemplo: prever se o tratamento foi PBS (1) ou outro (0)
if 'Treatment' in df_effect.columns:
    df_effect['target'] = df_effect['Treatment'].apply(lambda x: 1 if str(x).upper() == 'PBS' else 0)

# Exportar. O arquivo agora terá a coluna 'Block_ID'
output_filename = 'JM-experiments_combinacoes_normalizadas_BLOCKED.csv'
df_effect.to_csv(output_filename, index=False)
print(f"Ficheiro guardado com sucesso: {output_filename}")
print(f"Total de combinações seguras geradas: {len(df_effect)}")