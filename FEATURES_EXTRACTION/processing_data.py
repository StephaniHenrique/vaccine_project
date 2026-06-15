import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
df = pd.read_csv('./data/dataset_completo_mesclado_por_cepa.csv')

# 2. Definir o "Núcleo Duro" baseado no Mapeamento Biológico (Humano)
# Note que o dataset tem dados para o dia 0, dia 1 e dia 7. Vamos englobar o prefixo base.
core_human_populations = [
    # --- T Cells ---
    'cd4+_helper_t_cells', 
    'cd8+_t_cells', # Vai capturar tanto 'other' quanto 'activated'
    'cd8+_regularoty_t_cells', # Nome exato que consta no dataset para Tregs
    
    # --- B Cells ---
    'unspecified_b_cells', # Vai capturar 'activated_unspecified_b_cells'
    'plasmablasts_/_plasma_cells', 
    
    # --- Myeloid / Innate ---
    'classical_monocytes', # Equivalente ao Ly6C+ do camundongo
    'pdcs', # Captura 'activated_pdcs' e 'cd2_functional_pdc_subsets' (Células Dendríticas)
    
    # --- Outros importantes se quiser manter (opcional, mas bom ter) ---
    'nk_cells', 
    'neutrophils', 
    'granulocytes'
]

# Pegar todas as colunas do dataset que contêm esses nomes cruciais (para todos os dias)
core_columns = [col for col in df.columns if any(pop in col for pop in core_human_populations)]
metadata_cols = ['Participant ID', 'Cohort', 'Phenotype', 'Gender', 'Age Reported', 'Species', 'Treatment', 'Tissue']

# 3. Filtragem de Colunas (Features)
# Vamos dropar colunas que não são do "núcleo duro" e têm > 45% de missing values
missing_percent = df.isnull().mean()
columns_to_keep = []

for col in df.columns:
    if col in metadata_cols or col in core_columns:
        # Para features essenciais, somos lenientes e aceitamos até 70% de ausência
        if missing_percent[col] < 0.70: 
            columns_to_keep.append(col)
    else:
        # Para outras features (ruído), cortamos rigidamente em 45%
        if missing_percent[col] < 0.45:
            columns_to_keep.append(col)

df_filtered_cols = df[columns_to_keep].copy()

# 4. Filtragem de Linhas (Samples)
# Se uma amostra (paciente) não tem dados para > 50% das features biológicas CHAVE, ela é inútil.
# Atualizamos a lista de core_columns com base no que sobreviveu à filtragem de colunas
surviving_core_cols = [c for c in core_columns if c in df_filtered_cols.columns]

# Threshold: O sample deve ter pelo menos 50% das colunas chave não-nulas
min_valid_core_features = len(surviving_core_cols) * 0.50
df_filtered_rows = df_filtered_cols.dropna(thresh=min_valid_core_features, subset=surviving_core_cols).copy()

print(f"Dataset original: {df.shape}")
print(f"Dataset após filtragem de ruído: {df_filtered_rows.shape}")

# 5. Preparação para Imputação
# Separar metadados categóricos dos dados numéricos contínuos (as células)
df_numeric = df_filtered_rows.select_dtypes(include=[np.number]).copy()
df_metadata = df_filtered_rows.select_dtypes(exclude=[np.number]).copy()

# 6. Escalonamento (Crucial para variáveis biológicas com contagens muito diferentes)
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(df_numeric)
df_scaled = pd.DataFrame(numeric_scaled, columns=df_numeric.columns, index=df_numeric.index)

# 7. Imputação Iterativa (MICE)
# Este algoritmo vai prever o valor de células CD8 ativadas baseando-se no comportamento de monócitos e NKs daquele mesmo paciente
imputer = IterativeImputer(max_iter=15, random_state=42, n_nearest_features=None)
df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)

# 8. Reverter o Escalonamento para obter os valores biológicos reais de volta
df_imputed_numeric = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df_imputed_scaled.columns, index=df_imputed_scaled.index)

# Garantir que não haja contagens negativas criadas pela regressão
df_imputed_numeric[df_imputed_numeric < 0] = 0

# 9. Juntar novamente com os metadados
df_final = pd.concat([df_metadata, df_imputed_numeric], axis=1)

# Salvar o dataset resgatado
df_final.to_csv('dataset_imputado_biologico.csv', index=False)
print("Dados imputados e salvos com sucesso!")