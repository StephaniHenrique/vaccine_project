import pandas as pd

# 1. Carregar os dados originais
df_hai = pd.read_excel('./Dataset/HAI_DATA/hai.xlsx')
df_demo = pd.read_excel('./Dataset/HAI_DATA/demographics.xlsx')

# --- (Opcional) Ver quantidade de pacientes por dia ---
pacientes_por_dia = df_hai.groupby('Study Time Collected')['Participant ID'].nunique().sort_index()
print("\n--- QUANTIDADE DE PACIENTES POR DIA ---")
for dia, quantidade in pacientes_por_dia.items():
    print(f"Dia {dia}: {quantidade} pacientes")
print("---------------------------------------\n")

# 2. Transformar de Vertical para Horizontal
df_hai_horizontal = df_hai.pivot_table(
    index=['Participant ID', 'Virus', 'Cohort'], 
    columns='Study Time Collected', 
    values='Value Preferred',
    aggfunc='mean' 
).reset_index()

# 3. Deixar os nomes das colunas mais bonitos
df_hai_horizontal.columns = [
    f'Day_{col}' if isinstance(col, (int, float)) else col 
    for col in df_hai_horizontal.columns
]

# =========================================================================
# FILTRAR APENAS DIA 0 E DIA 28
# =========================================================================
col_dia_0 = 'Day_0'
col_dia_28 = 'Day_28'

# A) Selecionar apenas as colunas de interesse
df_filtrado = df_hai_horizontal[['Participant ID', 'Virus', 'Cohort', col_dia_0, col_dia_28]].copy()

# B) Remover as linhas que não possuem o Day 0 OU o Day 28
df_filtrado = df_filtrado.dropna(subset=[col_dia_0, col_dia_28])

# =========================================================================
# NOVO: MESCLAR COM DADOS DEMOGRÁFICOS
# =========================================================================
# Selecionar apenas as colunas úteis da demografia e remover duplicados
colunas_demo = ['Participant ID', 'Phenotype', 'Gender', 'Age Reported', 'Ethnicity', 'Race']
df_demo_clean = df_demo[colunas_demo].drop_duplicates()

# Juntar as duas tabelas usando o 'Participant ID'
# how='inner' garante que só mantemos pacientes que existam nas duas tabelas
df_final = pd.merge(df_filtrado, df_demo_clean, on='Participant ID', how='inner')

# =========================================================================

# 4. Visualizar como ficou
print("Visualização do Dataset Final (Com Demografia):")
print(df_final.head(10))
print(f"\nTamanho do dataset final: {df_final.shape[0]} linhas (Combinações Paciente+Vírus)")

# 5. Salvar o resultado final
df_final.to_csv('HAI_Dia0_Dia28_Com_Demografia.csv', index=False)
print("\nArquivo salvo com sucesso como 'HAI_Dia0_Dia28_Com_Demografia.csv'!")
colunas_para_modelo = [
    'Virus', 'Cohort', 'Day_0', 'Gender', 'Age Reported', 
    'Ethnicity', 'Race', 'Day_28'
]

# Criar uma cópia apenas com o necessário
df_preparado = df_final[colunas_para_modelo].copy()

# 3. Limpeza: Remover linhas onde o valor que queremos prever (Day_28) está vazio
df_preparado = df_preparado.dropna(subset=['Day_28'])

# 4. Transformação: Converter texto em números (One-Hot Encoding)
# Isso transformará Virus, Gender, Ethnicity e Race em colunas de 0 e 1
df_final_tudo_junto = pd.get_dummies(
    df_preparado, 
    columns=['Virus', 'Cohort', 'Gender', 'Ethnicity', 'Race'], 
    drop_first=True
    d_type=int
)

# 5. Salvar o arquivo final pronto para o modelo
df_final_tudo_junto.to_csv('dataset_treino_final.csv', index=False)