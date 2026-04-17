import pandas as pd

dataset = 'JM-experiments'
df = pd.read_csv(f'Dataset/Private_dt/{dataset}.csv')


#Extracting basic metrics about the dataset
total_rows = len(df)
unique_experiments = df['Experiment'].nunique()
unique_timepoints = df['Timepoint'].unique()
unique_mice = df['Mouse'].nunique()
unique_tissues = df['Tissue'].unique()

print("=== MÉTRICAS BÁSICAS ===")
print(f"Total de registros: {total_rows}")
print(f"Total de experimentos diferentes: {unique_experiments}")
print(f"Timepoints encontrados: {', '.join(unique_timepoints)}")
print(f"Tecidos analisados: {', '.join(unique_tissues)}")
print(f"Total de camundongos (Mouse) únicos na base: {unique_mice}\n")

# ==========================================
# 3. CONSTRUIR O MATCH ENTRE TIMEPOINTS
# ==========================================
# Definimos as colunas que, juntas, identificam uma amostra/sujeito único
chaves_de_busca = ['Treatment', 'Tissue']

# Agrupamos pelas chaves e contamos quantos Timepoints únicos cada grupo possui
contagem_timepoints = df.groupby(chaves_de_busca)['Timepoint'].nunique().reset_index()
contagem_timepoints.rename(columns={'Timepoint': 'Qtd_Timepoints'}, inplace=True)

# Mesclamos essa informação de volta ao dataframe original
df_completo = pd.merge(df, contagem_timepoints, on=chaves_de_busca, how='left')

# Separamos os dados: 
# MATCH: Grupos que apareceram em mais de 1 timepoint (ex: Pre e 15dpc)
# SEM MATCH: Grupos que apareceram em apenas 1 timepoint
df_com_match = df_completo[df_completo['Qtd_Timepoints'] > 1].copy()
df_sem_match = df_completo[df_completo['Qtd_Timepoints'] == 1].copy()

print("=== RESULTADOS DO MATCHING ===")
print(f"Registros COM match (monitorados ao longo do tempo): {len(df_com_match)}")
print(f"Registros SEM match (aparecem em apenas 1 timepoint): {len(df_sem_match)}\n")

print(df_com_match)

# ==========================================
# 4. VISUALIZAR OS RESULTADOS
# ==========================================
print("-> Exemplos de amostras COM match:")
# Mostramos as chaves + o Timepoint, ordenados para facilitar a visualização
# display(df_com_match[chaves_de_busca + ['Timepoint']].sort_values(by=chaves_de_busca).head(6))

# print("\n-> Exemplos de amostras SEM match (órfãs):")
# display(df_sem_match[chaves_de_busca + ['Timepoint']].head(6))

# Opcional: Salvar os resultados em novos arquivos CSV para inspecionar no Excel
# df_com_match.to_csv('amostras_com_match.csv', index=False)
# df_sem_match.to_csv('amostras_sem_match.csv', index=False)