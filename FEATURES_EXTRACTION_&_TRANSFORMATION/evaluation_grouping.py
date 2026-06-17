import pandas as pd

df = pd.read_csv('./results_intermediate.csv')

df_agrupado = df.groupby('Intermediate_Population')['Population Definition Reported'].unique().apply(list).reset_index()

df_agrupado.rename(columns={'Population Definition Reported': 'Lista_de_Definicoes'}, inplace=True)

df_agrupado.to_csv('grouped_pop.csv', index=False)