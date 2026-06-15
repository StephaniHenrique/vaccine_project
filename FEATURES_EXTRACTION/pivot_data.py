import pandas as pd
import numpy as np

def processar_dados_dinamicos(caminho_dados, caminho_mapeamento, caminho_saida):
    # 1 e 2. Carregar e Preparar (Mantido igual ao seu original)
    print("Carregando e mapeando dados...")
    df_dados = pd.read_excel(caminho_dados)
    df_map = pd.read_csv(caminho_mapeamento)

    df_map_melted = df_map.melt(id_vars=['Generic_Feature'], var_name='Study_ID', value_name='Population_Definition')
    df_map_melted = df_map_melted.dropna(subset=['Population_Definition'])
    df_map_melted = df_map_melted[df_map_melted['Population_Definition'] != 'None']

    dicionario_map = {
        (str(row['Study_ID']).strip(), str(row['Population_Definition']).strip()): row['Generic_Feature']
        for _, row in df_map_melted.iterrows()
    }

    def mapear_feature(row):
        participant_id = str(row['Participant ID'])
        study_id = participant_id.split('.')[-1].strip() if '.' in participant_id else ''
        pop_def = str(row['Population Definition Reported']).strip()
        return dicionario_map.get((study_id, pop_def), None)

    df_dados['Generic_Feature'] = df_dados.apply(mapear_feature, axis=1)
    df = df_dados.dropna(subset=['Generic_Feature']).copy()

    # --- NOVA LÓGICA DE VETORES E DISTÂNCIA ---
    print("Agregando features genéricas e identificando Baseline...")

    # Como reduzimos a dimensionalidade, 2 células específicas podem ter virado a mesma genérica no mesmo dia.
    # Tiramos a média para garantir que haja apenas 1 valor por feature genérica por dia.
    colunas_identificadoras = ['Participant ID', 'Age Reported', 'Gender', 'Race', 'Cohort']
    df_agg = df.groupby(colunas_identificadoras + ['Study Time Collected', 'Generic_Feature'])['Population Cell Number'].mean().reset_index()

    # Descobrir o Baseline Dinâmico (o menor dia registrado para cada paciente)
    baselines = df_agg.groupby('Participant ID')['Study Time Collected'].min().reset_index()
    baselines.rename(columns={'Study Time Collected': 'Baseline_Day'}, inplace=True)
    df_agg = df_agg.merge(baselines, on='Participant ID')

    # Pivotar temporariamente para criar os "Vetores" (cada coluna é uma célula)
    print("Calculando perturbação sistêmica (Z-score e Distância Euclidiana)...")
    df_vectors = df_agg.pivot_table(
        index=['Participant ID', 'Study Time Collected', 'Baseline_Day'], 
        columns='Generic_Feature', 
        values='Population Cell Number'
    ).reset_index()

    features_cols = df_vectors.columns.drop(['Participant ID', 'Study Time Collected', 'Baseline_Day'])

    # Normalização (Z-score): Essencial para que células abundantes não esmaguem células raras
    df_norm = df_vectors.copy()
    df_norm[features_cols] = (df_vectors[features_cols] - df_vectors[features_cols].mean()) / df_vectors[features_cols].std()
    
    # Preencher NaN com 0 (0 no Z-score significa "na média", neutro no cálculo da distância)
    df_norm[features_cols] = df_norm[features_cols].fillna(0)

    # Isolar os vetores de Baseline para o cálculo da distância
    df_baseline_vectors = df_norm[df_norm['Study Time Collected'] == df_norm['Baseline_Day']].set_index('Participant ID')[features_cols]

    # Função para calcular a Distância Euclidiana entre o dia atual e o baseline
    def calcular_distancia(row):
        pid = row['Participant ID']
        if row['Study Time Collected'] == row['Baseline_Day']:
            return 0.0 # A distância do baseline para ele mesmo é zero
        
        if pid in df_baseline_vectors.index:
            vetor_baseline = df_baseline_vectors.loc[pid].values
            vetor_atual = row[features_cols].values
            return np.linalg.norm(vetor_atual - vetor_baseline)
        return 0.0

    df_norm['Distance_to_Baseline'] = df_norm.apply(calcular_distancia, axis=1)

    # Descobrir o dia do Pico (o dia com a maior distância do baseline)
    idx_picos = df_norm.groupby('Participant ID')['Distance_to_Baseline'].idxmax()
    picos = df_norm.loc[idx_picos, ['Participant ID', 'Study Time Collected']]
    picos.rename(columns={'Study Time Collected': 'Peak_Day'}, inplace=True)

    # --- JUNTANDO TUDO E FORMATANDO A SAÍDA ---
    print("Montando o dataset final (Baseline e Peak)...")
    
    # Juntar a informação do dia de pico de volta ao dataframe longo
    df_agg = df_agg.merge(picos, on='Participant ID')

    # Filtrar: Manter apenas as linhas que são o Baseline OU o Pico do paciente
    df_final = df_agg[(df_agg['Study Time Collected'] == df_agg['Baseline_Day']) | 
                      (df_agg['Study Time Collected'] == df_agg['Peak_Day'])].copy()

    # Criar a nomenclatura das colunas
    def classificar_dia(row):
        feature = str(row['Generic_Feature']).strip().lower().replace(" ", "_")
        if row['Study Time Collected'] == row['Baseline_Day']:
            return f"{feature}_baseline"
        else:
            return f"{feature}_peak"

    df_final['Feature_Time'] = df_final.apply(classificar_dia, axis=1)

    # Pivot final para deixar no formato Largo (Wide)
    df_wide = pd.pivot_table(
        df_final,
        index=colunas_identificadoras,
        columns='Feature_Time',
        values='Population Cell Number',
        aggfunc='mean'
    ).reset_index()

    df_wide.columns.name = None
    df_wide.to_csv(caminho_saida, index=False)
    print(f"Processo concluído! Arquivo salvo como: {caminho_saida}")
    
    return df_wide

# Execução
# dataset_final = processar_dados_dinamicos(caminho_dados, caminho_mapeamento, caminho_saida)
# --- Execução do Código ---
caminho_dados = './data/fcs_analyzed_result.xlsx'
caminho_mapeamento = './data/features_uniformizadas.csv'
caminho_saida = './data/dataset_pacientes_features_por_dia.csv'

dataset_final = processar_dados_celulas(caminho_dados, caminho_mapeamento, caminho_saida)