import pandas as pd
import numpy as np

def mesclar_dados_estudo_por_cepa(caminho_features, caminho_demografia, caminho_hai, caminho_saida):
    print("1. A carregar os conjuntos de dados...")
    df_features = pd.read_csv(caminho_features)
    
    # Ler ficheiros (suporta csv ou excel)
    df_demo = pd.read_excel(caminho_demografia) if caminho_demografia.endswith('.xlsx') else pd.read_csv(caminho_demografia)
    df_hai = pd.read_excel(caminho_hai) if caminho_hai.endswith('.xlsx') else pd.read_csv(caminho_hai)

    print("2. A limpar os IDs para garantir a junção correta...")
    df_features['Participant ID'] = df_features['Participant ID'].astype(str).str.strip().str.upper()
    df_demo['Participant ID'] = df_demo['Participant ID'].astype(str).str.strip().str.upper()
    df_hai['Participant ID'] = df_hai['Participant ID'].astype(str).str.strip().str.upper()

    print("3. A processar os dados de HAI (Cálculo de Baseline, Pico, Taxa e Tempo)...")
    # Garantir que o tempo e os valores sejam tratados como números
    df_hai['Study Time Collected'] = pd.to_numeric(df_hai['Study Time Collected'], errors='coerce')
    df_hai['Value Preferred'] = pd.to_numeric(df_hai['Value Preferred'], errors='coerce')
    
    # Remover linhas sem tempo ou valor para evitar erros nos cálculos
    df_hai_valido = df_hai.dropna(subset=['Study Time Collected', 'Value Preferred']).copy()

    # Ordenar pelo tempo para garantir a captura correta do primeiro dia
    df_hai_valido = df_hai_valido.sort_values(by=['Participant ID', 'Virus', 'Study Time Collected'])

    # --- BASELINE ---
    # Captura o primeiro registo de cada grupo (dia mais antigo disponível)
    df_baseline = df_hai_valido.groupby(['Participant ID', 'Virus']).first().reset_index()
    df_baseline = df_baseline[['Participant ID', 'Virus', 'Value Preferred', 'Study Time Collected']]
    df_baseline = df_baseline.rename(columns={'Value Preferred': 'hai_baseline', 'Study Time Collected': 'day_baseline'})

    # --- PICO ---
    # 1. Trazer a informação do dia do baseline para a tabela completa
    df_hai_valido = df_hai_valido.merge(df_baseline[['Participant ID', 'Virus', 'day_baseline']], on=['Participant ID', 'Virus'])

    # 2. FILTRO CRUCIAL: Só podemos procurar o pico nos dias POSTERIORES ao baseline
    df_hai_pos_vacina = df_hai_valido[df_hai_valido['Study Time Collected'] > df_hai_valido['day_baseline']].copy()

    # 3. Encontrar o índice do valor máximo absoluto APENAS nos dias válidos (> baseline)
    idx_pico = df_hai_pos_vacina.groupby(['Participant ID', 'Virus'])['Value Preferred'].idxmax()
    df_pico = df_hai_pos_vacina.loc[idx_pico, ['Participant ID', 'Virus', 'Value Preferred', 'Study Time Collected']]
    df_pico = df_pico.rename(columns={'Value Preferred': 'hai_peak', 'Study Time Collected': 'day_peak'})

    # Juntar Baseline e Pico
    # Usamos how='left' para não perder os pacientes que SÓ tiveram colheita no baseline (abandono de estudo)
    df_hai_wide = pd.merge(df_baseline, df_pico, on=['Participant ID', 'Virus'], how='left')

    # --- NOVAS FEATURES DE CINÉTICA ---
    
    # Se o paciente não tiver 'day_peak', isto resultará automaticamente em NaN, o que é o correto.
    df_hai_wide['time_to_peak'] = df_hai_wide['day_peak'] - df_hai_wide['day_baseline']

    # Taxa de crescimento (Fold Change)
    df_hai_wide['hai_rate'] = np.where(
        df_hai_wide['hai_baseline'] == 0, 
        np.nan, 
        df_hai_wide['hai_peak'] / df_hai_wide['hai_baseline']
    )

    # Velocidade da Resposta (Aumento absoluto do título por dia)
    # Como garantimos que o pico é DEPOIS do baseline, 'time_to_peak' nunca será 0, evitando erros matemáticos.
    df_hai_wide['hai_velocity'] = (df_hai_wide['hai_peak'] - df_hai_wide['hai_baseline']) / df_hai_wide['time_to_peak']

    print("4. A limpar colunas sobrepostas de demografia...")
    colunas_sobrepostas = ['Age Reported', 'Gender', 'Race', 'Cohort', 'Age Unit', 'Age Event', 'Ethnicity', 'Species']
    colunas_remover_features = [c for c in colunas_sobrepostas if c in df_features.columns]
    df_features_limpo = df_features.drop(columns=colunas_remover_features)

    print("5. A fundir as tabelas (A duplicar a base para cada estirpe viral)...")
    # 1º: Junta demografia com as features (1 linha por paciente)
    df_base = pd.merge(df_demo, df_features_limpo, on='Participant ID', how='outer')

    # 2º: Junta a base com o HAI processado (Repete os dados do paciente para cada vírus existente)
    df_final = pd.merge(df_base, df_hai_wide, on='Participant ID', how='outer')

    print("6. A guardar o conjunto de dados final...")
    df_final.to_csv(caminho_saida, index=False)
    print(f"Sucesso! Ficheiro final guardado em: {caminho_saida}")
    
    return df_final

# --- Execução do Código ---
caminho_features = './data/dataset_pacientes_features_por_dia.csv'
caminho_demografia = './data/demographics_2026-05-14_15-43-57.xlsx'
caminho_hai = './data/hai_2026-05-14_15-44-31.xlsx'
caminho_saida = './data/dataset_completo_mesclado_por_cepa.csv'

dataset_completo = mesclar_dados_estudo_por_cepa(caminho_features, caminho_demografia, caminho_hai, caminho_saida)