import pandas as pd
import numpy as np

# 1. Carregar os dados (USANDO OS DADOS PADRONIZADOS!)
df_effect = pd.read_csv('./Dataset/Public_dt/FCS_PUBLIC_final_standard.csv')
df_demo = pd.read_excel('./Dataset/Public_dt/Before_preprocessing/demographics_404.xlsx')

# Juntar usando o Participant ID
df_base = pd.merge(df_effect, df_demo[['Participant ID', 'Phenotype', 'Age Reported']], on='Participant ID')

# 2. Calcular os pilares base (Biológico e Clínico) para usar nos cortes
colunas_protetoras = ['B.memory', 'CD8.GrzmB']
colunas_supressoras = ['cd4.treg']

# Pilar Biológico (O Sangue) - Como é Z-score, a soma faz sentido!
df_base['Score_Celular'] = df_base[colunas_protetoras].sum(axis=1) - df_base[colunas_supressoras].sum(axis=1)
df_base['Resposta_Biologica'] = np.where(df_base['Score_Celular'] > 0, 1, 0) # 1 = Acima da média

# Pilar Clínico (O Prontuário)
grupos_fortes = ['Younger adult', 'Older adult, not-frail']
df_base['Resposta_Clinica'] = np.where(df_base['Phenotype'].isin(grupos_fortes), 1, 0) # 1 = Jovem ou Idoso Saudável


# ===============================================================================
# DATASET 1: Apenas os Extremos (Híbrido "Purista")
# ===============================================================================
df_dataset1 = df_base.copy()

condicoes = [
    (df_dataset1['Resposta_Biologica'] == 1) & (df_dataset1['Resposta_Clinica'] == 1),
    (df_dataset1['Resposta_Biologica'] == 0) & (df_dataset1['Resposta_Clinica'] == 0)
]
escolhas = ['Alta', 'Baixa']

df_dataset1['Classificacao'] = np.select(condicoes, escolhas, default='Intermediario')
df_dataset1 = df_dataset1[df_dataset1['Classificacao'] != 'Intermediario'].copy()
df_dataset1['Target'] = np.where(df_dataset1['Classificacao'] == 'Alta', 1, 0)

# Deleta as colunas temporárias DE FORMA DEFINITIVA do dataframe antes de salvar
colunas_para_remover = ['Participant ID', 'Score_Celular', 'Resposta_Biologica', 'Resposta_Clinica', 'Classificacao', 'Phenotype', 'Age Reported']
df_dataset1 = df_dataset1.drop(columns=colunas_para_remover)


# ===============================================================================
# DATASET 2: Apenas Resposta Imunológica (Foco no Sangue)
# ===============================================================================
df_dataset2 = df_base.copy()
df_dataset2['Target'] = df_dataset2['Resposta_Biologica']
df_dataset2 = df_dataset2.drop(columns=['Participant ID','Score_Celular', 'Resposta_Biologica', 'Resposta_Clinica', 'Phenotype', 'Age Reported'])


# ===============================================================================
# DATASET 3: Apenas Resposta Clínica (Foco na Demografia)
# ===============================================================================
df_dataset3 = df_base.copy()
df_dataset3['Target'] = df_dataset3['Resposta_Clinica']
df_dataset3 = df_dataset3.drop(columns=['Participant ID','Score_Celular', 'Resposta_Biologica', 'Resposta_Clinica', 'Phenotype', 'Age Reported'])


# ===============================================================================
# SALVAR (Sem sobrescrever arquivos errados!)
# ===============================================================================
df_dataset1.to_csv('Dataset1_Extremos_Hibrido.csv', index=False)
df_dataset2.to_csv('Dataset2_Apenas_Imunologico.csv', index=False)
df_dataset3.to_csv('Dataset3_Apenas_Clinico.csv', index=False)

print("Tudo pronto! 3 Datasets criados com sucesso.")