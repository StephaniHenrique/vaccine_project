import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def criar_label_por_features(caminho_dados, caminho_saida, p_value_threshold=0.05):
    print("1. Carregando os dados padronizados...")
    df = pd.read_csv(caminho_dados)
    
    # Separar os dados pelos Targets
    df_t0 = df[df['Target'] == 0]
    df_t1 = df[df['Target'] == 1]
    
    # Identificar todas as colunas que são numéricas e ignorar a coluna 'Target' (e possíveis IDs)
    colunas_features = [col for col in df.columns if col not in ['Target', 'Participant ID'] and pd.api.types.is_numeric_dtype(df[col])]
    
    features_positivas = []
    features_negativas = []
    
    print(f"2. Realizando T-test para identificar features com p-value < {p_value_threshold}...")
    
    # Avaliar cada feature individualmente
    for feature in colunas_features:
        # Extrair os valores, removendo possíveis NaNs
        valores_t1 = df_t1[feature].dropna()
        valores_t0 = df_t0[feature].dropna()
        
        # Realizar o T-test independente (Welch's T-test assumindo variâncias potencialmente diferentes)
        stat, p_value = ttest_ind(valores_t1, valores_t0, equal_var=False)
        
        # Se a diferença for estatisticamente significativa
        if p_value < p_value_threshold:
            # Verifica a direção da mudança
            diff_media = valores_t1.mean() - valores_t0.mean()
            
            if diff_media > 0:
                features_positivas.append(feature)
            elif diff_media < 0:
                features_negativas.append(feature)
    
    print("\n--- Regra Matemática Criada (Validação por T-Test) ---")
    print(f"Total de features analisadas: {len(colunas_features)}")
    print(f"Sobe na Resposta (+) [Significativas]: {len(features_positivas)} features -> {features_positivas}")
    print(f"Desce na Resposta (-) [Significativas]: {len(features_negativas)} features -> {features_negativas}")
    
    if len(features_positivas) == 0 and len(features_negativas) == 0:
        print("\nALERTA: Nenhuma feature obteve significância estatística com o p-value escolhido.")
        return df
    
    # 3. Criar o "Immune Score" (Cálculo Direto usando apenas as features significativas)
    print("\n3. Calculando o Immune Score...")
    score_positivo = df[features_positivas].sum(axis=1) if features_positivas else 0
    score_negativo = df[features_negativas].sum(axis=1) if features_negativas else 0
    
    df['Immune_Score'] = score_positivo - score_negativo
    
    # 4. Definir a Label baseada no Score
    # O valor 0 ainda é um bom corte base para dados padronizados (Z-scores)
    ponto_de_corte = 0 
    
    # Cria a nova Label: Se o Score > 0, é 1. Senão, é 0.
    df['Label_Baseada_Features'] = (df['Immune_Score'] > ponto_de_corte).astype(int)
    
    # 5. Comparar com a Realidade
    acc = accuracy_score(df['Target'], df['Label_Baseada_Features'])
    
    print(f"\n=======================================================")
    print(f"ACURÁCIA DA REGRA MATEMÁTICA vs LABEL REAL: {acc * 100:.2f}%")
    print(f"=======================================================\n")
    
    print("Relatório de Classificação Detalhado:")
    print(classification_report(df['Target'], df['Label_Baseada_Features']))
    
    # 6. Salvar e Visualizar
    df.drop(columns=['Immune_Score']).to_csv(caminho_saida, index=False)
    
    # Gráfico 1: Matriz de Confusão
    cm = confusion_matrix(df['Target'], df['Label_Baseada_Features'])
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Regra: 0', 'Regra: 1'], 
                yticklabels=['Real: 0', 'Real: 1'])
    plt.title("Matriz de Confusão")
    plt.xlabel('Label Criada pelas Features')
    plt.ylabel('Label Real')
    
    # Gráfico 2: Distribuição do Score criado
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df, x='Immune_Score', hue='Target', fill=True, common_norm=False, palette=['red', 'blue'])
    plt.axvline(x=ponto_de_corte, color='black', linestyle='--', label='Ponto de Corte (Limiar)')
    plt.title("Distribuição do Immune Score (Apenas Features Significativas)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('analise_baseada_features_ttest.png')
    print("\nGráficos salvos como 'analise_baseada_features_ttest.png'.")
    
    return df

# --- Execução ---
arquivo_entrada = '../Dataset/Private_dt/TRAIN_combo_j&j_standard.csv'
arquivo_saida = 'TRAIN_combo_com_labels_por_features_ttest.csv'

# Você pode ajustar o p_value_threshold conforme a necessidade do estudo (ex: 0.05 ou 0.01)
df_resultado = criar_label_por_features(arquivo_entrada, arquivo_saida, p_value_threshold=0.05)


# 0,0 60/163
# 1,1 58/163
# 0,1 18/163
# 1,0 27/163