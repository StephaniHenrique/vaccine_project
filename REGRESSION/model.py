import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

# 1. Carregando o seu dataset original
df = pd.read_csv("JM_SD_no_standard.csv")

# 2. Definindo os alvos e separando features
# Você mencionou querer predizer Body_score_peak e Weight_peak.
# Vamos focar no Weight_peak como exemplo principal de regressão contínua.
TARGET = "Body_score_peak"

# Colunas de metadados que não são resultados de citometria de fluxo
meta_cols = [
    'Treatment', 'Tissue', 'Protection', 'Mouse_pre', 'Mouse_peak', 
    'Age_prime_peak', 'Age_challenge_peak', 'Body_score_peak', 'Weight_peak'
]

# Filtrando os dados
X = df.drop(columns=meta_cols, errors='ignore') # Apenas features de citometria
y = df[TARGET]

# Tratando possíveis valores nulos (caso existam no seu dataset)
X = X.fillna(X.median())
y = y.fillna(y.median())

# 3. Criando o fluxo de Validação Cruzada
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# IMPROVEMENT 1: OTIMIZAÇÃO COM GridSearchCV
# ==========================================
# Em vez de testar o modelo com parâmetros fixos, vamos forçar o algoritmo 
# a testar múltiplas combinações para encontrar o melhor R2 e Menor Erro.

print("Iniciando Otimização de Hiperparâmetros...\n")

# A) Otimizando Random Forest
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()), # Random forest não precisa tanto, mas ajuda na estabilidade
    ("model", RandomForestRegressor(random_state=42))
])

rf_params = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_grid.fit(X, y)

print(f"Melhor Random Forest MAE (Erro Médio Absoluto): {-rf_grid.best_score_:.3f}")
print(f"Melhores parâmetros: {rf_grid.best_params_}\n")

# B) Otimizando Lasso (Ótimo para selecionar features em dados biológicos)
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000, random_state=42))
])

# Testando a força da regularização (alpha)
lasso_params = {
    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}

lasso_grid = GridSearchCV(lasso_pipeline, lasso_params, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
lasso_grid.fit(X, y)

print(f"Melhor Lasso MAE (Erro Médio Absoluto): {-lasso_grid.best_score_:.3f}")
print(f"Melhores parâmetros: {lasso_grid.best_params_}\n")

# ==========================================
# IMPROVEMENT 2: EXTRAÇÃO DAS MELHORES FEATURES
# ==========================================
# Agora pegamos o melhor modelo treinado (Lasso costuma ser mais interpretável) 
# e plotamos o que causou a perda de peso.

# best_lasso = lasso_grid.best_estimator_
# coefs = best_lasso.named_steps["model"].coef_

# # Juntando os nomes das colunas com os coeficientes encontrados
# feature_importance = pd.DataFrame({
#     'Feature': X.columns,
#     'Coeficiente': coefs
# })

# # Filtrando o que o Lasso zerou (features inúteis)
# feature_importance = feature_importance[feature_importance['Coeficiente'] != 0]

# # Ordenando pelo impacto absoluto na perda de peso
# feature_importance['Abs_Impact'] = feature_importance['Coeficiente'].abs()
# feature_importance = feature_importance.sort_values(by='Abs_Impact', ascending=False).head(15)

# # Plotando
# plt.figure(figsize=(10, 6))
# colors = ['red' if c < 0 else 'green' for c in feature_importance['Coeficiente'][::-1]]
# plt.barh(feature_importance['Feature'][::-1], feature_importance['Coeficiente'][::-1], color=colors)
# plt.title(f'Top 15 Populações Celulares Associadas ao Weight Peak (Lasso)')
# plt.xlabel('Impacto no Peso (Verde = Aumenta Peso / Vermelho = Maior Perda de Peso)')
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

best_rf = rf_grid.best_estimator_
rf_model = best_rf.named_steps["model"]

# Extract the importance of each cell population
rf_importances = rf_model.feature_importances_

# Create a DataFrame for visualization
rf_feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_importances
})

# Get the top 15 features that contributed most to predicting weight loss
rf_feature_df = rf_feature_df.sort_values(by='Importance', ascending=False).head(15)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.barh(rf_feature_df['Feature'][::-1], rf_feature_df['Importance'][::-1], color='steelblue')
plt.title('Top 15 Most Important Cell Populations for body score (Random Forest)')
plt.xlabel('Relative Importance (Higher means greater impact on prediction)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()