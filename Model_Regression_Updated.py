import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from sklearn.linear_model import (
    Ridge,
    ElasticNet
)

from sklearn.decomposition import PCA

# ======================================================
# CONFIGURAÇÕES
# ======================================================

DATASET_PATH = './Dataset/HAI_DATA/dataset_treino_final.csv'

TARGET_COLUMN = 'Day_28'

RANDOM_STATE = 42

N_JOBS = -1

K_FOLDS = 5

USE_LOG_TARGET = True

REMOVE_DAY0 = False

# ======================================================
# LEITURA DO DATASET
# ======================================================

print('='*60)
print('LOADING DATASET')
print('='*60)

df = pd.read_csv(DATASET_PATH)

print(df.shape)

# ======================================================
# REMOVER LINHAS SEM TARGET
# ======================================================

df = df.dropna(subset=[TARGET_COLUMN])

# ======================================================
# FEATURES E TARGET
# ======================================================

drop_columns = [TARGET_COLUMN]

if REMOVE_DAY0 and 'Day_0' in df.columns:
    drop_columns.append('Day_0')

X = df.drop(columns=drop_columns)

y = df[TARGET_COLUMN].astype(float)

# ======================================================
# TRANSFORMAÇÃO LOG2 NAS FEATURES (DAY_0)
# ======================================================
if 'Day_0' in X.columns:
    print('\nUsing log2 transformation on feature Day_0')
    X['Day_0'] = np.log2(X['Day_0'] + 1)

# ======================================================
# TRANSFORMAÇÃO LOG2
# ======================================================

if USE_LOG_TARGET:

    print('\nUsing log2 transformation on target')

    y = np.log2(y + 1)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print('\nTrain shape:', X_train.shape)
print('Test shape :', X_test.shape)

# ======================================================
# CROSS VALIDATION
# ======================================================

kf = KFold(
    n_splits=K_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ======================================================
# MODELOS E HIPERPARÂMETROS
# ======================================================

models_parameters = {

    # ==================================================
    # RANDOM FOREST
    # ==================================================

    'RandomForestRegressor': {
        'pipeline': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', SelectKBest(score_func=mutual_info_regression)),
            ('model', RandomForestRegressor())
        ]),
        'params': {
            'selector__k': [5, 10, 15, 20],
            'model__n_estimators': [100, 250, 500],
            'model__max_depth': [ None, 5, 10, 20 ],
            'model__min_samples_split': [ 2, 5, 10 ],
            'model__min_samples_leaf': [ 1, 2, 4 ],
            'model__max_features': [ 'sqrt', 'log2' ],
            'model__random_state': [RANDOM_STATE],
            'model__n_jobs': [N_JOBS]
        }
    },

    # ==================================================
    # MLP REGRESSOR
    # ==================================================

    'MLPRegressor': {
        'pipeline': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mutual_info_regression)),
            ('model', MLPRegressor())
        ]),
        'params': {
            'selector__k': [5, 10, 15, 20],
            'model__hidden_layer_sizes': [
                (32,),
                (64,),
                (64, 32),
                (128, 64),
                (128, 64, 32)
            ],
            'model__activation': ['relu','tanh'],
            'model__solver': ['adam'],
            'model__alpha': [1e-5,1e-4,1e-3],
            'model__learning_rate_init': [0.0001,0.001,0.01],
            'model__max_iter': [2000],
            'model__random_state': [RANDOM_STATE]
        }
    },

    # ==================================================
    # SVR
    # ==================================================

    'SVR': {
        'pipeline': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mutual_info_regression)),
            ('model', SVR())
        ]),
        'params': {
            'selector__k': [5, 10, 15, 20],
            'model__kernel': ['rbf'],
            'model__C': [0.1,1,10,100],
            'model__gamma': ['scale','auto'],
            'model__epsilon': [0.01,0.1,0.5]
        }
    },

    # ==================================================
    # RIDGE
    # ==================================================

    'Ridge': {
        'pipeline': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mutual_info_regression)),
            ('model', Ridge())
        ]),
        'params': {
            'selector__k': [5, 10, 15, 20],
            'model__alpha': [0.01,0.1,1,10,100]
        }
    },

    # ==================================================
    # ELASTIC NET
    # ==================================================

    'ElasticNet': {
        'pipeline': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=mutual_info_regression)),
            ('model', ElasticNet())
        ]),
        'params': {
            'selector__k': [5, 10, 15, 20],
            'model__alpha': [0.001,0.01,0.1,1],
            'model__l1_ratio': [0.1,0.3,0.5,0.7,0.9],
            'model__max_iter': [5000],
            'model__random_state': [RANDOM_STATE]
        }
    }
}

# ======================================================
# TREINAMENTO
# ======================================================

results = []
best_model = None
best_rmse = np.inf
best_model_name = None

for model_name, model_data in models_parameters.items():

    print('\n')
    print('='*70)
    print(f'TRAINING: {model_name}')
    print('='*70)

    grid = GridSearchCV(

        estimator=model_data['pipeline'],
        param_grid=model_data['params'],
        cv=kf,

        scoring='neg_root_mean_squared_error',

        n_jobs=N_JOBS,
        verbose=1,

        refit=True
    )

    grid.fit(X_train, y_train)
    pred = grid.predict(X_test)

    # ==============================================
    # REVERTER LOG2
    # ==============================================

    if USE_LOG_TARGET:
        pred_eval = (2 ** pred) - 1
        y_eval = (2 ** y_test) - 1

    else:
        pred_eval = pred
        y_eval = y_test

    # ==============================================
    # MÉTRICAS
    # ==============================================

    rmse = np.sqrt(
        mean_squared_error(
            y_eval,
            pred_eval
        )
    )

    mae = mean_absolute_error(
        y_eval,
        pred_eval
    )

    r2 = r2_score(
        y_eval,
        pred_eval
    )

    print('\nBEST PARAMS:')
    print(grid.best_params_)

    print('\nMETRICS:')
    print('RMSE:', rmse)
    print('MAE :', mae)
    print('R2  :', r2)

    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'BestParams': grid.best_params_
    })

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = grid.best_estimator_
        best_model_name = model_name

# ======================================================
# RESULTADOS FINAIS
# ======================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    'RMSE'
)

print('\n')
print('='*70)
print('FINAL RESULTS')
print('='*70)

print(
    results_df[
        ['Model', 'RMSE', 'MAE', 'R2']
    ]
)

print('\nBEST MODEL:', best_model_name)

print('BEST RMSE :', best_rmse)

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

if best_model_name == 'RandomForestRegressor':

    print('\n')
    print('='*70)
    print('FEATURE IMPORTANCE')
    print('='*70)

    selector = best_model.named_steps['selector']

    model = best_model.named_steps['model']

    selected_features = X.columns[
        selector.get_support()
    ]

    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        'Importance',
        ascending=False
    )

    print(importance_df)

# ======================================================
# PREDIÇÕES EXEMPLO
# ======================================================

print('\n')
print('='*70)
print('EXAMPLE PREDICTIONS')
print('='*70)

example_predictions = pd.DataFrame({
    'Real': y_eval[:10],
    'Predicted': pred_eval[:10]
})

print(example_predictions)

# ======================================================
# SALVAR RESULTADOS
# ======================================================

results_df.to_csv(
    'regression_results.csv',
    index=False
)

example_predictions.to_csv(
    'example_predictions.csv',
    index=False
)

print('\nResults saved successfully.')