import copy
# import latex
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
import tqdm
import warnings

from datetime import datetime

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as imbpipeline

from imblearn.under_sampling import RandomUnderSampler

from itertools import cycle

from mlxtend.plotting import plot_learning_curves

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

#from sklearn.exceptions import ConvergenceWarning

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.svm import SVC

from sklearn.model_selection import LeaveOneGroupOut

#from sklearn.utils import parallel_backend

### Standard font size in the Matplotlib graphs

matplotlib.rcParams.update({'font.size': 15})
exp_id = 'WEIGHT_DECAY' #MUDAR
script_path = 'LOGO_EXP' #MUDAR

input_folder = f'./Dataset/weight_decay'
input_folder_public = f'./Dataset/weight_decay'

output_folder = f'./Results/{script_path}/{exp_id}'

dataset_validation = 'processed_data_RAB_percentage_change_grouped_3' #MUDAR
dataset = 'processed_data_percentage_change_grouped_3' #MUDAR


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime('%Y-%m-%d_%H-%M-%S')
dt_script_start_string = now.strftime(r'%d/%m/%Y %H:%M:%S')
dt_script_start_string

logging.basicConfig(filename=f'{output_folder}/log_{dt_string}.txt', level=logging.DEBUG)
warnings.filterwarnings('ignore')

logging.info(f'>>>>>>>>>>>>>>> START OF SCRIPT EXECUTION: {dt_script_start_string}. <<<<<<<<<<<<<<<')

dataset_file = None
balancer = None

development_test = False #ADJUST

# hout_test_size = 0.2
# hout_test_size = 0.3

k_hout = 3
k_cv = 3

n_jobs = 2

reducer = None

pca_n_components = 11 #TRATAR OS DADOS PRIMEIROS

reducer_option = 0 # 0 = no dimensionality reduction / 1 = PCA / 2 = LDA

random_state_hout = [20, 40, 60, 80, 100]

random_state_cv = 42
random_state_model = 42
random_state_inner_cv = 42
random_state_outer_cv = 42

ref_metric = 'f1_weighted'
ref_metric_2 = 'accuracy'

sample_balancing_option = 0 # 0 = no balancing / 1 = undersampling / 2 = oversampling

logging.info(f'>>>>>>>>>> PIPELINE: Code/{script_path} <<<<<<<<<<<<<<<')

logging.info(f'>>>>>>>>>> PIPELINE PARAMETERS:')
logging.info(f'\t dataset = {dataset}')
logging.info(f'\t development_test = {development_test}')
logging.info(f'\t k_hout = {k_hout}')
logging.info(f'\t k_cv = {k_cv}')
logging.info(f'\t n_jobs = {n_jobs}')
logging.info(f'\t pca_n_components = {pca_n_components}')
logging.info(f'\t reducer_option = {reducer_option}')
logging.info(f'\t random_state_hout = {random_state_hout}')
logging.info(f'\t random_state_cv = {random_state_cv}')
logging.info(f'\t random_state_model = {random_state_model}')
logging.info(f'\t ref_metric = {ref_metric}')
logging.info(f'\t ref_metric_2 = {ref_metric_2}')
logging.info(f'\t sample_balancing_option = {sample_balancing_option}')

scoring = {
    'f1_weighted_scorer': make_scorer(f1_score, average='weighted', zero_division=0),
    'roc_auc_scorer': make_scorer(roc_auc_score, response_method='predict_proba'),
    'accuracy_scorer': make_scorer(accuracy_score)
}

dataset_file = pd.read_csv(f"{input_folder}/{dataset}.csv")
df_val = pd.read_csv(f"{input_folder_public}/{dataset_validation}.csv")

dataset_q_rows = len(dataset_file)
dataset_q_features = len(dataset_file.columns) - 1
n_classes = dataset_file['Target'].nunique()

dataset_features = dataset_file.drop(['Target', 'Group'], axis=1)
X = dataset_features.to_numpy()

y = dataset_file.loc[:, 'Target']
y = y.to_numpy()

X_val = df_val.drop(['Target', 'Group'], axis=1).to_numpy()
y_val = df_val.loc[:, 'Target'].to_numpy()

logo_outer = LeaveOneGroupOut()
outer_cv_folds = logo_outer.split(X, y, groups=dataset_file['Group'])
k_hout = logo_outer.get_n_splits(X, y, groups=dataset_file['Group'])
logging.info(f'\t quantidade de grupos = {k_hout}')

X_tuni = np.empty([ k_hout ], dtype=object)
y_tuni = np.empty([ k_hout ], dtype=object)
groups_tuni = np.empty([ k_hout ], dtype=object)

X_hout = np.empty([ k_hout ], dtype=object)
y_hout = np.empty([ k_hout ], dtype=object)

for i, (train_index, test_index) in enumerate(outer_cv_folds):

    X_tuni[i] = np.array(X[train_index])
    y_tuni[i] = np.array(y[train_index])
    groups_tuni[i] = np.array(dataset_file['Group'].iloc[train_index])
    
    X_hout[i] = np.array(X[test_index])
    y_hout[i] = np.array(y[test_index])



models_parameters = None


if not(development_test):
    n_neurons_r1_l1 = int(math.ceil((dataset_q_features+n_classes)/2))
    n_neurons_r1_l2 = int(math.ceil((n_neurons_r1_l1+n_classes)/2))
    n_neurons_r1_l3 = int(math.ceil((n_neurons_r1_l2+n_classes)/2))

    n_neurons_r2_l1 = int(math.ceil(((2*dataset_q_features)/3)+n_classes))
    n_neurons_r2_l2 = int(math.ceil(((2*n_neurons_r2_l1)/3)+n_classes))
    n_neurons_r2_l3 = int(math.ceil(((2*n_neurons_r2_l2)/3)+n_classes))

    n_neurons_r3_l1 = int(math.ceil((2*dataset_q_features)-1))
    n_neurons_r3_l2 = int(math.ceil((2*n_neurons_r3_l1)-1))
    n_neurons_r3_l3 = int(math.ceil((2*n_neurons_r3_l2)-1))
    
    n_components_pca = [2, 5, 10] 
    n_components_lda = [1, 2]

    preprocessing_blocks = [
        { # 1. Base
            'select': [None], 'transform': [None]
        },
        { # 2. SelectKBest
            'select': [SelectKBest(mutual_info_classif)], 'select__k': [10, 30, 50], 'transform': [None]
        },
        { # 3. SelectKBest + PCA
            'select': [SelectKBest(mutual_info_classif)], 'select__k': [10, 30, 50],
            'transform': [PCA(svd_solver='full', random_state=random_state_model)], 'transform__n_components': n_components_pca
        },
        { # 4. SelectKBest + LDA
            'select': [SelectKBest(mutual_info_classif)], 'select__k': [10, 30, 50],
            'transform': [LinearDiscriminantAnalysis(solver='svd')], 'transform__n_components': n_components_lda
        },
        { # 8. PCA
            'select': [None], 'transform': [PCA(svd_solver='full', random_state=random_state_model)], 'transform__n_components': n_components_pca
        },
        { # 9. LDA
            'select': [None], 'transform': [LinearDiscriminantAnalysis(solver='svd')], 'transform__n_components': n_components_lda
        }
    ]

    # 2. Definimos os parâmetros "seguros" para cada Classificador
    classifiers_safe_params = {
        'LinearDiscriminantAnalysis': {
            'model': LinearDiscriminantAnalysis(),
            'base_params': {
                'classify__solver': ['svd'], # Mais estável
                'classify__shrinkage': [None] # svd só aceita None, evita crashes
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(),
            'base_params': {
                'impute': [None, SimpleImputer(strategy='median')],
                'classify__solver': ['lbfgs'], # Compatível sempre com l2
                'classify__penalty': ['l2'], 
                'classify__C': [0.1, 1.0, 10.0],
                'classify__max_iter': [1000],
                'classify__n_jobs': [n_jobs],
                'classify__random_state': [random_state_model]
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'base_params': {
                'impute': [None, SimpleImputer(strategy='median')],
                'classify__n_estimators': [100, 300, 500],
                'classify__criterion': ['gini'],
                'classify__max_features': ['sqrt', None],
                'classify__n_jobs': [n_jobs],
                'classify__random_state': [random_state_model]
            }
        },
        'SVC': {
            'model': SVC(),
            'base_params': {
                'classify__C': [0.1, 1.0, 10.0],
                'classify__kernel': ['linear', 'rbf'], # Removido poly para evitar loops infinitos
                'classify__gamma': ['scale'],
                'classify__probability': [True],
                'classify__random_state': [random_state_model]
            }
        },
        'MLPClassifier': {
            'model': MLPClassifier(),
            'base_params': {
                'classify__hidden_layer_sizes': [(n_neurons_r1_l1, n_neurons_r1_l2, n_neurons_r1_l3)], # Mantida apenas 1 arquitetura para agilidade
                'classify__activation': ['relu'], # Padrão ouro
                'classify__solver': ['adam'], # Converge mais rápido
                'classify__learning_rate_init': [0.001, 0.01],
                'classify__max_iter': [1000],
                'classify__random_state': [random_state_model]
            }
        }
    }

    models_parameters = {}
    
    for clf_name, clf_data in classifiers_safe_params.items():
        params_list = []
        for prep_block in preprocessing_blocks:
            # Junta as opções de balanceamento com o bloco de pré-processamento
            combined_dict = {
                'balance': [None, SMOTE(random_state=random_state_model)]
            }
            combined_dict.update(prep_block)
            combined_dict.update(clf_data['base_params'])
            params_list.append(combined_dict)
            
        models_parameters[clf_name] = {
            'model': clf_data['model'],
            'params': params_list
        }

logging.info(f'\t models_parameters = {models_parameters}')

def calculate_confidence_interval(values, p_alpha=0.05):
    return sms.DescrStatsW(values).tconfint_mean(alpha=p_alpha)

def init_calib_cv_it_metric():
    metric_name = f'{ref_metric}'
    
    calib_cv_it_metric = {
        metric_name: np.empty([ 0 ])
    }
    
    return calib_cv_it_metric

def init_cv_it_metrics():
    cv_it_metrics = {
        'params': np.empty([ 0 ], dtype=object),
        'accuracy': np.empty([ 0 ]),
        'balanced_accuracy': np.empty([ 0 ]),
        'f1_fail': np.empty([ 0 ]),
        'f1_success': np.empty([ 0 ]),
        'f1_micro': np.empty([ 0 ]),
        'f1_macro': np.empty([ 0 ]),
        'f1_weighted': np.empty([ 0 ]),
        'precision_fail': np.empty([ 0 ]),
        'precision_success': np.empty([ 0 ]),
        'precision_micro': np.empty([ 0 ]),
        'precision_macro': np.empty([ 0 ]),
        'precision_weighted': np.empty([ 0 ]),
        'recall_fail': np.empty([ 0 ]),
        'recall_success': np.empty([ 0 ]),
        'recall_micro': np.empty([ 0 ]),
        'recall_macro': np.empty([ 0 ]),
        'recall_weighted': np.empty([ 0 ]),
        'jaccard_fail': np.empty([ 0 ]),
        'jaccard_success': np.empty([ 0 ]),
        'jaccard_micro': np.empty([ 0 ]),
        'jaccard_macro': np.empty([ 0 ]),
        'jaccard_weighted': np.empty([ 0 ]),
        }
    return cv_it_metrics

def calculate_calib_metric(calib_cv_it_metric, y, y_pred, y_score):
    metric_name = f'{ref_metric}'
    metric_value = 0.0
    
    if metric_name == 'f1_weighted':
        metric_value = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    calib_cv_it_metric[metric_name] = np.append(calib_cv_it_metric[metric_name], metric_value)

def fill_calib_metric(calib_cv_it_metric):
    metric_name = f'{ref_metric}'
    calib_cv_it_metric[metric_name] = np.append(calib_cv_it_metric[metric_name], 0.0)

def calculate_metrics(cv_it_metrics, y, y_pred, y_score):
    cv_it_metrics['accuracy'] = np.append(cv_it_metrics['accuracy'], accuracy_score(y, y_pred))
    cv_it_metrics['balanced_accuracy'] = np.append(cv_it_metrics['balanced_accuracy'], balanced_accuracy_score(y, y_pred))
    
    f1_fail, f1_success = f1_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
    
    cv_it_metrics['f1_fail'] = np.append(cv_it_metrics['f1_fail'], f1_fail)
    cv_it_metrics['f1_success'] = np.append(cv_it_metrics['f1_success'], f1_success)
    
    cv_it_metrics['f1_micro'] = np.append(cv_it_metrics['f1_micro'], f1_score(y, y_pred, average='micro', zero_division=0))
    cv_it_metrics['f1_macro'] = np.append(cv_it_metrics['f1_macro'], f1_score(y, y_pred, average='macro', zero_division=0))
    cv_it_metrics['f1_weighted'] = np.append(cv_it_metrics['f1_weighted'], f1_score(y, y_pred, average='weighted', zero_division=0))
    
    pre_fail, pre_success = precision_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
    
    cv_it_metrics['precision_fail'] = np.append(cv_it_metrics['precision_fail'], pre_fail)
    cv_it_metrics['precision_success'] = np.append(cv_it_metrics['precision_success'], pre_success)
    
    cv_it_metrics['precision_micro'] = np.append(cv_it_metrics['precision_micro'], precision_score(y, y_pred, average='micro', zero_division=0))
    cv_it_metrics['precision_macro'] = np.append(cv_it_metrics['precision_macro'], precision_score(y, y_pred, average='macro', zero_division=0))
    cv_it_metrics['precision_weighted'] = np.append(cv_it_metrics['precision_weighted'], precision_score(y, y_pred, average='weighted', zero_division=0))
    
    rec_fail, rec_success = recall_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
    
    cv_it_metrics['recall_fail'] = np.append(cv_it_metrics['recall_fail'], rec_fail)
    cv_it_metrics['recall_success'] = np.append(cv_it_metrics['recall_success'], rec_success)

    cv_it_metrics['recall_micro'] = np.append(cv_it_metrics['recall_micro'], recall_score(y, y_pred, average='micro', zero_division=0))
    cv_it_metrics['recall_macro'] = np.append(cv_it_metrics['recall_macro'], recall_score(y, y_pred, average='macro', zero_division=0))
    cv_it_metrics['recall_weighted'] = np.append(cv_it_metrics['recall_weighted'], recall_score(y, y_pred, average='weighted', zero_division=0))
    
    jac_fail, jac_success = jaccard_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
    
    cv_it_metrics['jaccard_fail'] = np.append(cv_it_metrics['jaccard_fail'], jac_fail)
    cv_it_metrics['jaccard_success'] = np.append(cv_it_metrics['jaccard_success'], jac_success)
    
    cv_it_metrics['jaccard_micro'] = np.append(cv_it_metrics['jaccard_micro'], jaccard_score(y, y_pred, average='micro', zero_division=0))
    cv_it_metrics['jaccard_macro'] = np.append(cv_it_metrics['jaccard_macro'], jaccard_score(y, y_pred, average='macro', zero_division=0))
    cv_it_metrics['jaccard_weighted'] = np.append(cv_it_metrics['jaccard_weighted'], jaccard_score(y, y_pred, average='weighted', zero_division=0))
    

def fill_metrics(cv_it_metrics):
    cv_it_metrics['accuracy'] = np.append(cv_it_metrics['accuracy'], 0.0)
    cv_it_metrics['balanced_accuracy'] = np.append(cv_it_metrics['balanced_accuracy'], 0.0)
    
    cv_it_metrics['f1_fail'] = np.append(cv_it_metrics['f1_fail'], 0.0)
    cv_it_metrics['f1_success'] = np.append(cv_it_metrics['f1_success'], 0.0)
    
    cv_it_metrics['f1_micro'] = np.append(cv_it_metrics['f1_micro'], 0.0)
    cv_it_metrics['f1_macro'] = np.append(cv_it_metrics['f1_macro'], 0.0)
    cv_it_metrics['f1_weighted'] = np.append(cv_it_metrics['f1_weighted'], 0.0)
    
    cv_it_metrics['precision_fail'] = np.append(cv_it_metrics['precision_fail'], 0.0)
    cv_it_metrics['precision_success'] = np.append(cv_it_metrics['precision_success'], 0.0)
    
    cv_it_metrics['precision_micro'] = np.append(cv_it_metrics['precision_micro'], 0.0)
    cv_it_metrics['precision_macro'] = np.append(cv_it_metrics['precision_macro'], 0.0)
    cv_it_metrics['precision_weighted'] = np.append(cv_it_metrics['precision_weighted'], 0.0)
    
    cv_it_metrics['recall_fail'] = np.append(cv_it_metrics['recall_fail'], 0.0)
    cv_it_metrics['recall_success'] = np.append(cv_it_metrics['recall_success'], 0.0)
    
    cv_it_metrics['recall_micro'] = np.append(cv_it_metrics['recall_micro'], 0.0)
    cv_it_metrics['recall_macro'] = np.append(cv_it_metrics['recall_macro'], 0.0)
    cv_it_metrics['recall_weighted'] = np.append(cv_it_metrics['recall_weighted'], 0.0)
    
    cv_it_metrics['jaccard_fail'] = np.append(cv_it_metrics['jaccard_fail'], 0.0)
    cv_it_metrics['jaccard_success'] = np.append(cv_it_metrics['jaccard_success'], 0.0)
    
    cv_it_metrics['jaccard_micro'] = np.append(cv_it_metrics['jaccard_micro'], 0.0)
    cv_it_metrics['jaccard_macro'] = np.append(cv_it_metrics['jaccard_macro'], 0.0)
    cv_it_metrics['jaccard_weighted'] = np.append(cv_it_metrics['jaccard_weighted'], 0.0)
    

def init_calib_cv_result_mean_metric():
    metric_name = f'{ref_metric}'
    mean_metric_name = f'mean_{metric_name}'
    std_metric_name = f'std_{metric_name}'

    calib_cv_result_mean_metric = {
        'params': np.empty([ 0 ], dtype=object),
        mean_metric_name: np.empty([ 0 ]),
        std_metric_name: np.empty([ 0 ])
    }

    return calib_cv_result_mean_metric

def init_cv_result_mean_metrics():
    cv_result_mean_metrics = {
        'mean_accuracy': np.empty([ 0 ]),
        'std_accuracy': np.empty([ 0 ]),
        'ci_lower_accuracy': np.empty([ 0 ]),
        'ci_upper_accuracy': np.empty([ 0 ]),
        
        'mean_balanced_accuracy': np.empty([ 0 ]),
        'std_balanced_accuracy': np.empty([ 0 ]),
        'ci_lower_balanced_accuracy': np.empty([ 0 ]),
        'ci_upper_balanced_accuracy': np.empty([ 0 ]),
        
        'mean_f1_fail': np.empty([ 0 ]),
        'std_f1_fail': np.empty([ 0 ]),
        'ci_lower_f1_fail': np.empty([ 0 ]),
        'ci_upper_f1_fail': np.empty([ 0 ]),
        
        'mean_f1_success': np.empty([ 0 ]),
        'std_f1_success': np.empty([ 0 ]),
        'ci_lower_f1_success': np.empty([ 0 ]),
        'ci_upper_f1_success': np.empty([ 0 ]),
        
        'mean_f1_micro': np.empty([ 0 ]),
        'std_f1_micro': np.empty([ 0 ]),
        'ci_lower_f1_micro': np.empty([ 0 ]),
        'ci_upper_f1_micro': np.empty([ 0 ]),
        
        'mean_f1_macro': np.empty([ 0 ]),
        'std_f1_macro': np.empty([ 0 ]),
        'ci_lower_f1_macro': np.empty([ 0 ]),
        'ci_upper_f1_macro': np.empty([ 0 ]),
        
        'mean_f1_weighted': np.empty([ 0 ]),
        'std_f1_weighted': np.empty([ 0 ]),
        'ci_lower_f1_weighted': np.empty([ 0 ]),
        'ci_upper_f1_weighted': np.empty([ 0 ]),
        
        'mean_precision_fail': np.empty([ 0 ]),
        'std_precision_fail': np.empty([ 0 ]),
        'ci_lower_precision_fail': np.empty([ 0 ]),
        'ci_upper_precision_fail': np.empty([ 0 ]),
        
        'mean_precision_success': np.empty([ 0 ]),
        'std_precision_success': np.empty([ 0 ]),
        'ci_lower_precision_success': np.empty([ 0 ]),
        'ci_upper_precision_success': np.empty([ 0 ]),
        
        'mean_precision_micro': np.empty([ 0 ]),
        'std_precision_micro': np.empty([ 0 ]),
        'ci_lower_precision_micro': np.empty([ 0 ]),
        'ci_upper_precision_micro': np.empty([ 0 ]),
        
        'mean_precision_macro': np.empty([ 0 ]),
        'std_precision_macro': np.empty([ 0 ]),
        'ci_lower_precision_macro': np.empty([ 0 ]),
        'ci_upper_precision_macro': np.empty([ 0 ]),
        
        'mean_precision_weighted': np.empty([ 0 ]),
        'std_precision_weighted': np.empty([ 0 ]),
        'ci_lower_precision_weighted': np.empty([ 0 ]),
        'ci_upper_precision_weighted': np.empty([ 0 ]),
        
        'mean_recall_fail': np.empty([ 0 ]),
        'std_recall_fail': np.empty([ 0 ]),
        'ci_lower_recall_fail': np.empty([ 0 ]),
        'ci_upper_recall_fail': np.empty([ 0 ]),
        
        'mean_recall_success': np.empty([ 0 ]),
        'std_recall_success': np.empty([ 0 ]),
        'ci_lower_recall_success': np.empty([ 0 ]),
        'ci_upper_recall_success': np.empty([ 0 ]),
        
        'mean_recall_micro': np.empty([ 0 ]),
        'std_recall_micro': np.empty([ 0 ]),
        'ci_lower_recall_micro': np.empty([ 0 ]),
        'ci_upper_recall_micro': np.empty([ 0 ]),
        
        'mean_recall_macro': np.empty([ 0 ]),
        'std_recall_macro': np.empty([ 0 ]),
        'ci_lower_recall_macro': np.empty([ 0 ]),
        'ci_upper_recall_macro': np.empty([ 0 ]),
        
        'mean_recall_weighted': np.empty([ 0 ]),
        'std_recall_weighted': np.empty([ 0 ]),
        'ci_lower_recall_weighted': np.empty([ 0 ]),
        'ci_upper_recall_weighted': np.empty([ 0 ]),
        
        'mean_jaccard_fail': np.empty([ 0 ]),
        'std_jaccard_fail': np.empty([ 0 ]),
        'ci_lower_jaccard_fail': np.empty([ 0 ]),
        'ci_upper_jaccard_fail': np.empty([ 0 ]),
        
        'mean_jaccard_success': np.empty([ 0 ]),
        'std_jaccard_success': np.empty([ 0 ]),
        'ci_lower_jaccard_success': np.empty([ 0 ]),
        'ci_upper_jaccard_success': np.empty([ 0 ]),
        
        'mean_jaccard_micro': np.empty([ 0 ]),
        'std_jaccard_micro': np.empty([ 0 ]),
        'ci_lower_jaccard_micro': np.empty([ 0 ]),
        'ci_upper_jaccard_micro': np.empty([ 0 ]),
        
        'mean_jaccard_macro': np.empty([ 0 ]),
        'std_jaccard_macro': np.empty([ 0 ]),
        'ci_lower_jaccard_macro': np.empty([ 0 ]),
        'ci_upper_jaccard_macro': np.empty([ 0 ]),
        
        'mean_jaccard_weighted': np.empty([ 0 ]),
        'std_jaccard_weighted': np.empty([ 0 ]),
        'ci_lower_jaccard_weighted': np.empty([ 0 ]),
        'ci_upper_jaccard_weighted': np.empty([ 0 ]),
    }
    
    return cv_result_mean_metrics

def calculate_calib_mean_metric(calib_cv_result_mean_metric, calib_cv_it_metric):
    metric_name = f'{ref_metric}'
    mean_metric_name = f'mean_{metric_name}'
    std_metric_name = f'std_{metric_name}'
    
    calib_cv_result_mean_metric[mean_metric_name] = np.append(calib_cv_result_mean_metric[mean_metric_name], np.mean(calib_cv_it_metric[metric_name], dtype=np.float64))
    calib_cv_result_mean_metric[std_metric_name] = np.append(calib_cv_result_mean_metric[std_metric_name], np.std(calib_cv_it_metric[metric_name], dtype=np.float64))

def calculate_mean_metrics(cv_result_mean_metrics, cv_it_metrics):
    cv_result_mean_metrics['mean_accuracy'] = np.append(cv_result_mean_metrics['mean_accuracy'], np.mean(cv_it_metrics['accuracy'], dtype=np.float64))
    cv_result_mean_metrics['std_accuracy'] = np.append(cv_result_mean_metrics['std_accuracy'], np.std(cv_it_metrics['accuracy'], dtype=np.float64))
    ci_accuracy = calculate_confidence_interval(values=cv_it_metrics['accuracy'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_accuracy'] = np.append(cv_result_mean_metrics['ci_lower_accuracy'], ci_accuracy[0])
    cv_result_mean_metrics['ci_upper_accuracy'] = np.append(cv_result_mean_metrics['ci_upper_accuracy'], ci_accuracy[1])
    
    cv_result_mean_metrics['mean_balanced_accuracy'] = np.append(cv_result_mean_metrics['mean_balanced_accuracy'], np.mean(cv_it_metrics['balanced_accuracy'], dtype=np.float64))
    cv_result_mean_metrics['std_balanced_accuracy'] = np.append(cv_result_mean_metrics['std_balanced_accuracy'], np.std(cv_it_metrics['balanced_accuracy'], dtype=np.float64))
    ci_balanced_accuracy = calculate_confidence_interval(values=cv_it_metrics['balanced_accuracy'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_balanced_accuracy'] = np.append(cv_result_mean_metrics['ci_lower_balanced_accuracy'], ci_balanced_accuracy[0])
    cv_result_mean_metrics['ci_upper_balanced_accuracy'] = np.append(cv_result_mean_metrics['ci_upper_balanced_accuracy'], ci_balanced_accuracy[1])
    
    cv_result_mean_metrics['mean_f1_fail'] = np.append(cv_result_mean_metrics['mean_f1_fail'], np.mean(cv_it_metrics['f1_fail'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_fail'] = np.append(cv_result_mean_metrics['std_f1_fail'], np.std(cv_it_metrics['f1_fail'], dtype=np.float64))
    ci_f1_fail = calculate_confidence_interval(values=cv_it_metrics['f1_fail'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_f1_fail'] = np.append(cv_result_mean_metrics['ci_lower_f1_fail'], ci_f1_fail[0])
    cv_result_mean_metrics['ci_upper_f1_fail'] = np.append(cv_result_mean_metrics['ci_upper_f1_fail'], ci_f1_fail[1])
    
    cv_result_mean_metrics['mean_f1_success'] = np.append(cv_result_mean_metrics['mean_f1_success'], np.mean(cv_it_metrics['f1_success'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_success'] = np.append(cv_result_mean_metrics['std_f1_success'], np.std(cv_it_metrics['f1_success'], dtype=np.float64))
    ci_f1_success = calculate_confidence_interval(values=cv_it_metrics['f1_success'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_f1_success'] = np.append(cv_result_mean_metrics['ci_lower_f1_success'], ci_f1_success[0])
    cv_result_mean_metrics['ci_upper_f1_success'] = np.append(cv_result_mean_metrics['ci_upper_f1_success'], ci_f1_success[1])
    
    cv_result_mean_metrics['mean_f1_micro'] = np.append(cv_result_mean_metrics['mean_f1_micro'], np.mean(cv_it_metrics['f1_micro'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_micro'] = np.append(cv_result_mean_metrics['std_f1_micro'], np.std(cv_it_metrics['f1_micro'], dtype=np.float64))
    ci_f1_micro = calculate_confidence_interval(values=cv_it_metrics['f1_micro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_f1_micro'] = np.append(cv_result_mean_metrics['ci_lower_f1_micro'], ci_f1_micro[0])
    cv_result_mean_metrics['ci_upper_f1_micro'] = np.append(cv_result_mean_metrics['ci_upper_f1_micro'], ci_f1_micro[1])
    
    cv_result_mean_metrics['mean_f1_macro'] = np.append(cv_result_mean_metrics['mean_f1_macro'], np.mean(cv_it_metrics['f1_macro'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_macro'] = np.append(cv_result_mean_metrics['std_f1_macro'], np.std(cv_it_metrics['f1_macro'], dtype=np.float64))
    ci_f1_macro = calculate_confidence_interval(values=cv_it_metrics['f1_macro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_f1_macro'] = np.append(cv_result_mean_metrics['ci_lower_f1_macro'], ci_f1_macro[0])
    cv_result_mean_metrics['ci_upper_f1_macro'] = np.append(cv_result_mean_metrics['ci_upper_f1_macro'], ci_f1_macro[1])
    
    cv_result_mean_metrics['mean_f1_weighted'] = np.append(cv_result_mean_metrics['mean_f1_weighted'], np.mean(cv_it_metrics['f1_weighted'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_weighted'] = np.append(cv_result_mean_metrics['std_f1_weighted'], np.std(cv_it_metrics['f1_weighted'], dtype=np.float64))
    ci_f1_weighted = calculate_confidence_interval(values=cv_it_metrics['f1_weighted'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_f1_weighted'] = np.append(cv_result_mean_metrics['ci_lower_f1_weighted'], ci_f1_weighted[0])
    cv_result_mean_metrics['ci_upper_f1_weighted'] = np.append(cv_result_mean_metrics['ci_upper_f1_weighted'], ci_f1_weighted[1])
    
    cv_result_mean_metrics['mean_precision_fail'] = np.append(cv_result_mean_metrics['mean_precision_fail'], np.mean(cv_it_metrics['precision_fail'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_fail'] = np.append(cv_result_mean_metrics['std_precision_fail'], np.std(cv_it_metrics['precision_fail'], dtype=np.float64))
    ci_precision_fail = calculate_confidence_interval(values=cv_it_metrics['precision_fail'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_precision_fail'] = np.append(cv_result_mean_metrics['ci_lower_precision_fail'], ci_precision_fail[0])
    cv_result_mean_metrics['ci_upper_precision_fail'] = np.append(cv_result_mean_metrics['ci_upper_precision_fail'], ci_precision_fail[1])
    
    cv_result_mean_metrics['mean_precision_success'] = np.append(cv_result_mean_metrics['mean_precision_success'], np.mean(cv_it_metrics['precision_success'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_success'] = np.append(cv_result_mean_metrics['std_precision_success'], np.std(cv_it_metrics['precision_success'], dtype=np.float64))
    ci_precision_success = calculate_confidence_interval(values=cv_it_metrics['precision_success'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_precision_success'] = np.append(cv_result_mean_metrics['ci_lower_precision_success'], ci_precision_success[0])
    cv_result_mean_metrics['ci_upper_precision_success'] = np.append(cv_result_mean_metrics['ci_upper_precision_success'], ci_precision_success[1])
    
    cv_result_mean_metrics['mean_precision_micro'] = np.append(cv_result_mean_metrics['mean_precision_micro'], np.mean(cv_it_metrics['precision_micro'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_micro'] = np.append(cv_result_mean_metrics['std_precision_micro'], np.std(cv_it_metrics['precision_micro'], dtype=np.float64))
    ci_precision_micro = calculate_confidence_interval(values=cv_it_metrics['precision_micro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_precision_micro'] = np.append(cv_result_mean_metrics['ci_lower_precision_micro'], ci_precision_micro[0])
    cv_result_mean_metrics['ci_upper_precision_micro'] = np.append(cv_result_mean_metrics['ci_upper_precision_micro'], ci_precision_micro[1])
    
    cv_result_mean_metrics['mean_precision_macro'] = np.append(cv_result_mean_metrics['mean_precision_macro'], np.mean(cv_it_metrics['precision_macro'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_macro'] = np.append(cv_result_mean_metrics['std_precision_macro'], np.std(cv_it_metrics['precision_macro'], dtype=np.float64))
    ci_precision_macro = calculate_confidence_interval(values=cv_it_metrics['precision_macro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_precision_macro'] = np.append(cv_result_mean_metrics['ci_lower_precision_macro'], ci_precision_macro[0])
    cv_result_mean_metrics['ci_upper_precision_macro'] = np.append(cv_result_mean_metrics['ci_upper_precision_macro'], ci_precision_macro[1])
    
    cv_result_mean_metrics['mean_precision_weighted'] = np.append(cv_result_mean_metrics['mean_precision_weighted'], np.mean(cv_it_metrics['precision_weighted'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_weighted'] = np.append(cv_result_mean_metrics['std_precision_weighted'], np.std(cv_it_metrics['precision_weighted'], dtype=np.float64))
    ci_precision_weighted = calculate_confidence_interval(values=cv_it_metrics['precision_weighted'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_precision_weighted'] = np.append(cv_result_mean_metrics['ci_lower_precision_weighted'], ci_precision_weighted[0])
    cv_result_mean_metrics['ci_upper_precision_weighted'] = np.append(cv_result_mean_metrics['ci_upper_precision_weighted'], ci_precision_weighted[1])
    
    cv_result_mean_metrics['mean_recall_fail'] = np.append(cv_result_mean_metrics['mean_recall_fail'], np.mean(cv_it_metrics['recall_fail'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_fail'] = np.append(cv_result_mean_metrics['std_recall_fail'], np.std(cv_it_metrics['recall_fail'], dtype=np.float64))
    ci_recall_fail = calculate_confidence_interval(values=cv_it_metrics['recall_fail'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_recall_fail'] = np.append(cv_result_mean_metrics['ci_lower_recall_fail'], ci_recall_fail[0])
    cv_result_mean_metrics['ci_upper_recall_fail'] = np.append(cv_result_mean_metrics['ci_upper_recall_fail'], ci_recall_fail[1])
    
    cv_result_mean_metrics['mean_recall_success'] = np.append(cv_result_mean_metrics['mean_recall_success'], np.mean(cv_it_metrics['recall_success'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_success'] = np.append(cv_result_mean_metrics['std_recall_success'], np.std(cv_it_metrics['recall_success'], dtype=np.float64))
    ci_recall_success = calculate_confidence_interval(values=cv_it_metrics['recall_success'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_recall_success'] = np.append(cv_result_mean_metrics['ci_lower_recall_success'], ci_recall_success[0])
    cv_result_mean_metrics['ci_upper_recall_success'] = np.append(cv_result_mean_metrics['ci_upper_recall_success'], ci_recall_success[1])
    
    cv_result_mean_metrics['mean_recall_micro'] = np.append(cv_result_mean_metrics['mean_recall_micro'], np.mean(cv_it_metrics['recall_micro'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_micro'] = np.append(cv_result_mean_metrics['std_recall_micro'], np.std(cv_it_metrics['recall_micro'], dtype=np.float64))
    ci_recall_micro = calculate_confidence_interval(values=cv_it_metrics['recall_micro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_recall_micro'] = np.append(cv_result_mean_metrics['ci_lower_recall_micro'], ci_recall_micro[0])
    cv_result_mean_metrics['ci_upper_recall_micro'] = np.append(cv_result_mean_metrics['ci_upper_recall_micro'], ci_recall_micro[1])
    
    cv_result_mean_metrics['mean_recall_macro'] = np.append(cv_result_mean_metrics['mean_recall_macro'], np.mean(cv_it_metrics['recall_macro'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_macro'] = np.append(cv_result_mean_metrics['std_recall_macro'], np.std(cv_it_metrics['recall_macro'], dtype=np.float64))
    ci_recall_macro = calculate_confidence_interval(values=cv_it_metrics['recall_macro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_recall_macro'] = np.append(cv_result_mean_metrics['ci_lower_recall_macro'], ci_recall_macro[0])
    cv_result_mean_metrics['ci_upper_recall_macro'] = np.append(cv_result_mean_metrics['ci_upper_recall_macro'], ci_recall_macro[1])
    
    cv_result_mean_metrics['mean_recall_weighted'] = np.append(cv_result_mean_metrics['mean_recall_weighted'], np.mean(cv_it_metrics['recall_weighted'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_weighted'] = np.append(cv_result_mean_metrics['std_recall_weighted'], np.std(cv_it_metrics['recall_weighted'], dtype=np.float64))
    ci_recall_weighted = calculate_confidence_interval(values=cv_it_metrics['recall_weighted'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_recall_weighted'] = np.append(cv_result_mean_metrics['ci_lower_recall_weighted'], ci_recall_weighted[0])
    cv_result_mean_metrics['ci_upper_recall_weighted'] = np.append(cv_result_mean_metrics['ci_upper_recall_weighted'], ci_recall_weighted[1])
    
    cv_result_mean_metrics['mean_jaccard_fail'] = np.append(cv_result_mean_metrics['mean_jaccard_fail'], np.mean(cv_it_metrics['jaccard_fail'], dtype=np.float64))
    cv_result_mean_metrics['std_jaccard_fail'] = np.append(cv_result_mean_metrics['std_jaccard_fail'], np.std(cv_it_metrics['jaccard_fail'], dtype=np.float64))
    ci_jaccard_fail = calculate_confidence_interval(values=cv_it_metrics['jaccard_fail'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_jaccard_fail'] = np.append(cv_result_mean_metrics['ci_lower_jaccard_fail'], ci_jaccard_fail[0])
    cv_result_mean_metrics['ci_upper_jaccard_fail'] = np.append(cv_result_mean_metrics['ci_upper_jaccard_fail'], ci_jaccard_fail[1])
    
    cv_result_mean_metrics['mean_jaccard_success'] = np.append(cv_result_mean_metrics['mean_jaccard_success'], np.mean(cv_it_metrics['jaccard_success'], dtype=np.float64))
    cv_result_mean_metrics['std_jaccard_success'] = np.append(cv_result_mean_metrics['std_jaccard_success'], np.std(cv_it_metrics['jaccard_success'], dtype=np.float64))
    ci_jaccard_success = calculate_confidence_interval(values=cv_it_metrics['jaccard_success'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_jaccard_success'] = np.append(cv_result_mean_metrics['ci_lower_jaccard_success'], ci_jaccard_success[0])
    cv_result_mean_metrics['ci_upper_jaccard_success'] = np.append(cv_result_mean_metrics['ci_upper_jaccard_success'], ci_jaccard_success[1])
    
    cv_result_mean_metrics['mean_jaccard_micro'] = np.append(cv_result_mean_metrics['mean_jaccard_micro'], np.mean(cv_it_metrics['jaccard_micro'], dtype=np.float64))
    cv_result_mean_metrics['std_jaccard_micro'] = np.append(cv_result_mean_metrics['std_jaccard_micro'], np.std(cv_it_metrics['jaccard_micro'], dtype=np.float64))
    ci_jaccard_micro = calculate_confidence_interval(values=cv_it_metrics['jaccard_micro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_jaccard_micro'] = np.append(cv_result_mean_metrics['ci_lower_jaccard_micro'], ci_jaccard_micro[0])
    cv_result_mean_metrics['ci_upper_jaccard_micro'] = np.append(cv_result_mean_metrics['ci_upper_jaccard_micro'], ci_jaccard_micro[1])
    
    cv_result_mean_metrics['mean_jaccard_macro'] = np.append(cv_result_mean_metrics['mean_jaccard_macro'], np.mean(cv_it_metrics['jaccard_macro'], dtype=np.float64))
    cv_result_mean_metrics['std_jaccard_macro'] = np.append(cv_result_mean_metrics['std_jaccard_macro'], np.std(cv_it_metrics['jaccard_macro'], dtype=np.float64))
    ci_jaccard_macro = calculate_confidence_interval(values=cv_it_metrics['jaccard_macro'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_jaccard_macro'] = np.append(cv_result_mean_metrics['ci_lower_jaccard_macro'], ci_jaccard_macro[0])
    cv_result_mean_metrics['ci_upper_jaccard_macro'] = np.append(cv_result_mean_metrics['ci_upper_jaccard_macro'], ci_jaccard_macro[1])
    
    cv_result_mean_metrics['mean_jaccard_weighted'] = np.append(cv_result_mean_metrics['mean_jaccard_weighted'], np.mean(cv_it_metrics['jaccard_weighted'], dtype=np.float64))
    cv_result_mean_metrics['std_jaccard_weighted'] = np.append(cv_result_mean_metrics['std_jaccard_weighted'], np.std(cv_it_metrics['jaccard_weighted'], dtype=np.float64))
    ci_jaccard_weighted = calculate_confidence_interval(values=cv_it_metrics['jaccard_weighted'], p_alpha=0.05)
    cv_result_mean_metrics['ci_lower_jaccard_weighted'] = np.append(cv_result_mean_metrics['ci_lower_jaccard_weighted'], ci_jaccard_weighted[0])
    cv_result_mean_metrics['ci_upper_jaccard_weighted'] = np.append(cv_result_mean_metrics['ci_upper_jaccard_weighted'], ci_jaccard_weighted[1])
    
metric_name = f'{ref_metric}'
mean_metric_name = f'mean_{metric_name}'
std_metric_name = f'std_{metric_name}'

for model_name, model_parameters in models_parameters.items():
    logging.info(f'>>>>>>>>>> PROCESSING ALGORITHM {model_name}.')

    cv_test_results = init_cv_result_mean_metrics()
    jt_test_results = init_cv_result_mean_metrics()
    
    test_cv_it_metrics = init_cv_it_metrics()
    test_jt_it_metrics = init_cv_it_metrics()

    best_model_params = None
    best_model_score = -np.inf

    ########## OUTER CV ##########
    for i in range(k_hout):
        try:

            # Faz uma cópia dos conjuntos originais para que nenhum algoritmo usado adiante os altere.
            X_tuni_it = copy.deepcopy(X_tuni[i])
            y_tuni_it = copy.deepcopy(y_tuni[i])
            groups_tuni_it = copy.deepcopy(groups_tuni[i])
            X_hout_it = copy.deepcopy(X_hout[i])
            y_hout_it = copy.deepcopy(y_hout[i])
            
            
            ########## TUNING ##########
            
            # datetime object containing current date and time
            now_model_ini = datetime.now()
            dt_model_ini_string = now_model_ini.strftime("%d/%m/%Y %H:%M:%S")
            logging.info(f'>>>>> HOLDOUT {i}: Start of algorithm hyperparameter tuning: {dt_model_ini_string}. <<<<<')
            
            cv_inner = LeaveOneGroupOut()
            clf = copy.deepcopy(model_parameters['model'])
            
            pipe = Pipeline(
                [
                    ('impute', 'passthrough'),
                    ('balance', 'passthrough'),
                    ('scale', 'passthrough'),
                    ('select', 'passthrough'),
                    ("transform", 'passthrough'),
                    ("classify", clf),
                ]
            )
            
            grid_search = GridSearchCV(estimator=pipe,
                                       param_grid=model_parameters['params'],
                                       cv=cv_inner,
                                       scoring=scoring,
                                       refit='f1_weighted_scorer',
                                       n_jobs=n_jobs)
            
            grid_search.fit(X_tuni_it, y_tuni_it, groups=groups_tuni_it)

            # Gera dataframe de todos os resultados obtidos para cada combinação de parâmetros 
            gridsearchcv_results = pd.DataFrame(grid_search.cv_results_)

            gridsearchcv_results = gridsearchcv_results.sort_values(by=['rank_test_f1_weighted_scorer'])#'f1_weighted_scorer'

            gridsearchcv_results.to_csv(f'{output_folder}/gridsearchcv_results_{dt_string}_{dataset}_{model_name}_hout_{i}.csv', index=False)

            # datetime object containing current date and time
            now_model_fin = datetime.now()
            dt_model_fin_string = now_model_fin.strftime("%d/%m/%Y %H:%M:%S")
            logging.info(f'>>>>> HOLDOUT {i}: End of algorithm hyperparameter tuning: {dt_model_fin_string}. <<<<<')
            
            ########## TEST ##########

            # Linha com o melhor resultado proveniente do GridSearchCV
            best_gridsearchcv_result = gridsearchcv_results.head(1)
            
            test_cv_it_metrics['params'] = np.append(test_cv_it_metrics['params'], best_gridsearchcv_result['params'])
            test_jt_it_metrics['params'] = np.append(test_jt_it_metrics['params'], best_gridsearchcv_result['params'])
            try:
                
                y_test_pred = grid_search.predict(X_hout_it)
                y_test_score = grid_search.predict_proba(X_hout_it)
                
                calculate_metrics(test_cv_it_metrics, y_hout_it, y_test_pred, y_test_score)

                y_jt_pred = grid_search.predict(X_val)
                y_jt_score = grid_search.predict_proba(X_val)
                calculate_metrics(test_jt_it_metrics, y_val, y_jt_pred, y_jt_score)

            except Exception as e:
                logging.error(f'An error occurred. {e}')
                if len(test_cv_it_metrics[metric_name]) <= i:
                    fill_metrics(test_cv_it_metrics)
        except Exception as e:
                print(f"\n[!!!] ERRO FATAL NO FOLD {i}: {e}\n") # <--- ADICIONE ISTO
                import traceback
                traceback.print_exc() # <--- ISTO VAI MOSTRAR A LINHA EXATA DO ERRO
                logging.error(f'An error occurred. {e}')
                if len(test_cv_it_metrics[metric_name]) <= i:
                    fill_metrics(test_cv_it_metrics)

    # print(test_cv_it_metrics)
    
    df_test_cv_it_metrics = pd.DataFrame(test_cv_it_metrics)
    df_test_cv_it_metrics.to_csv(f'{output_folder}/cv_test_it_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)
    
    df_test_jt_it_metrics = pd.DataFrame(test_jt_it_metrics)
    df_test_jt_it_metrics.to_csv(f'{output_folder}/cv_test_jt_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)

    calculate_mean_metrics(cv_test_results, test_cv_it_metrics)
    
    df_cv_test_results = pd.DataFrame(cv_test_results)
    df_cv_test_results.to_csv(f'{output_folder}/cv_test_mean_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)

    calculate_mean_metrics(jt_test_results, test_jt_it_metrics)
    
    df_jt_test_results = pd.DataFrame(jt_test_results)
    df_jt_test_results.to_csv(f'{output_folder}/jt_test_mean_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)

    # datetime object containing current date and time
now = datetime.now()
dt_script_end_string = now.strftime(r'%d/%m/%Y %H:%M:%S')
dt_script_end_string

logging.info(f'>>>>>>>>>>>>>>> END OF SCRIPT EXECUTION: {dt_script_end_string}. <<<<<<<<<<<<<<<')
logging.shutdown()