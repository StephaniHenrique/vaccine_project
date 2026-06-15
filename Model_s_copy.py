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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.svm import SVC

matplotlib.rcParams.update({'font.size': 15})
exp_id = 'PUBLIC_HAI_HOT_ENCODING' #MUDAR
script_path = 'New_PUBLIC_DATA' #MUDAR

output_folder = f'./Results/{script_path}/{exp_id}'
dataset = 'FCS_PUBLIC_efeito_calculado_hot_encoding' #MUDAR
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
reducer = None

development_test = False 

n_jobs = 2

pca_n_components = 11 #TRATAR OS DADOS PRIMEIROS
reducer_option = 0 # 0 = no dimensionality reduction / 1 = PCA / 2 = LDA
sample_balancing_option = 0 # 0 = no balancing / 1 = undersampling / 2 = oversampling

random_state_hout = [20, 40, 60, 80, 100]
k_cv=3

random_state_model = 42
random_state_inner_cv = 42

ref_metric = 'f1_weighted'
ref_metric_2 = 'accuracy'

logging.info(f'>>>>>>>>>> PIPELINE: Code/{script_path} <<<<<<<<<<<<<<<')

logging.info(f'>>>>>>>>>> PIPELINE PARAMETERS:')
logging.info(f'\t dataset = {dataset}')
logging.info(f'\t development_test = {development_test}')
logging.info(f'\t n_jobs = {n_jobs}')
logging.info(f'\t pca_n_components = {pca_n_components}')
logging.info(f'\t reducer_option = {reducer_option}')
logging.info(f'\t random_state_hout = {random_state_hout}')
logging.info(f'\t random_state_model = {random_state_model}')
logging.info(f'\t ref_metric = {ref_metric}')
logging.info(f'\t ref_metric_2 = {ref_metric_2}')
logging.info(f'\t sample_balancing_option = {sample_balancing_option}')

scoring = {
    'f1_weighted_scorer': make_scorer(f1_score, average='weighted', zero_division=0),
    'roc_auc_scorer': make_scorer(roc_auc_score, response_method='predict_proba'),
    'accuracy_scorer': make_scorer(accuracy_score)
}

dataset_file = pd.read_csv(f"./{dataset}.csv")

dataset_q_rows = len(dataset_file)
dataset_q_features = len(dataset_file.columns) - 1
dataset_file['Study_ID'] = dataset_file['Participant ID'].apply(lambda x: int(x.split('.')[-1]))
print(dataset_file['Study_ID'].nunique())

dataset_features = dataset_file.drop(columns=['Participant ID', 'Label_HAI', 'Cluster', 'Study_ID'])
X = dataset_features.to_numpy()

n_classes = dataset_file['Label_HAI'].nunique()
y = dataset_file.loc[:, 'Label_HAI']
y = y.to_numpy()

groups = dataset_file['Study_ID'].to_numpy()

# X_val = df_val.drop('Target', axis=1).to_numpy()
# y_val = df_val.loc[:, 'Target'].to_numpy()

if sample_balancing_option == 1:
    balancer = RandomUnderSampler(random_state=random_state_model)

if sample_balancing_option == 2:
    balancer = SMOTE(random_state=random_state_model)

if reducer_option == 1:
    reducer = PCA(n_components=pca_n_components, svd_solver='full', random_state=random_state_model)

if reducer_option == 2:
    reducer = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None)


k_hout = len(dataset_file['Study_ID'].unique())

X_tuni = np.empty([ k_hout ], dtype=object)

y_tuni = np.empty([ k_hout ], dtype=object)

X_hout = np.empty([ k_hout ], dtype=object)

y_hout = np.empty([ k_hout ], dtype=object)

logo_outer = LeaveOneGroupOut()
outer_cv_folds = logo_outer.split(X, y, groups=groups)


for i, (train_index, test_index) in enumerate(outer_cv_folds):
    print(i)
    X_tuni[i] = np.array(X[train_index])
    y_tuni[i] = np.array(y[train_index])
    
    X_hout[i] = np.array(X[test_index])
    y_hout[i] = np.array(y[test_index])

    print(X_tuni[i].shape)
    print(y_tuni[i].shape)
    print(X_hout[i].shape)
    print(y_hout[i].shape)


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
    
    models_parameters = {
          'LinearDiscriminantAnalysis': { #LDA
            'model': LinearDiscriminantAnalysis(),
            'params': [
                {#Sem seleção de caracteristicas e transformação
                   'balance': [
                        None,
                        RandomUnderSampler(random_state=random_state_model),
                        SMOTE(random_state=random_state_model)
                    ],
                    'select': [ None ],
                    'transform': [ None ],
                    'classify__solver': ['svd','lsqr','eigen'],
                    'classify__shrinkage': [None,'auto',0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                    'classify__n_components': [1,None],  # 'n_components' = None => 'n_components' = min(n_classes - 1, n_features) => Se n_features >= 2 e n_classes = 3, então 'n_components' = 2 = None
                    'classify__store_covariance': [True],
                    'classify__tol': [1.0e-4] 
                },
            ]
        },
        'LogisticRegression': { #LogisticRegression
            'model': LogisticRegression(),
            'params': [
                {#Sem seleção de caracteristicas e transformação
                    'impute': [ None, SimpleImputer(strategy='median') ],
                    'balance': [
                        None,
                        RandomUnderSampler(random_state=random_state_model),
                        SMOTE(random_state=random_state_model)
                    ],
                    'select': [ SelectKBest(mutual_info_classif) ],
                    'select__k': [5, 7, 9],
                    'transform': [ None ], 
                    'classify__penalty': ['l2'],
                    'classify__tol': [1.0e-8,1.0e-6,1.0e-4],
                    'classify__C': [0.01,0.1,1.0,10.0,100.0],
                    'classify__random_state': [random_state_model],
                    'classify__solver': ['lbfgs'],
                    'classify__max_iter': [100,200,500,1000,2000],
                    'classify__n_jobs': [n_jobs]
                },
            ]
        },
        'RandomForestClassifier': { #RandomForest
            'model': RandomForestClassifier(),
            'params': [
                {#Sem seleção de caracteristicas e transformação
                   'impute': [ None, SimpleImputer(strategy='median') ],
                   'balance': [
                        None,
                        RandomUnderSampler(random_state=random_state_model),
                        SMOTE(random_state=random_state_model)
                    ],
                    'select': [ None ],
                    'transform': [ None ], 
                    'classify__n_estimators': [10,50,100,250,500,750,1000],
                    'classify__criterion': ['gini','entropy','log_loss'],
                    'classify__max_features': ['sqrt','log2',None],
                    'classify__n_jobs': [n_jobs],
                    'classify__ccp_alpha': [0.0,0.02,0.04,0.08],
                    'classify__random_state': [random_state_model]
                },
            ]
        },
        'SVC': { #SVC
            'model': SVC(),
            'params': [
                {#Sem seleção de caracteristicas e transformação
                   'balance': [
                        None,
                        RandomUnderSampler(random_state=random_state_model),
                        SMOTE(random_state=random_state_model)
                    ],
                    'select': [ None ],
                    'transform': [ None ], 
                    'classify__C': [0.001,0.01,0.1,1.0,10.0,100.0],
                    'classify__kernel': ['linear','poly','rbf','sigmoid'],
                    'classify__degree': [3, 5, 10],
                    'classify__gamma': ['auto','scale',0.001,0.01,0.1],
                    'classify__probability': [True],
                    'classify__tol': [1.0e-4,1.0e-3],
                    'classify__max_iter': [1500,2000,2500,3000],
                    'classify__random_state': [random_state_model]
                },
            ]
        },
        'MLPClassifier': { #MLP
            'model': MLPClassifier(),
            'params': [
                {#Sem seleção de caracteristicas e transformação
                   'balance': [
                        None,
                        RandomUnderSampler(random_state=random_state_model),
                        SMOTE(random_state=random_state_model)
                    ],
                    'select': [ None ],
                    'transform': [ None ], 
                    'classify__hidden_layer_sizes': [
                        (n_neurons_r1_l1,n_neurons_r1_l2,n_neurons_r1_l3,),(n_neurons_r2_l1,n_neurons_r2_l2,n_neurons_r2_l3,),(n_neurons_r3_l1,n_neurons_r3_l2,n_neurons_r3_l3,)
                    ],
                    'classify__activation': ['identity','logistic','tanh','relu'],
                    'classify__solver': ['lbfgs','sgd','adam'],
                    'classify__alpha': [1.0e-5,1.0e-4,1.0e-3],
                    'classify__learning_rate': ['constant','invscaling','adaptive'],
                    'classify__learning_rate_init': [0.001,0.01,0.1,1.0],
                    'classify__max_iter': [1500],
                    'classify__random_state': [random_state_model],
                    'classify__tol': [1.0e-5,1.0e-4],
                    'classify__n_iter_no_change': [10,30,50]
                },
            ]
        },
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
        'f1_weighted': np.empty([ 0 ]),
        'precision_fail': np.empty([ 0 ]),
        'precision_success': np.empty([ 0 ]),
        'precision_weighted': np.empty([ 0 ]),
        'recall_fail': np.empty([ 0 ]),
        'recall_success': np.empty([ 0 ]),
        'recall_weighted': np.empty([ 0 ]),
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
    
    cv_it_metrics['f1_weighted'] = np.append(cv_it_metrics['f1_weighted'], f1_score(y, y_pred, average='weighted', zero_division=0))
    
    pre_fail, pre_success = precision_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
    
    cv_it_metrics['precision_fail'] = np.append(cv_it_metrics['precision_fail'], pre_fail)
    cv_it_metrics['precision_success'] = np.append(cv_it_metrics['precision_success'], pre_success)
    
    cv_it_metrics['precision_weighted'] = np.append(cv_it_metrics['precision_weighted'], precision_score(y, y_pred, average='weighted', zero_division=0))
    
    rec_fail, rec_success = recall_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
    
    cv_it_metrics['recall_fail'] = np.append(cv_it_metrics['recall_fail'], rec_fail)
    cv_it_metrics['recall_success'] = np.append(cv_it_metrics['recall_success'], rec_success)

    cv_it_metrics['recall_weighted'] = np.append(cv_it_metrics['recall_weighted'], recall_score(y, y_pred, average='weighted', zero_division=0))
    

def fill_metrics(cv_it_metrics):
    cv_it_metrics['accuracy'] = np.append(cv_it_metrics['accuracy'], 0.0)
    cv_it_metrics['balanced_accuracy'] = np.append(cv_it_metrics['balanced_accuracy'], 0.0)
    
    cv_it_metrics['f1_fail'] = np.append(cv_it_metrics['f1_fail'], 0.0)
    cv_it_metrics['f1_success'] = np.append(cv_it_metrics['f1_success'], 0.0)
    
    cv_it_metrics['f1_weighted'] = np.append(cv_it_metrics['f1_weighted'], 0.0)
    
    cv_it_metrics['precision_fail'] = np.append(cv_it_metrics['precision_fail'], 0.0)
    cv_it_metrics['precision_success'] = np.append(cv_it_metrics['precision_success'], 0.0)
    
    cv_it_metrics['precision_weighted'] = np.append(cv_it_metrics['precision_weighted'], 0.0)
    
    cv_it_metrics['recall_fail'] = np.append(cv_it_metrics['recall_fail'], 0.0)
    cv_it_metrics['recall_success'] = np.append(cv_it_metrics['recall_success'], 0.0)
    
    cv_it_metrics['recall_weighted'] = np.append(cv_it_metrics['recall_weighted'], 0.0)
    
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
        
        'mean_balanced_accuracy': np.empty([ 0 ]),
        'std_balanced_accuracy': np.empty([ 0 ]),
        
        'mean_f1_fail': np.empty([ 0 ]),
        'std_f1_fail': np.empty([ 0 ]),
        
        'mean_f1_success': np.empty([ 0 ]),
        'std_f1_success': np.empty([ 0 ]),
        
        
        'mean_f1_weighted': np.empty([ 0 ]),
        'std_f1_weighted': np.empty([ 0 ]),
        
        'mean_precision_fail': np.empty([ 0 ]),
        'std_precision_fail': np.empty([ 0 ]),
        
        'mean_precision_success': np.empty([ 0 ]),
        'std_precision_success': np.empty([ 0 ]),

        'mean_precision_weighted': np.empty([ 0 ]),
        'std_precision_weighted': np.empty([ 0 ]),
        
        'mean_recall_fail': np.empty([ 0 ]),
        'std_recall_fail': np.empty([ 0 ]),
        
        'mean_recall_success': np.empty([ 0 ]),
        'std_recall_success': np.empty([ 0 ]),
        
        'mean_recall_weighted': np.empty([ 0 ]),
        'std_recall_weighted': np.empty([ 0 ]),
        
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

    cv_result_mean_metrics['mean_balanced_accuracy'] = np.append(cv_result_mean_metrics['mean_balanced_accuracy'], np.mean(cv_it_metrics['balanced_accuracy'], dtype=np.float64))
    cv_result_mean_metrics['std_balanced_accuracy'] = np.append(cv_result_mean_metrics['std_balanced_accuracy'], np.std(cv_it_metrics['balanced_accuracy'], dtype=np.float64))

    cv_result_mean_metrics['mean_f1_fail'] = np.append(cv_result_mean_metrics['mean_f1_fail'], np.mean(cv_it_metrics['f1_fail'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_fail'] = np.append(cv_result_mean_metrics['std_f1_fail'], np.std(cv_it_metrics['f1_fail'], dtype=np.float64))
    
    cv_result_mean_metrics['mean_f1_success'] = np.append(cv_result_mean_metrics['mean_f1_success'], np.mean(cv_it_metrics['f1_success'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_success'] = np.append(cv_result_mean_metrics['std_f1_success'], np.std(cv_it_metrics['f1_success'], dtype=np.float64))
    
    cv_result_mean_metrics['mean_f1_weighted'] = np.append(cv_result_mean_metrics['mean_f1_weighted'], np.mean(cv_it_metrics['f1_weighted'], dtype=np.float64))
    cv_result_mean_metrics['std_f1_weighted'] = np.append(cv_result_mean_metrics['std_f1_weighted'], np.std(cv_it_metrics['f1_weighted'], dtype=np.float64))
    
    cv_result_mean_metrics['mean_precision_fail'] = np.append(cv_result_mean_metrics['mean_precision_fail'], np.mean(cv_it_metrics['precision_fail'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_fail'] = np.append(cv_result_mean_metrics['std_precision_fail'], np.std(cv_it_metrics['precision_fail'], dtype=np.float64))
    
    cv_result_mean_metrics['mean_precision_success'] = np.append(cv_result_mean_metrics['mean_precision_success'], np.mean(cv_it_metrics['precision_success'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_success'] = np.append(cv_result_mean_metrics['std_precision_success'], np.std(cv_it_metrics['precision_success'], dtype=np.float64))
   
    cv_result_mean_metrics['mean_precision_weighted'] = np.append(cv_result_mean_metrics['mean_precision_weighted'], np.mean(cv_it_metrics['precision_weighted'], dtype=np.float64))
    cv_result_mean_metrics['std_precision_weighted'] = np.append(cv_result_mean_metrics['std_precision_weighted'], np.std(cv_it_metrics['precision_weighted'], dtype=np.float64))
    
    cv_result_mean_metrics['mean_recall_fail'] = np.append(cv_result_mean_metrics['mean_recall_fail'], np.mean(cv_it_metrics['recall_fail'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_fail'] = np.append(cv_result_mean_metrics['std_recall_fail'], np.std(cv_it_metrics['recall_fail'], dtype=np.float64))
    
    cv_result_mean_metrics['mean_recall_success'] = np.append(cv_result_mean_metrics['mean_recall_success'], np.mean(cv_it_metrics['recall_success'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_success'] = np.append(cv_result_mean_metrics['std_recall_success'], np.std(cv_it_metrics['recall_success'], dtype=np.float64))
    
    cv_result_mean_metrics['mean_recall_weighted'] = np.append(cv_result_mean_metrics['mean_recall_weighted'], np.mean(cv_it_metrics['recall_weighted'], dtype=np.float64))
    cv_result_mean_metrics['std_recall_weighted'] = np.append(cv_result_mean_metrics['std_recall_weighted'], np.std(cv_it_metrics['recall_weighted'], dtype=np.float64))
    
metric_name = f'{ref_metric}'
mean_metric_name = f'mean_{metric_name}'
std_metric_name = f'std_{metric_name}'

for model_name, model_parameters in models_parameters.items():
    logging.info(f'>>>>>>>>>> PROCESSING ALGORITHM {model_name}.')

    cv_test_results = init_cv_result_mean_metrics()
    # jt_test_results = init_cv_result_mean_metrics()
    
    test_cv_it_metrics = init_cv_it_metrics()
    # test_jt_it_metrics = init_cv_it_metrics()

    best_model_params = None
    best_model_score = -np.inf

    ########## OUTER CV ##########
    for i in range(k_hout):
        try:
            # print(X_tuni[i])
            # print(y_tuni[i])
            # print(X_hout[i])
            # print(y_hout[i])

            # Faz uma cópia dos conjuntos originais para que nenhum algoritmo usado adiante os altere.
            X_tuni_it = copy.deepcopy(X_tuni[i])
            y_tuni_it = copy.deepcopy(y_tuni[i])
            X_hout_it = copy.deepcopy(X_hout[i])
            y_hout_it = copy.deepcopy(y_hout[i])
            
            # print(X_tuni_it)
            # print(y_tuni_it)
            # print(X_hout_it)
            # print(y_hout_it)
            
            ########## TUNING ##########
            
            # datetime object containing current date and time
            now_model_ini = datetime.now()
            dt_model_ini_string = now_model_ini.strftime("%d/%m/%Y %H:%M:%S")
            logging.info(f'>>>>> HOLDOUT {i}: Start of algorithm hyperparameter tuning: {dt_model_ini_string}. <<<<<')
            
            skf_inner = StratifiedKFold(n_splits=k_cv, shuffle=True, random_state=random_state_inner_cv)

            clf = copy.deepcopy(model_parameters['model'])
            
            pipe = Pipeline(
                [
                    ('impute', 'passthrough'),
                    ('balance', 'passthrough'),
                    ('select', 'passthrough'),
                    ("transform", StandardScaler()),
                    ("classify", clf),
                ]
            )
            
            grid_search = GridSearchCV(estimator=pipe,
                                       param_grid=model_parameters['params'],
                                       cv=skf_inner,
                                       scoring=scoring,
                                       refit='f1_weighted_scorer',
                                       n_jobs=n_jobs)
            
            grid_search.fit(X_tuni_it, y_tuni_it)

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
            # test_jt_it_metrics['params'] = np.append(test_jt_it_metrics['params'], best_gridsearchcv_result['params'])
            try:
                # Calcula as métricas

                # ------> MAIOR PONTO DE ATENÇÃO!!! Entender que modelo (treinamento) a variável grid_search contém.

                # O que deveria ser feito?
                ## Pega o conjunto X_tuni[i], y_tuni[i] original
                ## Aplica a melhor combinação de técnicas e parâmetros:
                ### Aplica o método de balanceamento 
                ### Aplica o método de redução de dimensionlidade
                ### Treina o algoritmo clf com os melhores parametros sobre X_tuni[i], y_tuni[i]
                ## Testa o classificador final resultante sobre X_hout[i]


                
                print(grid_search)
                
                y_test_pred = grid_search.predict(X_hout_it)
                y_test_score = grid_search.predict_proba(X_hout_it)
                
                calculate_metrics(test_cv_it_metrics, y_hout_it, y_test_pred, y_test_score)

            #     # y_jt_pred = grid_search.predict(X_val)
            #     # y_jt_score = grid_search.predict_proba(X_val)
            #     # calculate_metrics(test_jt_it_metrics, y_val, y_jt_pred, y_jt_score)

            except Exception as e:
                logging.error(f'An error occurred. {e}')
                if len(test_cv_it_metrics[mean_metric_name]) <= i:
                    fill_metrics(test_cv_it_metrics)
        except Exception as e:
                print(f"\n[!!!] ERRO FATAL NO FOLD {i}: {e}\n") # <--- ADICIONE ISTO
                import traceback
                traceback.print_exc() # <--- ISTO VAI MOSTRAR A LINHA EXATA DO ERRO
                logging.error(f'An error occurred. {e}')
                if len(test_cv_it_metrics[mean_metric_name]) <= i:
                    fill_metrics(test_cv_it_metrics)

    # print(test_cv_it_metrics)
    
    df_test_cv_it_metrics = pd.DataFrame(test_cv_it_metrics)
    df_test_cv_it_metrics.to_csv(f'{output_folder}/cv_test_it_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)
    
    # df_test_jt_it_metrics = pd.DataFrame(test_jt_it_metrics)
    # df_test_jt_it_metrics.to_csv(f'{output_folder}/cv_test_jt_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)

    calculate_mean_metrics(cv_test_results, test_cv_it_metrics)
    
    df_cv_test_results = pd.DataFrame(cv_test_results)
    df_cv_test_results.to_csv(f'{output_folder}/cv_test_mean_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)

    # calculate_mean_metrics(jt_test_results, test_jt_it_metrics)
    
    # df_jt_test_results = pd.DataFrame(jt_test_results)
    # df_jt_test_results.to_csv(f'{output_folder}/jt_test_mean_metrics_{dt_string}_{dataset}_{model_name}.csv', index=False)

    # datetime object containing current date and time
now = datetime.now()
dt_script_end_string = now.strftime(r'%d/%m/%Y %H:%M:%S')
dt_script_end_string

logging.info(f'>>>>>>>>>>>>>>> END OF SCRIPT EXECUTION: {dt_script_end_string}. <<<<<<<<<<<<<<<')
logging.shutdown()