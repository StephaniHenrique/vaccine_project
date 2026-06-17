import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)


df = pd.read_csv("./301_final_standard.csv")

TARGET = "Label_HAI"

#removing id
if "Participant ID" in df.columns:
    df = df.drop(columns=["Participant ID"])


X = df.drop(columns=[TARGET])
y = df[TARGET]


meta_cols = ['Participant ID', 'Gender', 'Label_HAI', 'Virus_A/California/7/2009', 'Virus_A/Perth/16/2009', 'Virus_A/Perth/19/2009', 'Virus_A/Victoria/361/2011', 'Virus_B/Brisbane/60/2008', 'Virus_B/Massachusetts/2/2012', 'Virus_B/Wisconsin/01/2010']

flow_columns = [c for c in X.columns if c not in meta_cols]

X_without_flow = X.drop(columns=flow_columns)

#MODELS AND EVALUATION SETUP
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=42,
    ),
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
}

def evaluate(X_data, title):

    print("=" * 60)
    print(title)
    print("=" * 60)

    for name, model in models.items():

        if name in ["Linear", "Ridge", "Lasso", "ElasticNet"]:

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ])

        else:

            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ])

        scores = cross_validate(
            pipe,
            X_data,
            y,
            cv=cv,
            scoring=scoring,
        )

        r2 = scores["test_R2"].mean()
        mae = -scores["test_MAE"].mean()

        print(
            f"{name:20s} "
            f"R² = {r2:.3f}    "
            f"MAE = {mae:.3f}"
        )


#EVALUATING

evaluate(X, "ALL FEATURES WITHOUT HAI")

# evaluate(
#     X_without_flow,
#     "SEM CITOMETRIA DE FLUXO",
# )