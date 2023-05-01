from typing import List, Dict
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
)
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

__all__ = ["process_df", "xgbclassifier_model"]

BEST_PARAMS = dict(learning_rate=0.1, max_depth=5, n_estimators=500)
RANDOM_STATE = 38


def normalize_df(xdf: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = xdf.select_dtypes(include=["float64", "int64"]).columns
    scaler = MinMaxScaler()
    xdf[numeric_cols] = scaler.fit_transform(xdf[numeric_cols])

    return xdf


def labels_encode(xdf: pd.DataFrame) -> pd.DataFrame:
    category_cols = xdf.select_dtypes("category").columns
    le = LabelEncoder()

    for c in category_cols:
        xdf[c] = le.fit_transform(xdf[c])

    return xdf


def apply_smote(xdf: pd.DataFrame) -> pd.DataFrame:
    smote = SMOTE()
    y = xdf["class"]
    X = xdf.drop("class", axis=1)

    X, y = smote.fit_resample(X, y)

    return pd.concat([X, y], axis=1)


def split_ml_df(
    xdf: pd.DataFrame, test_size: float, random_state=RANDOM_STATE
) -> List[pd.DataFrame]:
    return train_test_split(
        xdf.drop("class", axis=1),
        xdf["class"],
        test_size=test_size,
        random_state=random_state,
    )


def process_df(xdf: pd.DataFrame) -> Dict["str", pd.DataFrame]:
    xdf = xdf.pipe(normalize_df).pipe(labels_encode).pipe(apply_smote)

    splitted = split_ml_df(xdf, 0.2)
    keys = ["X_train", "X_test", "y_train", "y_test"]

    return {k: v for [k, v] in zip(keys, splitted)}


def xgbclassifier_model(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(**{**BEST_PARAMS, "random_state": RANDOM_STATE})

    model.fit(X_train, y_train)

    return model
