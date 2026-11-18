from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import xgboost as xgb

from src.features.feature_engineering import FEATURE_COLUMNS


def get_model_pipeline(model_name: str, random_state: int = 42) -> Tuple[BaseEstimator, bool, Dict[str, Any]]:
    """
    Returns (estimator, requires_scaling, initial_hyperparams) for each of the 5 model families.
    """
    if model_name == "logistic_regression":
        params = {
            "C": 1.0,
            "max_iter": 500,
            "solver": "lbfgs",
            "random_state": random_state,
        }
        clf = LogisticRegression(**params)
        return clf, True, params

    elif model_name == "knn":
        params = {
            "n_neighbors": 15,
            "weights": "distance",
            "metric": "minkowski",
            "n_jobs": -1,
        }
        clf = KNeighborsClassifier(**params)
        return clf, True, params

    elif model_name == "random_forest":
        params = {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 10,
            "random_state": random_state,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)
        return clf, False, params

    elif model_name == "lightgbm":
        params = {
            "n_estimators": 100,
            "learning_rate": 0.08,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        clf = lgb.LGBMClassifier(**params)
        return clf, False, params

    elif model_name == "xgboost":
        params = {
            "n_estimators": 100,
            "learning_rate": 0.08,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        clf = xgb.XGBClassifier(**params)
        return clf, False, params

    else:
        raise ValueError(f"Unknown model name: {model_name}")
