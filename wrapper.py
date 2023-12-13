import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler
)
from lazypredict.Supervised import LazyClassifier

from typing import List, Literal


def process_data(data_path: str,
                 list_column: List[str],
                 fill_na: Literal[0, "median", "mean", "auto"] = "auto",
                 scaler: Literal["standard", "minmax", "maxabs", "normal", "power", "robust", "quantile"] = "standard"):
    df = pd.read_csv(data_path, usecols=list_column)

    y1 = df["homeScoreCurrent"]   # homeScoreCurrent
    y2 = df["awayScoreCurrent"]   # awayScoreCurrent
    y = pd.concat([y1, y2], axis=1)

    conditions = [
        (y['homeScoreCurrent'] < y['awayScoreCurrent']),
        (y['homeScoreCurrent'] == y['awayScoreCurrent']),
        (y['homeScoreCurrent'] > y['awayScoreCurrent'])
    ]
    values = ["Lose", "Draw", "Win"]
    y = pd.DataFrame(np.select(conditions, values))

    thisFilter = df.filter(["id", "awayScoreCurrent", "homeScoreCurrent"])
    X = df.drop(thisFilter, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    if fill_na == 0:
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)
    elif fill_na == "median":
        X_train.fillna(X_train.median(), inplace=True)
        X_test.fillna(X_test.median(), inplace=True)
    elif fill_na == "mean":
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_test.mean(), inplace=True)
    elif fill_na == "auto":   # fill based on feature engineering
        X_train['homeAvgRating'] = X_train['homeAvgRating'].fillna(X_train['homeAvgRating'].mean())
        X_train['homePoint'] = X_train['homePoint'].fillna(X_train['homePoint'].median())
        X_train['homeForm'] = X_train['homeForm'].fillna(X_train['homeForm'].mean())
        X_train['awayAvgRating'] = X_train['awayAvgRating'].fillna(X_train['awayAvgRating'].mean())
        X_train['awayPoint'] = X_train['awayPoint'].fillna(X_train['awayPoint'].median())
        X_train['awayForm'] = X_train['awayForm'].fillna(X_train['awayForm'].mean())
        
        X_test['homeAvgRating'] = X_test['homeAvgRating'].fillna(X_test['homeAvgRating'].mean())
        X_test['homePoint'] = X_test['homePoint'].fillna(X_test['homePoint'].median())
        X_test['homeForm'] = X_test['homeForm'].fillna(X_test['homeForm'].mean())
        X_test['awayAvgRating'] = X_test['awayAvgRating'].fillna(X_test['awayAvgRating'].mean())
        X_test['awayPoint'] = X_test['awayPoint'].fillna(X_test['awayPoint'].median())
        X_test['awayForm'] = X_test['awayForm'].fillna(X_test['awayForm'].mean())
        

    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "minmax":
        sc = MinMaxScaler()
    elif scaler == "maxabs":
        sc = MaxAbsScaler()
    elif scaler == "normal":
        sc = Normalizer()
    elif scaler == "power":
        sc = PowerTransformer()
    elif scaler == "robust":
        sc = RobustScaler()
    elif scaler == "quantile":
        sc = QuantileTransformer()
    else:
        raise ValueError

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return sc, X_train, X_test, y_train, y_test

def lazy_feature_selection(data_path: str,
                            list_column: List[str],
                            fill_na: Literal[0, "median", "mean", "auto"] = "auto",
                           scaler: Literal["standard", "minmax", "maxabs", "normal", "power", "robust", "quantile"] = "standard"):
    sc, X_train, X_test, y_train, y_test = process_data(data_path, list_column, fill_na, scaler)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    score, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    return score