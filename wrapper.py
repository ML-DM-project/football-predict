import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from typing import List, Literal


def process_data(data_path: str,
                 list_column: List[str],
                 fill_na: Literal[0, "median", "mean"],
                 scaler: Literal["standard", "minmax"]):
    df = pd.read_csv(data_path, usecols=list_column)

    y1 = df.iloc[:, -3]   # homeScoreCurrent
    y2 = df.iloc[:, -1]   # awayScoreCurrent
    y = pd.concat([y1, y2], axis=1)

    conditions = [
        (y['homeScoreCurrent'] < y['awayScoreCurrent']),
        (y['homeScoreCurrent'] == y['awayScoreCurrent']),
        (y['homeScoreCurrent'] > y['awayScoreCurrent'])
    ]
    values = ["Lose", "Draw", "Win"]
    y = pd.DataFrame(np.select(conditions, values))

    X = df.drop(["id", "awayScoreCurrent", "homeScoreCurrent"], axis=1)

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

    if scaler == "standard":
        sc = StandardScaler()
    else:
        sc = MinMaxScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return sc, X_train, X_test, y_train, y_test
