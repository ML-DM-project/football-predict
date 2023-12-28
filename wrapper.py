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
    elif fill_na == "auto": # fill based on feature engineering
        if 'homeAvgRating' in list_column:
            X_train['homeAvgRating'] = X_train['homeAvgRating'].fillna(X_train['homeAvgRating'].mean())
            X_test['homeAvgRating'] = X_test['homeAvgRating'].fillna(X_train['homeAvgRating'].mean())
        if 'homePoint' in list_column:
            X_train['homePoint'] = X_train['homePoint'].fillna(X_train['homePoint'].median())
            X_test['homePoint'] = X_test['homePoint'].fillna(X_train['homePoint'].median())
        if 'homeForm' in list_column:
            X_train['homeForm'] = X_train['homeForm'].fillna(X_train['homeForm'].mean())
            X_test['homeForm'] = X_test['homeForm'].fillna(X_train['homeForm'].mean())
        if 'awayAvgRating' in list_column:
            X_train['awayAvgRating'] = X_train['awayAvgRating'].fillna(X_train['awayAvgRating'].mean())
            X_test['awayAvgRating'] = X_test['awayAvgRating'].fillna(X_train['awayAvgRating'].mean())
        if 'awayPoint' in list_column:
            X_train['awayPoint'] = X_train['awayPoint'].fillna(X_train['awayPoint'].median())
            X_test['awayPoint'] = X_test['awayPoint'].fillna(X_train['awayPoint'].median())
        if 'awayForm' in list_column:
            X_train['awayForm'] = X_train['awayForm'].fillna(X_train['awayForm'].mean())
            X_test['awayForm'] = X_test['awayForm'].fillna(X_train['awayForm'].mean())
        

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

def lazy_train(X_train, X_test, y_train, y_test, 
               scaler: Literal["standard", "minmax", "maxabs", "normal", "power", "robust", "quantile"] = "quantile"):
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
    
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    score, predictions = clf.fit(X_train, X_test, y_train, y_test)
    
    return score

list_column = ['previousHomeWin', 'previousAwayWin', 'previousDraw',
       'previousManagerHomeWin', 'previousManagerAwayWin',
       'previousManagerDraw', 'homeAvgRating', 'homePosition', 'homePoint',
       'homeForm', 'awayAvgRating', 'awayPosition', 'awayPoint', 'awayForm',
       'homeExpectedGoal', 'awayExpectedGoal', 'homeExpectedAssist',
       'awayExpectedAssist', 'homeBallPosession', 'homeShotOnTarget',
       'awayShotOnTarget', 'homeShotOffTarget', 'awayShotOffTarget',
       'homeBlockedShot', 'awayBlockedShot', 'homeCorner', 'awayCorner',
       'homeOffside', 'awayOffside', 'homeYellowCard', 'awayYellowCard',
       'homeRedCard', 'awayRedCard', 'homeFreekick', 'awayFreekick',
       'homeThrowIn', 'awayThrowIn', 'homeGoalkick', 'awayGoalkick',
       'homeBigChance', 'awayBigChance', 'homeBigChanceMissed',
       'awayBigChanceMissed', 'homeHitWoodwork', 'awayHitWoodwork',
       'homeCounterAttack', 'awayCounterAttack', 'homeCounterAttackShot',
       'awayCounterAttackShot', 'homeCounterAttackGoal',
       'awayCounterAttackGoal', 'homeShotInsideBox', 'awayShotInsideBox',
       'homeGoalSave', 'awayGoalSave', 'homePass', 'awayPass',
       'homeAccuratePass', 'awayAccuratePass', 'homeLongPass', 'awayLongPass',
       'homeAccurateLongPass', 'awayAccurateLongPass', 'homeCross',
       'awayCross', 'homeAccurateCross', 'awayAccurateCross', 'homeDribble',
       'awwayDribble', 'homeSuccessfulDribble', 'awaySuccessfulDribble',
       'homePossesionLost', 'awayPossesionLost', 'homeDuelWon', 'awayDuelWon',
       'homeAerialWon', 'awayAerialWon', 'homeTackle', 'awayTackle',
       'homeInterception', 'awayInterception', 'homeClearance',
       'awayClearance', 'homeTeamId', 'awayTeamId', 'homeScorePeriod1',
       'homeScoreCurrent', 'awayScorePeriod1', 'awayScoreCurrent']

def load_fillna(data_path: str = "match_all_statistic.csv", list_column: List[str] = list_column, seed=42):
    df = pd.read_csv(data_path, usecols=list_column)
    
    # Get label
    y1 = df["homeScoreCurrent"]   # homeScoreCurrent
    y2 = df["awayScoreCurrent"]   # awayScoreCurrent
    y = pd.concat([y1, y2], axis=1)

    conditions = [
        (y['homeScoreCurrent'] < y['awayScoreCurrent']),
        (y['homeScoreCurrent'] == y['awayScoreCurrent']),
        (y['homeScoreCurrent'] > y['awayScoreCurrent'])
    ]
    values = [0, 1, 3]
    y = pd.DataFrame(np.select(conditions, values))

    thisFilter = df.filter(["id", "awayScoreCurrent", "homeScoreCurrent"])
    X = df.drop(thisFilter, axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)
    
    X_train['homeAvgRating'] = X_train['homeAvgRating'].fillna(X_train['homeAvgRating'].mean())
    X_test['homeAvgRating'] = X_test['homeAvgRating'].fillna(X_train['homeAvgRating'].mean())

    X_train['homePoint'] = X_train['homePoint'].fillna(X_train['homePoint'].median())
    X_test['homePoint'] = X_test['homePoint'].fillna(X_train['homePoint'].median())

    X_train['homeForm'] = X_train['homeForm'].fillna(X_train['homeForm'].mean())
    X_test['homeForm'] = X_test['homeForm'].fillna(X_train['homeForm'].mean())

    X_train['awayAvgRating'] = X_train['awayAvgRating'].fillna(X_train['awayAvgRating'].mean())
    X_test['awayAvgRating'] = X_test['awayAvgRating'].fillna(X_train['awayAvgRating'].mean())

    X_train['awayPoint'] = X_train['awayPoint'].fillna(X_train['awayPoint'].median())
    X_test['awayPoint'] = X_test['awayPoint'].fillna(X_train['awayPoint'].median())

    X_train['awayForm'] = X_train['awayForm'].fillna(X_train['awayForm'].mean())
    X_test['awayForm'] = X_test['awayForm'].fillna(X_train['awayForm'].mean())
    
    # Deal with xG and xA
    X_train['homeExpectedGoal'] = X_train['homeExpectedGoal'].replace(0, X_train['homeExpectedGoal'].median())
    X_test['homeExpectedGoal'] = X_test['homeExpectedGoal'].replace(0, X_train['homeExpectedGoal'].median())

    X_train['awayExpectedGoal'] = X_train['awayExpectedGoal'].replace(0, X_train['awayExpectedGoal'].median())
    X_test['awayExpectedGoal'] = X_test['awayExpectedGoal'].replace(0, X_train['awayExpectedGoal'].median())
    
    X_train.drop(["homeExpectedAssist", "awayExpectedAssist"], axis=1, inplace=True)
    X_test.drop(["homeExpectedAssist", "awayExpectedAssist"], axis=1, inplace=True)
    
    return X_train, X_test, y_train, y_test