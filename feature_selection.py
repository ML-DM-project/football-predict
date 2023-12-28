import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import wrapper

from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectFromModel
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureAddition
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wrapper import load_fillna

def scale(X_train, X_test):
    scaler1 = QuantileTransformer()
    scaler1.fit(X_train)

    X_train_scaler = scaler1.transform(X_train)
    X_test_scaler = scaler1.transform(X_test)

    scaler2 = StandardScaler()
    scaler2.fit(X_train_scaler)

    X_train_scaler = scaler2.transform(X_train_scaler)
    X_test_scaler = scaler2.transform(X_test_scaler)
    
    return X_train_scaler, X_test_scaler

class ModelSelector():
    def __init__(self, seed=42, kfold=10, outfile="result.txt"):
        self.seed = seed
        self.kfold = kfold
        self.outfile = outfile
        self.models = [LogisticRegression(random_state=seed), 
                       RidgeClassifier(random_state=seed), 
                       DecisionTreeClassifier(random_state=seed), 
                       KNeighborsClassifier(), 
                       LinearDiscriminantAnalysis(), 
                       SVC(random_state=seed), 
                       AdaBoostClassifier(random_state=seed),
                       RandomForestClassifier(random_state=seed), 
                       ExtraTreesClassifier(random_state=seed)]
        self.parameters = [{'C': (0.05, 0.1, 0.2, 1, 5, 10, 50), 'max_iter': (50, 100, 200, 500, 1000)},
                           {'alpha': (0.1, 0.5, 1, 5, 10, 50, 100), 'max_iter': (None, 50, 100, 200, 500)},
                           {'max_depth': (None, 5, 10, 20, 30), 'class_weight': (None, "balanced")},
                           {'n_neighbors': (1, 3, 5, 7, 9), 'weights': ('uniform', 'distance')},
                           {'solver' : ('svd', 'lsqr', 'eigen')},
                           {'C': (0.05, 0.1, 0.2, 1, 5, 10, 50), 'kernel': ('linear', 'poly', 'rbf')},
                           {'estimator': (None, LogisticRegression()), 'n_estimators': (10, 30, 50, 80, 100)},
                           {'n_estimators': (10, 30, 50, 80, 100), 'max_depth': (None, 2, 5, 10, 20)},
                           {'n_estimators': (10, 30, 50, 80, 100), 'max_depth': (None, 2, 5, 10, 20)}]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def fit(self, data_path="match_all_statistic.csv"):
        self.X_train, self.X_test, self.y_train, self.y_test = load_fillna(data_path=data_path, seed=self.seed)
        
        print("Start Correlation Feature Selector\n")
        self.CorrelationFeatureSelector(method="variance")
        self.CorrelationFeatureSelector(method="cardinality")
            
        print("Start Statistical Feature Selector\n")
        self.StatisticalFeatureSelector(method="mi")
        self.StatisticalFeatureSelector(method="anova")
        
        print("Start Wrapper Feature Selector\n")
        for i in range(5, 21):
            self.WrapperFeatureSelector(forward=True, k_features=i)
        for i in range(60, 81):
            self.WrapperFeatureSelector(forward=False, k_features=i)
            
        print("Start Feature Importance Selector\n")
        self.FeatureImportanceSelector()
        
        print("Start Recursive Feature Addition Selector\n")
        self.RecursiveFeatureAdditionSelector()
        
        print(f"Finish!!! Check the result in {self.outfile}\n")
    
    def CorrelationFeatureSelector(self, method="variance"):      
        sel = SmartCorrelatedSelection(threshold=0.9, selection_method=method)
        sel.fit(self.X_train, self.y_train)
        
        X_train_fe_variance = sel.transform(self.X_train)
        X_test_fe_variance = sel.transform(self.X_test)
        
        X_train_scaler, X_test_scaler = scale(X_train_fe_variance, X_test_fe_variance)
        with open(self.outfile, "a", encoding="utf-8") as f:
            f.write(f"Correlation Feature Selector with selection method={method} completed. Start tuning hyperparameters:\n")
        self.tuning(X_train_scaler, X_test_scaler, self.y_train, self.y_test)
        
    def StatisticalFeatureSelector(self, method="anova"):
        if method not in ["mi", "anova"]:
            raise ValueError('Method not supported!')
        
        if method == "mi":
            sel = SelectKBest(mutual_info_classif, k=20)
        else:
            sel = SelectKBest(f_classif, k=53)
        
        sel.fit(self.X_train, self.y_train)
        
        X_train_scaler, X_test_scaler = scale(sel.transform(self.X_train), sel.transform(self.X_test))
        with open(self.outfile, "a", encoding="utf-8") as f:
            f.write(f"Statistical Feature Selector with method={method} completed. Start tuning hyperparameters:\n")
        self.tuning(X_train_scaler, X_test_scaler, self.y_train, self.y_test)
        
    def WrapperFeatureSelector(self, forward: bool, k_features: int):
        if k_features >= len(self.X_train.columns):
            raise ValueError('k_features must fewer than the number of initial features!')
        
        X_train_scaler, X_test_scaler = scale(self.X_train, self.X_test)
        
        direction = 'forward' if forward else 'backward'
        
        available_models = [LogisticRegression(random_state=self.seed),
                            RandomForestClassifier(n_estimators=20, random_state=self.seed),
                            AdaBoostClassifier(random_state=self.seed)]
        for model in available_models:
            sfs = SFS(model, 
                n_features_to_select=k_features,
                tol=None,
                direction=direction,
                scoring='accuracy',
                cv=self.kfold,
                n_jobs=3
            )
            sfs1 = sfs.fit(X_train_scaler, self.y_train)
            
            X_train_sfs = sfs1.transform(X_train_scaler) 
            X_test_sfs = sfs1.transform(X_test_scaler)   
        
            with open(self.outfile, "a", encoding="utf-8") as f:
                if direction == 'forward':
                    f.write(f"Step Forward Feature Selection with {model} to select {k_features} features completed. Start tuning hyperparameters:\n")
                else:
                    f.write(f"Step Backward Feature Selection with {model} to select {k_features} features completed. Start tuning hyperparameters:\n")
            self.tuning(X_train_sfs, X_test_sfs, self.y_train, self.y_test)
    
    def FeatureImportanceSelector(self):    
        X_train_scaler, X_test_scaler = scale(self.X_train, self.X_test)
        
        available_models = [LogisticRegression(C=1000, random_state=self.seed),
                            DecisionTreeClassifier(random_state=self.seed),
                            RandomForestClassifier(random_state=self.seed),
                            AdaBoostClassifier(random_state=self.seed)]
        
        for model in available_models:
            sel = SelectFromModel(model)
            sel.fit(X_train_scaler, self.y_train)

            X_train_sfm = sel.transform(X_train_scaler)
            X_test_sfm = sel.transform(X_test_scaler)
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                f.write(f"Select Feature Importance with {model} completed. Start tuning hyperparameters:\n")
            self.tuning(X_train_sfm, X_test_sfm, self.y_train, self.y_test)
    
    def RecursiveFeatureAdditionSelector(self):
        X_train_scaler, X_test_scaler = scale(self.X_train, self.X_test)
        
        available_models = [LogisticRegression(random_state=self.seed),
                            DecisionTreeClassifier(random_state=self.seed),
                            RandomForestClassifier(random_state=self.seed),
                            AdaBoostClassifier(random_state=self.seed)]
        
        for model in available_models:
            rfa = RecursiveFeatureAddition(
                variables=None,  # automatically evaluate all numerical variables
                estimator=model,  # the ML model
                scoring='accuracy',  # the metric we want to evalute
                threshold=0.0001,  # the minimum performance increase needed to select a feature
                cv=self.kfold,  # cross-validation
            )
            rfa.fit(X_train_scaler, self.y_train)
            
            X_train_rfa = rfa.transform(X_train_scaler)
            X_test_rfa = rfa.transform(X_test_scaler)
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                f.write(f"Recursive Feature Addition with {model} completed. Start tuning hyperparameters:\n")
            self.tuning(X_train_rfa, X_test_rfa, self.y_train, self.y_test)
        
    def tuning(self, X_train, X_test, y_train, y_test):
        for model, parameter in zip(self.models, self.parameters):
            grid = GridSearchCV(estimator=model, param_grid=parameter, scoring="accuracy", cv=self.kfold)
            
            grid_result = grid.fit(X_train, y_train)
            test_score = grid_result.score(X_test, y_test)
            
            with open(self.outfile, "a", encoding="utf-8") as f:
                f.write(f"Result of {model}:\n")
                f.write(f"Best accuracy: {grid_result.best_score_}, best hyperparameter: {grid_result.best_params_}\n")
                f.write(f"Accuracy on test set: {test_score}\n\n")