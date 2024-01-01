# Football Result Prediction in Half-time

Inspired by the half-time result guess of match's commentators, we create a simple Machine Learning model to predict the result of a football match given the prematch data and first-half data of 2 teams and the match. 

We only predict the ratio of win/draw/lose, not the exact goal scored by each team.

# 1. About the data and model
## 1.1. Data
- The data we used is from all the matches in Premier League, including 2022-2023 season and part of 2023-2024 season (from the beginning to 7/12/2023). You can access the structured dataset via **match_all_statistic.csv**
- 516 samples with 89 features (include 2 label features)
- Crawling method and features: check out at https://github.com/ML-DM-project/sofascore-crawler/
- Data is splited with the train/test ratio is 0.85/0.15, stratified=True (so the label distributions of train and test datasets are the same)

## 1.2. Model
- We use Logistic Regression model with the default hyperparameters mentioned in [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Model Evaluation: the 10-folds cross-validation accuracy of the model is 0.8354 

# 2. Feature engineering
## 2.1. Libraries
- sklearn: to implement models and feature engineering
- feature_engine, mlxtend: for specific feature selection techniques
- lazypredict: to train and evaluation multiple models and compare them
- Other libraries: numpy, pandas, json, etc.
## 2.2. Techniques
- Scaler: we use QuantileTransformer to reduce the effect of outliers and transform each feature to uniform distribution 
- The initial input has 87 features (89 features - 2 label features), and we apply plenty of techniques mentioned in **all_feature_selection.ipynb**. 
- We use those techniques to tuning hyperparameters with 10-folds cross-validation using GridSearchCV (check **feature_selection.py** for more detail of my implement). We run 5 times with different random seed (0, 42, 100, 1234, 2023) and calculate the mean/std values and put in **result_all.txt**.
- Using 4 values (mean/std 10-folds acc, mean/std test acc) of each model, we decided to choose:
   - Model: Logistic Regression
   - Feature selection: Step Forward Feature Selection (estimator=Logistic Regression, n_features=9)
- Finally, we tuning Logistic Regression with Step Forward Feature Selection (estimator=Logistic Regression, n_features=9) using all dataset (check out **model_selection.py**)
- If you want to reproduce our resul, make sure to install required libraries by ```pip install -r requirements.txt```
# 3. API
## 3.1. Prerequisites: 
- Python >= 3.6
- All operation systems are fine
## 3.2. Install
- ```cd api```
- Install the required libraries: ```pip install -r requirements.txt```
## 3.3. Run the API
- Run the api
    - Normal running: ```uvicorn main:app```
    - With reload for debug: ```uvicorn main:app --reload```
- Now the api has been deployed at port 8000 of your local machine. Inference the model by sending the GET request to the api: 
```http://127.0.0.1:8000/predict?match_id={match_id}```
- The match_id value is the match id of Sofascore, directly in the match url. For instance, the url of the match between Liverpool and Arsenal in 00:30 24/12/2023: https://www.sofascore.com/liverpool-arsenal/HQfh#id:11352520 => the match_id is 11352520
- Example inference (on Postman):
    - Send request:

       ![image](https://github.com/ML-DM-project/football-predict/assets/77562200/272da5f6-a236-4737-b30e-d16bf19f1948)

    - Get result:

       ![image](https://github.com/ML-DM-project/football-predict/assets/77562200/0a5dce37-71f8-435a-be38-8b88821ae3f2)
 
    - The result contains the winning chance of home team (in this case is Liverpool), winning chance of away team (in this case is Arsenal), and draw chance.
 
  # 4. Issue
  - If there is any problem with our model and api, please post it in the Issue tab in this repository or contact viethuy061002@gmail.com
