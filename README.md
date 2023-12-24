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
- We use Adaboost model with the default hyperparameters mentioned in [sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
- Model Evaluation: the accuracy and balanced accuracy of the model are 0.88 on the train dataset, and 0.85 on the test dataset

# 2. Feature engineering
## 2.1. Libraries
- sklearn: to implement models and feature engineering
- feature_engine, mlxtend: for specific feature selection techniques
- lazypredict: to train and evaluation multiple models and compare them
- Other libraries: numpy, pandas, json, etc.
## 2.2. Techniques
- The initial input has 87 features (89 features - 2 label features), and we remove 8 of them: *homeExpectedAssist*, *awayExpectedAssist*, *homeBallPosession*, *homeCounterAttackShot*, *awayCounterAttackShot*, *homePass*, *awayPass*, *awayAccuratePass*.
- Scaler: we use QuantileTransformer to reduce the effect of outliers, and StandardScaler to scale the features to same range
- For another transformation and feature selection techniques, check **all_feature_selection.ipynb** to know the reason of our above assumptions. Make sure to install required libraries by ```pip install -r requirements.txt```
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
  - If there is any problem with our model and api, please post it in the Issue tab in this repository
