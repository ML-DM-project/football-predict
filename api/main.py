import os

from typing import Union, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import pandas as pd
import pickle, joblib
from sklearn.pipeline import Pipeline
import datetime
from zoneinfo import ZoneInfo
import requests
import json

from utils import *

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["pipeline"] = pickle.load(open("pipeline.pkl", "rb"))
    models["homeAvgRating"] = 6.88
    models["homePoint"] = 17.0
    models["homeForm"] = 1.37
    models["awayAvgRating"] = 6.89
    models["awayPoint"] = 17.0
    models["awayForm"] = 1.43
    
    models["headers"] = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    yield 
    models.clear()
    
app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
async def predict(match_id: str):
    today = datetime.datetime.now(tz=ZoneInfo("Asia/Ho_Chi_Minh"))
    today = today.strftime("%Y-%m-%d")
    
    # get team id and period1 result
    res = requests.get(f'https://api.sofascore.com/api/v1/sport/football/scheduled-events/{today}', headers=models["headers"])
    if res.status_code != 200:
        return {"success": 0, "message": "Can't get data from Sofascore API"}
    
    find_match = 0
    matches = json.loads(res.content)["events"]
    for match in matches:
        if str(match.get("id", None)) == match_id:
            find_match = 1
            homeTeamId = match["homeTeam"]["id"]
            awayTeamId = match["awayTeam"]["id"]
            homeScorePeriod1 = match["homeScore"]["period1"]
            awayScorePeriod1 = match["awayScore"]["period1"]
    
    if find_match == 0:
        return {"success": 0, "message": f"Match with id {match_id} not found"}
    
    # get head2head
    res = requests.get(f'https://api.sofascore.com/api/v1/event/{match_id}/h2h', headers=models["headers"])
    if res.status_code != 200:
        return {"success": 0, "message": "Can't get data from Sofascore API"}
    
    h2h = json.loads(res.content)
    previousHomeWin = h2h.get("teamDuel", dict()).get("homeWins", 0)
    previousAwayWin = h2h.get("teamDuel", dict()).get("awayWins", 0)
    previousDraw = h2h.get("teamDuel", dict()).get("draws", 0)
    previousManagerHomeWin = h2h.get("managerDuel", dict()).get("homeWins", 0)
    previousManagerAwayWin = h2h.get("managerDuel", dict()).get("awayWins", 0)
    previousManagerDraw = h2h.get("managerDuel", dict()).get("draws", 0)
    
    # get pregame form
    res = requests.get(f'https://api.sofascore.com/api/v1/event/{match_id}/pregame-form', headers=models["headers"])
    if res.status_code != 200:
        return {"success": 0, "message": "Can't get data from Sofascore API"}
    
    pregame_form = json.loads(res.content)
    homeAvgRating = pregame_form.get("homeTeam", dict()).get("avgRating", models["homeAvgRating"])
    homePosition = pregame_form.get("homeTeam", dict()).get("position", "")
    homePoint = pregame_form.get("homeTeam", dict()).get("value", models["homePoint"])
    homeForm = form2Point(pregame_form.get("homeTeam", dict()).get("form", models["homeForm"]))
    awayAvgRating = pregame_form.get("awayTeam", dict()).get("avgRating", models["awayAvgRating"])
    awayPosition = pregame_form.get("awayTeam", dict()).get("position", "")
    awayPoint = pregame_form.get("awayTeam", dict()).get("value", models["awayPoint"])
    awayForm = form2Point(pregame_form.get("awayTeam", dict()).get("form", models["awayForm"]))
    
    # get first half statistic
    res = requests.get(f'https://api.sofascore.com/api/v1/event/{match_id}/statistics', headers=models["headers"])
    if res.status_code != 200:
        return {"success": 0, "message": "Can't get data from Sofascore API"}
    
    statistic = json.loads(res.content).get("statistics", None)
    if statistic is None:
        return {"success": 0, "message": f"Match with id {match_id} doesn't exist or hasn't started yet"}
    
    find_period1 = 0
    for item in statistic:
        if item["period"] == "1ST":
            find_period1 = 1
            groupList = item["groups"]
            data_statistic = getInfoFromGroup(groupList)
            
    if find_period1 == 0:
        return {"success": 0, "message": "Statistic of first half not found"}
    
    # create dataframe
    list_column = ['previousHomeWin', 'previousAwayWin', 'previousDraw',
       'previousManagerHomeWin', 'previousManagerAwayWin',
       'previousManagerDraw', 'homeAvgRating', 'homePosition', 'homePoint',
       'homeForm', 'awayAvgRating', 'awayPosition', 'awayPoint', 'awayForm',
       'homeExpectedGoal', 'awayExpectedGoal', 'homeBallPosession', 'homeShotOnTarget',
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
       'awayScorePeriod1']
    
    data_h2h = [previousHomeWin, previousAwayWin, previousDraw, previousManagerHomeWin, previousManagerAwayWin, previousManagerDraw]
    data_pregame_form = [homeAvgRating, homePosition, homePoint, homeForm, 
                         awayAvgRating, awayPosition, awayPoint, awayForm]
    data_result = [homeTeamId, awayTeamId, homeScorePeriod1, awayScorePeriod1]
    
    data = data_h2h + data_pregame_form + data_statistic + data_result
    print(len(data))
    print(len(list_column))
    
    input = pd.DataFrame(columns=list_column)
    input.loc[len(input)] = data
    
    # prediction
    prediction = models["pipeline"].predict_proba(input)
    print(prediction)
    
    return {
        "success": 1,
        "result": {
            "homeWin": prediction[0][2],
            "draw": prediction[0][1],
            "awayWin": prediction[0][0]
        }
    }
    
    
    
