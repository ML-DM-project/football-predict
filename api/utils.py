def char2Point(res):
    if res == 'L':
        return 0
    if res == 'D':
        return 1
    return 3

def form2Point(form):
    if form is None or form == "":
        return ""
    if isinstance(form, int):
        return form
    points = [char2Point(match) for match in form]
    return sum(points) / len(points)

def getExpected(statisticsItems):
    homeExpectedGoal = 0
    awayExpectedGoal = 0
    
    for item in statisticsItems:
        if item["name"] == "Expected goals":
            homeExpectedGoal = item["home"]
            awayExpectedGoal = item["away"]      
    return [homeExpectedGoal, awayExpectedGoal]  

def getPossession(statisticsItems):
    homeBallPosession = 0
    for item in statisticsItems:
        if item["name"] == "Ball possession":
            homeBallPosession = int(item["home"][:-1])
            
    return [homeBallPosession]

def getShot(statisticsItems):
    homeShotOnTarget = 0
    awayShotOnTarget = 0
    homeShotOffTarget = 0
    awayShotOffTarget = 0
    homeBlockedShot = 0
    awayBlockedShot = 0   
    
    for item in statisticsItems:
        if item["name"] == "Shots on target":
            homeShotOnTarget = item["home"]
            awayShotOnTarget = item["away"]
        if item["name"] == "Shots off target":
            homeShotOffTarget = item["home"]
            awayShotOffTarget = item["away"] 
        if item["name"] == "Blocked shots":
            homeBlockedShot = item["home"]
            awayBlockedShot = item["away"] 
            
    return [homeShotOnTarget, awayShotOnTarget, homeShotOffTarget, awayShotOffTarget, homeBlockedShot, awayBlockedShot]

def getTVData(statisticsItems):
    homeCorner = 0
    awayCorner = 0
    homeOffside = 0
    awayOffside = 0
    homeYellowCard = 0
    awayYellowCard = 0
    homeRedCard = 0
    awayRedCard = 0
    homeFreekick = 0
    awayFreekick = 0
    homeThrowIn = 0
    awayThrowIn = 0
    homeGoalkick = 0
    awayGoalkick = 0  
    
    for item in statisticsItems:
        if item["name"] == "Corner kicks":
            homeCorner = item["home"]
            awayCorner = item["away"]
        if item["name"] == "Offsides":
            homeOffside = item["home"]
            awayOffside = item["away"] 
        if item["name"] == "Yellow cards":
            homeYellowCard = item["home"]
            awayYellowCard = item["away"]
        if item["name"] == "Red cards":
            homeRedCard = item["home"]
            awayRedCard = item["away"] 
        if item["name"] == "Free kicks":
            homeFreekick = item["home"]
            awayFreekick = item["away"]
        if item["name"] == "Throw-ins":
            homeThrowIn = item["home"]
            awayThrowIn = item["away"]
        if item["name"] == "Goal kicks":
            homeGoalkick = item["home"]
            awayGoalkick = item["away"] 
            
    return [homeCorner, awayCorner, homeOffside, awayOffside, homeYellowCard, awayYellowCard, 
            homeRedCard, awayRedCard, homeFreekick, awayFreekick, homeThrowIn, awayThrowIn, homeGoalkick, awayGoalkick]
    
def getShotExtra(statisticsItems):
    homeBigChance = 0
    awayBigChance = 0
    homeBigChanceMissed = 0
    awayBigChanceMissed = 0
    homeHitWoodwork = 0
    awayHitWoodwork = 0
    homeCounterAttack = 0
    awayCounterAttack = 0
    homeCounterAttackShot = 0
    awayCounterAttackShot = 0
    homeCounterAttackGoal = 0
    awayCounterAttackGoal = 0
    homeShotInsideBox = 0
    awayShotInsideBox = 0   # shot outside = shot on + shot off - shot inside
    homeGoalSave = 0
    awayGoalSave = 0
    
    for item in statisticsItems:
        if item["name"] == "Big chances":
            homeBigChance = item["home"]
            awayBigChance = item["away"]
        if item["name"] == "Big chances missed":
            homeBigChanceMissed = item["home"]
            awayBigChanceMissed = item["away"] 
        if item["name"] == "Shots inside box":
            homeShotInsideBox = item["home"]
            awayShotInsideBox = item["away"]
        if item["name"] == "Hit woodwork":
            homeHitWoodwork = item["home"]
            awayHitWoodwork = item["away"]
        if item["name"] == "Counter attacks":
            homeCounterAttack = item["home"]
            awayCounterAttack = item["away"]
        if item["name"] == "Counter attack shots":
            homeCounterAttackShot = item["home"]
            awayCounterAttackShot = item["away"]
        if item["name"] == "Counter attack goals":
            homeCounterAttackGoal = item["home"]
            awayCounterAttackGoal = item["away"]  
        if item["name"] == "Goalkeeper saves":
            homeGoalSave = item["home"]
            awayGoalSave = item["away"]
            
    return [homeBigChance, awayBigChance, homeBigChanceMissed, awayBigChanceMissed, 
            homeHitWoodwork, awayHitWoodwork, homeCounterAttack, awayCounterAttack, 
            homeCounterAttackShot, awayCounterAttackShot, homeCounterAttackGoal, awayCounterAttackGoal,
            homeShotInsideBox, awayShotInsideBox, homeGoalSave, awayGoalSave]
    
def getPass(statisticsItems):
    homePass = 0
    awayPass = 0
    homeAccuratePass = 0
    awayAccuratePass = 0
    homeLongPass = 0
    awayLongPass = 0
    homeAccurateLongPass = 0
    awayAccurateLongPass = 0
    homeCross = 0
    awayCross = 0
    homeAccurateCross = 0
    awayAccurateCross = 0
    
    for item in statisticsItems:
        if item["name"] == "Passes":
            homePass = item["home"]
            awayPass = item["away"]
        if item["name"] == "Accurate passes":
            homeAccuratePass = item["home"].split(" ")[0]
            awayAccuratePass = item["away"].split(" ")[0]
        if item["name"] == "Long balls":
            homeAccurateLongPass = item["home"].split(" ")[0].split("/")[0]
            awayAccurateLongPass = item["away"].split(" ")[0].split("/")[0]
            homeLongPass = item["home"].split(" ")[0].split("/")[1]
            awayLongPass = item["away"].split(" ")[0].split("/")[1]
        if item["name"] == "Crosses":
            homeAccurateCross = item["home"].split(" ")[0].split("/")[0]
            awayAccurateCross = item["away"].split(" ")[0].split("/")[0]
            homeCross = item["home"].split(" ")[0].split("/")[1]
            awayCross = item["away"].split(" ")[0].split("/")[1]
            
    return [homePass, awayPass, homeAccuratePass, awayAccuratePass, homeLongPass, awayLongPass,
            homeAccurateLongPass, awayAccurateLongPass, homeCross, awayCross, homeAccurateCross, awayAccurateCross]
    
def getDuel(statisticsItems):
    homeDribble = 0
    awayDribble = 0
    homeSuccessfulDribble = 0
    awaySuccessfulDribble = 0
    homePossesionLost = 0
    awayPossesionLost = 0
    homeDuelWon = 0
    awayDuelWon = 0
    homeAerialWon = 0
    awayAerialWon = 0
    
    for item in statisticsItems:
        if item["name"] == "Dribbles":
            homeSuccessfulDribble = item["home"].split(" ")[0].split("/")[0]
            awaySuccessfulDribble = item["away"].split(" ")[0].split("/")[0]
            homeDribble = item["home"].split(" ")[0].split("/")[1]
            awayDribble = item["away"].split(" ")[0].split("/")[1]
        if item["name"] == "Possession lost":
            homePossesionLost = item["home"]
            awayPossesionLost = item["away"]
        if item["name"] == "Duels won":
            homeDuelWon = item["home"]
            awayDuelWon = item["away"]
        if item["name"] == "Aerials won":
            homeAerialWon = item["home"]
            awayAerialWon = item["away"]
            
    return [homeDribble, awayDribble, homeSuccessfulDribble, awaySuccessfulDribble, homePossesionLost,
            awayPossesionLost, homeDuelWon, awayDuelWon, homeAerialWon, awayAerialWon]
              
def getDefending(statisticsItems):
    homeTackle = 0
    awayTackle = 0
    homeInterception = 0
    awayInterception = 0
    homeClearance = 0
    awayClearance = 0
    
    for item in statisticsItems:
        if item["name"] == "Tackles":
            homeTackle = item["home"]
            awayTackle = item["away"]
        if item["name"] == "Interceptions":
            homeInterception = item["home"]
            awayInterception = item["away"]
        if item["name"] == "Clearances":
            homeClearance = item["home"]
            awayClearance = item["away"]
            
    return [homeTackle, awayTackle, homeInterception, awayInterception, homeClearance, awayClearance]

def getInfoFromGroup(groupList):
    info = []
    for group in groupList:
        if group["groupName"] == "Expected":
            info += getExpected(group["statisticsItems"])
            # print(f'Expected: {len(getExpected(group["statisticsItems"]))}')
        if group["groupName"] == "Possession":
            info += getPossession(group["statisticsItems"])
            # print(f'Possession: {len(getPossession(group["statisticsItems"]))}')
        if group["groupName"] == "Shots":
            info += getShot(group["statisticsItems"])
            # print(f'Shots: {len(getShot(group["statisticsItems"]))}')
        if group["groupName"] == "TVData":
            info += getTVData(group["statisticsItems"])
            # print(f'TVData: {len(getTVData(group["statisticsItems"]))}')
        if group["groupName"] == "Shots extra":
            info += getShotExtra(group["statisticsItems"])
            # print(f'Shots extra: {len(getShotExtra(group["statisticsItems"]))}')
        if group["groupName"] == "Passes":
            info += getPass(group["statisticsItems"])
            # print(f'Passes: {len(getPass(group["statisticsItems"]))}')
        if group["groupName"] == "Duels":
            info += getDuel(group["statisticsItems"])
            # print(f'Duels: {len(getDuel(group["statisticsItems"]))}')
        if group["groupName"] == "Defending":
            info += getDefending(group["statisticsItems"])
            # print(f'Defending: {len(getDefending(group["statisticsItems"]))}')
            
    info = [float(i) for i in info]
    return info