import re
def ball_in_over(x):
    try:
        return (float(str(x).split('.')[0])*6)+(float(str(x).split('.')[1]))
    except:
        return 0
    
def bowl_Avg(row):
    try:
        return row["cum_runs"]/row["cum_Wkts"]
    except:
        return 0
    

def bowl_SR(row):
    try:
        return row["cum_balls"]/row["cum_Wkts"]
    except:
        return 0
    
def bowl_Eco(row):
    try:
        return row["cum_runs"]*6/row["cum_balls"]
    except:
        return 0


def get_bowler_features(player_df = None):
       
    
    player_df["Runs"] = player_df.Runs.map(lambda x : str(x).replace('-','0')).astype(int)
    player_df["cum_runs"] = player_df.Runs.cumsum()

    player_df["Wkts"] = player_df.Wkts.map(lambda x : str(x).replace('-','0')).astype(int)
    player_df["cum_Wkts"] = player_df.Wkts.cumsum()

    player_df["Balls"] = player_df.Overs.map(lambda x : ball_in_over(x))
    player_df["cum_balls"] = player_df.Balls.cumsum()
    
    player_df["bowl_Avg"] = player_df.apply(bowl_Avg,axis =1)
    player_df["bowl_SR"] = player_df.apply(bowl_SR,axis =1)
    player_df["bowl_Eco"] = player_df.apply(bowl_Eco,axis =1)
  
    
    return player_df


def is_number(s):
    try:
        float(s)
        return 1
    except ValueError:
        return 0

def bat_avg(row):
    try:
        return row["cum_runs"]/row["cum_out"]
    except:
        return 0

def bat_SR(row):
    try:
        return row["cum_runs"] * 100/row["cum_balls"]
    except:
        return 0
    
    
def get_batsman_features(player_df = None):
     
       player_df['odi_id'] = player_df['odi_id'].astype(int)
       player_df['Runs1']= player_df['Runs'].astype(str).map(lambda x : re.sub("[^0-9A-Za-z]","",x))
       player_df['DNB']= player_df['Runs1'].map(lambda x : 1 if x in ['DNB','TDNB','sub','absent'] else 0 )
       player_df['Runs2']= player_df['Runs1'].map(lambda x : '0' if x in ['DNB','TDNB','sub','absent'] else x ).astype(int)      
       player_df["cum_runs"] = player_df.Runs2.cumsum()
       player_df["out"] = player_df.Runs.map(lambda x : is_number(x))
       player_df["cum_out"] = player_df.out.cumsum()
       player_df["BF1"] = player_df.BF.map(lambda x : str(x).replace('-','0')).astype(int)
       player_df["cum_balls"] = player_df.BF1.cumsum()
       player_df["bat_avg"] = player_df.apply(bat_avg,axis =1)
       player_df["bat_SR"] = player_df.apply(bat_SR,axis =1)
       
       return player_df