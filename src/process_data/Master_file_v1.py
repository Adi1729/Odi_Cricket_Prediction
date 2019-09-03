#%reset -f
import pandas as pd
import os
import re
from scipy import stats
os.chdir(r'C:/Users/Vikash/Downloads/Hackathon/src')
import sys
from player_record_v4 import *
import time

#Loading the all match dataframe

df_matches = pd.read_csv("../Raw_Data/Match/all_matches_vF1.csv")
df_matches = df_matches[df_matches["odi_id"]>4142]

#lineup =  pd.read_csv("../Raw_Data/Match/match_lineup_v3.csv")
#
#lineup['team1'] = lineup['team1'].apply(lambda x: eval(x))
#lineup['team2'] = lineup['team2'].apply(lambda x: eval(x))
#
#def match_squad(match_id,index):
#    
#    team1= list(lineup[lineup['match_id']==match_id]['team1'][index].values())[0]
#    team2= list(lineup[lineup['match_id']==match_id]['team2'][index].values())[0]
#    player = [x for x,y in team1.items()] + [x for x,y in team2.items()] 
#    player_id2 = [y for x,y in team1.items()] +  [y for x,y in team2.items()]
#
#    return pd.DataFrame(data = {'Players':player, 'player_id':player_id2 , 'match_id': match_id})
#
#squad_11_all_matches = pd.DataFrame()
#for rownum,row in lineup.iterrows():
#    match_id = row['match_id']
#    df = match_squad(match_id,rownum)
#    squad_11_all_matches= squad_11_all_matches.append(df)
#
#squad_11_all_matches.to_csv("../Raw_Data/Match/squad_11_all_matches.csv",index=False)

#### FIRST UPDATE THE ABOVE CREATED FILE AND THEN UODATE IT WITH WORLD CUP SQUAD AND RE READ
squad_11_all_matches = pd.read_csv("../Raw_Data/Match/squad_11_all_matches_vF1.csv")


df_match_Squad_data = pd.merge(df_matches, squad_11_all_matches , how = 'left', left_on = 'match_id',right_on = 'match_id')
df_match_squad_id = df_match_Squad_data.drop (['match_link'], axis = 1)

Teams = ['India','Australia','Bangladesh','England','New Zealand','Pakistan','South Africa','Sri Lanka','West Indies','Afghanistan','Ireland','Zimbabwe']

def team_selection(row):
    team1 = row['Team 1']
    team2 = row['Team 2']
    try:
        if (team1 in Teams and team2 in Teams) :
            return 1
    except:
        return 0
    

df_match_squad_id['team_select']= df_match_squad_id.apply(team_selection, axis =1)
df_match_squad_id = df_match_squad_id[(df_match_squad_id['team_select']==1)]
df_match_squad_id = df_match_squad_id[(df_match_squad_id['player_id']>0)]
df_match_squad_id['player_id'] = df_match_squad_id['player_id'].astype(int)
df_match_squad_id = df_match_squad_id.drop (['team_select'], axis = 1)


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
        return row["cum_runs"]/row["cum_balls"]
    except:
        return "0"



def ball_in_over(x):
    try:
        return (float(str(x).split('.')[0])*6)+(float(str(x).split('.')[1]))
    except:
        return 0



def bowl_SR(row):
    try:
        return row["cum_balls"]/row["cum_Wkts"]
    except:
        return "0"


def bowl_Eco(row):
    try:
        return row["cum_runs"]*6/row["cum_balls"]
    except:
        return "0"



def bowl_Avg(row):
    try:
        return row["cum_runs"]/row["cum_Wkts"]
    except:
        return "0"

#### bATSMAN FEATURES

def get_batman_features(row):
       
    player_id= row['player_id']
    ODI_number = row['odi_id']
     
    
    try:
        bat_df = pd.read_csv("../Raw_Data/Players/batting_{}.csv".format(player_id))
        
    except:
        bat_df = player_record(player_id,role_type='batting')
        bat_df.to_csv("../Raw_Data/Players/batting_{}.csv".format(player_id),index=False)
       
    try:
        bat_df['odi_id'] = bat_df['odi_id'].astype(int)
        bat_df = bat_df[bat_df['odi_id']<ODI_number].reset_index()
        Pos_no = stats.mode(bat_df[bat_df['Pos']!='-']['Pos'])[0][0]
        Pos_cnt = stats.mode(bat_df[bat_df['Pos']!='-']['Pos'])[1][0]

        
        bat_df['Runs1']= bat_df['Runs'].astype(str).map(lambda x : re.sub("[^0-9A-Z]","",x))
        bat_df['Runs2']= bat_df['Runs1'].map(lambda x : '0' if x in ['DNB','TDNB'] else x ).astype(int)
        bat_df["cum_runs"] = bat_df.Runs2.cumsum()
        bat_df["out"] = bat_df.Runs.map(lambda x : is_number(x))
        bat_df["cum_out"] = bat_df.out.cumsum()
        bat_df["BF1"] = bat_df.BF.map(lambda x : str(x).replace('-','0')).astype(int)
        bat_df["cum_balls"] = bat_df.BF1.cumsum()
        bat_df["bat_avg"] = bat_df.apply(bat_avg,axis =1)
        bat_df["bat_SR"] = bat_df.apply(bat_SR,axis =1)
        bat_avg1 = bat_df['bat_avg'].iloc[len(bat_df)-1]
        bat_sr1 = bat_df['bat_SR'].iloc[len(bat_df)-1]
    except:
        bat_sr1 = 0
        bat_avg1 = 0
        Pos_no = 0
        Pos_cnt = 0
        
    return [bat_avg1,bat_sr1,Pos_no,Pos_cnt]

batting_features = pd.read_csv(r'../Processed/batting_avg_sr.csv')
batting_df = batting_features

for idx  in range(len(df_match_squad_id)):
    st = time.time()
    try: 
        player_id = df_match_squad_id['player_id'].iloc[idx]
        match_id =  df_match_squad_id['match_id'].iloc[idx]
        if player_id in batting_features['player_id'].tolist()  and  match_id in batting_features['match_id'].tolist():
            print('Processed... %d  TimeTaken.. %.2f '%((idx+1),time.time()-st))  
            continue
        batting_dict= dict(zip(['bat_avg1','bat_sr1','Pos_no','Pos_cnt'],get_batman_features(df_match_squad_id.iloc[idx])))
        batting_dict.update({'key':str(match_id) + str(player_id),
                            'match_id':match_id ,
                            'player_id':player_id})
       
        batting_df = batting_df.append(pd.DataFrame([batting_dict]))
        if ((idx+1)%1)==0:
            print('Processed... %d  TimeTaken.. %.2f '%((idx+1),time.time()-st))
            batting_df.to_csv(r'../Processed/batting_avg_sr.csv',index= False)
            
    except:
        print('%d : %s'%(match_id,player_id))
        break
        
  
##### bOWLING FEATURES
        
def get_bowler_features(row):
       
    player_id= row['player_id']
    ODI_number = row['odi_id']
    try:
        bowl_df = pd.read_csv("../Raw_Data/Players/bowling_{}.csv".format(player_id))
        
    except:
        bowl_df = player_record(player_id,role_type='bowling')
        bowl_df.to_csv("../Raw_Data/Players/bowling_{}.csv".format(player_id),index=False)
        
        
    try:
        bowl_df['odi_id'] = bowl_df['odi_id'].astype(int)
        bowl_df = bowl_df[bowl_df['odi_id']<ODI_number].reset_index()
        Pos_no = stats.mode(bowl_df[bowl_df['Pos']!='-']['Pos'])[0][0]
        Pos_cnt = stats.mode(bowl_df[bowl_df['Pos']!='-']['Pos'])[1][0]

        bowl_df["Runs"] = bowl_df.Runs.map(lambda x : str(x).replace('-','0')).astype(int)
        bowl_df["cum_runs"] = bowl_df.Runs.cumsum()

        bowl_df["Wkts"] = bowl_df.Wkts.map(lambda x : str(x).replace('-','0')).astype(int)
        bowl_df["cum_Wkts"] = bowl_df.Wkts.cumsum()

        bowl_df["Balls"] = bowl_df.Overs.map(lambda x : ball_in_over(x))
        bowl_df["cum_balls"] = bowl_df.Balls.cumsum()
        
        bowl_df["bowl_Avg"] = bowl_df.apply(bowl_Avg,axis =1)
        bowl_df["bowl_SR"] = bowl_df.apply(bowl_SR,axis =1)
        bowl_df["bowl_Eco"] = bowl_df.apply(bowl_Eco,axis =1)
        
        bowl_Avg1 = bowl_df['bowl_Avg'].iloc[len(bowl_df)-1]
        bowl_SR1 = bowl_df['bowl_SR'].iloc[len(bowl_df)-1]
        bowl_Eco1 = bowl_df['bowl_Eco'].iloc[len(bowl_df)-1]
        
    except :
        
        bowl_Avg1 = 0
        bowl_SR1 = 0    
        bowl_Eco1 = 0
        Pos_no = 0
        Pos_cnt = 0

    return [bowl_Avg1,bowl_SR1,bowl_Eco1,Pos_no,Pos_cnt]

bowling_features = pd.read_csv(r'../Processed/bowling_avg_sr_eco.csv')
bowling_df = bowling_features

for idx  in range(len(df_match_squad_id)):
    st = time.time()
 #   import pdb; pdb.set_trace()
    try: 
        player_id = df_match_squad_id['player_id'].iloc[idx]
        match_id =  df_match_squad_id['match_id'].iloc[idx]
#        if player_id in bowling_features['player_id'].tolist()  and  match_id in bowling_features['match_id'].tolist():
#            print('Processed... %d  TimeTaken.. %.2f '%((idx+1),time.time()-st)) 
#            continue
        bowling_dict= dict(zip(['bowl_Avg1','bowl_SR1','bowl_Eco1','Pos_no','Pos_cnt'],get_bowler_features(df_match_squad_id.iloc[idx])))
        bowling_dict.update({'key':str(match_id) + str(player_id),
                            'match_id':match_id ,
                            'player_id':player_id})
       
        bowling_df = bowling_df.append(pd.DataFrame([bowling_dict]))
        if ((idx+1)%1)==0:
            print('Processed... %d  TimeTaken.. %.2f '%((idx+1),time.time()-st))
            bowling_df.to_csv(r'../Processed/bowling_avg_sr_eco.csv',index= False)
            
    except:
        print('%d : %s'%(match_id,player_id))
        break




### creating squad file

squad = pd.DataFrame()
match_ids = df_match_squad_id['match_id'].unique().tolist()

for match_id in match_ids:
    match_lineup = df_match_squad_id[df_match_squad_id['match_id'] == match_id]
    
    match_lineup['Team'] = (['T1']* int(len(match_lineup)/2)) + (['T2']* (len(match_lineup) -int(len(match_lineup)/2)))
    squad = squad.append(pd.DataFrame(match_lineup))
    squad = squad[['match_id','player_id', 'Team']]
squad.to_csv(r'../Processed/squad.csv',index= False)


df_match_final_data = pd.merge(df_match_squad_id, squad , how = 'left', left_on = ['match_id','player_id'],right_on = ['match_id','player_id'])


df_match_squad_id = df_match_squad_id.rename(columns = {'team' : 'Team'})
#### Merging the attrubutes

list1 = ['T1_P1','T1_P2','T1_P3','T1_P4','T1_P5','T1_P6','T1_P7','T1_P8','T1_P9','T1_P10','T1_P11']        
list2 = ['T2_P1','T2_P2','T2_P3','T2_P4','T2_P5','T2_P6','T2_P7','T2_P8','T2_P9','T2_P10','T2_P11']

#### batting features

batting_avg_sr = pd.read_csv("../Processed/batting_avg_sr.csv")
batting_avg_sr['Pos_no']=batting_avg_sr['Pos_no'].astype(str).replace('0','12').astype(int)

df_bat_atr_merged = pd.merge(df_match_squad_id, batting_avg_sr , how = 'left', left_on = ['match_id','player_id'],right_on = ['match_id','player_id'])
df_bat_atr_merged_srt = df_bat_atr_merged.sort_values(['match_id','Team','Pos_no','Pos_cnt','bat_avg1'],ascending = [1,1,1,0,0])

###pulling bat_avg

Col_Bat_T1_Avg = [s + "_Bat_Avg" for s in list1]
Col_Bat_T2_Avg = [s + "_Bat_Avg" for s in list2]

df_bat_avg = pd.DataFrame()

def bat_avg_attr(row):
    match = row['match_id']
    
    bat_df = df_bat_atr_merged_srt[df_bat_atr_merged_srt['match_id']==match]
    bat_df_T1 = bat_df[bat_df['Team']=='T1']['bat_avg1'].iloc[0:11].tolist()
    bat_df_T2 = bat_df[bat_df['Team']=='T2']['bat_avg1'].iloc[0:11].tolist()
    
    dict_T1 = dict(zip(Col_Bat_T1_Avg,bat_df_T1))
    dict_T2 = dict(zip(Col_Bat_T2_Avg,bat_df_T2))
    
    dict_T1.update({'match_id':match})
    dict_T1.update(dict_T2)
    
    return dict_T1
    
for i in range(len(df_matches)):
    try:
        df_bat_avg = df_bat_avg.append(pd.DataFrame([bat_avg_attr(df_matches.iloc[i])]))
    except:
        break

 
###pulling bat_SR

Col_Bat_T1_SR = [s + "_Bat_SR" for s in list1]
Col_Bat_T2_SR = [s + "_Bat_SR" for s in list2]

df_bat_SR = pd.DataFrame()

def bat_SR_attr(row):
    match = row['match_id']
    
    bat_df = df_bat_atr_merged_srt[df_bat_atr_merged_srt['match_id']==match]
    bat_df_T1 = bat_df[bat_df['Team']=='T1']['bat_sr1'].iloc[0:11].tolist()
    bat_df_T2 = bat_df[bat_df['Team']=='T2']['bat_sr1'].iloc[0:11].tolist()
    
    dict_T1 = dict(zip(Col_Bat_T1_SR,bat_df_T1))
    dict_T2 = dict(zip(Col_Bat_T2_SR,bat_df_T2))
    
    dict_T1.update({'match_id':match})
    dict_T1.update(dict_T2)
    
    return dict_T1
    
for i in range(len(df_matches)):
    try:
        df_bat_SR = df_bat_SR.append(pd.DataFrame([bat_SR_attr(df_matches.iloc[i])]))
    except:
        break



#### bowling features

bowling_avg_sr_eco = pd.read_csv("../Processed/bowling_avg_sr_eco.csv")
bowling_avg_sr_eco['avg_sr_eco'] = bowling_avg_sr_eco['bowl_Avg1']*bowling_avg_sr_eco['bowl_Eco1']*bowling_avg_sr_eco['bowl_SR1']

bowling_avg_sr_eco['Pos_no']=bowling_avg_sr_eco['Pos_no'].astype(str).replace('0','12').astype(int)

df_bowl_atr_merged = pd.merge(df_match_squad_id, bowling_avg_sr_eco , how = 'left', left_on = ['match_id','player_id'],right_on = ['match_id','player_id'])
df_bowl_atr_merged_srt = df_bowl_atr_merged.sort_values(['match_id','Team','Pos_no','Pos_cnt','avg_sr_eco'],ascending = [1,1,1,0,1])
 
###pulling bowl_avg

Col_Bowl_T1_avg = [s + "_Bowl_avg" for s in list1]
Col_Bowl_T2_avg = [s + "_Bowl_avg" for s in list2]

df_bowl_avg = pd.DataFrame()

def bowl_avg_attr(row):
    match = row['match_id']
    
    bowl_df = df_bowl_atr_merged_srt[df_bowl_atr_merged_srt['match_id']==match]
    bowl_df_T1 = bowl_df[bowl_df['Team']=='T1']['bowl_Avg1'].iloc[0:5].tolist()
    bowl_df_T2 = bowl_df[bowl_df['Team']=='T2']['bowl_Avg1'].iloc[0:5].tolist()
    
    dict_T1 = dict(zip(Col_Bowl_T1_avg,bowl_df_T1))
    dict_T2 = dict(zip(Col_Bowl_T2_avg,bowl_df_T2))
    
    dict_T1.update({'match_id':match})
    dict_T1.update(dict_T2)
    
    return dict_T1
    
for i in range(len(df_matches)):
    try:
        df_bowl_avg = df_bowl_avg.append(pd.DataFrame([bowl_avg_attr(df_matches.iloc[i])]))
    except:
        break

###pulling bowl_SR

Col_Bowl_T1_SR = [s + "_Bowl_SR" for s in list1]
Col_Bowl_T2_SR = [s + "_Bowl_SR" for s in list2]

df_bowl_sr = pd.DataFrame()

def bowl_sr_attr(row):
    match = row['match_id']
    
    bowl_df = df_bowl_atr_merged_srt[df_bowl_atr_merged_srt['match_id']==match]
    bowl_df_T1 = bowl_df[bowl_df['Team']=='T1']['bowl_SR1'].iloc[0:5].tolist()
    bowl_df_T2 = bowl_df[bowl_df['Team']=='T2']['bowl_SR1'].iloc[0:5].tolist()
    
    dict_T1 = dict(zip(Col_Bowl_T1_SR,bowl_df_T1))
    dict_T2 = dict(zip(Col_Bowl_T2_SR,bowl_df_T2))
    
    dict_T1.update({'match_id':match})
    dict_T1.update(dict_T2)
    
    return dict_T1
    
for i in range(len(df_matches)):
    try:
        df_bowl_sr = df_bowl_sr.append(pd.DataFrame([bowl_sr_attr(df_matches.iloc[i])]))
    except:
        break



###pulling bowl_eco

Col_Bowl_T1_eco = [s + "_Bowl_eco" for s in list1]
Col_Bowl_T2_eco = [s + "_Bowl_eco" for s in list2]

df_bowl_eco = pd.DataFrame()

def bowl_eco_attr(row):
    
    match = row['match_id']
    
    bowl_df = df_bowl_atr_merged_srt[df_bowl_atr_merged_srt['match_id']==match]
    bowl_df_T1 = bowl_df[bowl_df['Team']=='T1']['bowl_Eco1'].iloc[0:5].tolist()
    bowl_df_T2 = bowl_df[bowl_df['Team']=='T2']['bowl_Eco1'].iloc[0:5].tolist()
    
    dict_T1 = dict(zip(Col_Bowl_T1_eco,bowl_df_T1))
    dict_T2 = dict(zip(Col_Bowl_T2_eco,bowl_df_T2))
    
    dict_T1.update({'match_id':match})
    dict_T1.update(dict_T2)
    
    return dict_T1
    
for i in range(len(df_matches)):
    try:
        df_bowl_eco = df_bowl_eco.append(pd.DataFrame([bowl_eco_attr(df_matches.iloc[i])]))
    except:
        break


### combining all the features
        
df_bat_bowl_features = pd.merge(df_bat_avg, df_bat_SR , how = 'left', left_on = ['match_id'],right_on = ['match_id'])

for l in [df_bowl_eco, df_bowl_sr , df_bowl_avg]:
    df_bat_bowl_features= df_bat_bowl_features.merge(l,on= ['match_id'],how='left')

df_bat_bowl_features.to_csv(r'../Processed/df_bat_bowl_features.csv',index= False)




### tEAM LEVEL FEATURES
Team_level_WR = df_matches[(df_matches['Winner'] != 'no result') & (df_matches['Winner'] != 'tied')]



def Team_win_rate(row):
    T1 = row['Team 1']
    T2 = row['Team 2']
    odi = row['odi_id']
    df1 = df_matches[((df_matches['Team 1'] == T1) & (df_matches['Team 2'] == T2)) | ((df_matches['Team 1'] == T2) & (df_matches['Team 2'] == T1))]
    df = df1[df1['odi_id']<odi]
    if len(df) > 20:
        df = df.iloc[len(df)-20:len(df)]
    if (len(df)==0):
        return 0,0
        
    T1_Win_rate = len(df[df['Winner']== T1])/len(df)
    T2_Win_rate = len(df[df['Winner']== T2])/len(df)
    return T1_Win_rate,T2_Win_rate

Team_level_WR[['T1_WR','T2_WR']] = df_matches.apply(Team_win_rate , axis =1, result_type = 'expand')
Team_level_WR.to_csv(r'../Processed/Team_level_WR.csv',index= False)

#### PITCH LEVEL FEATURES


def pitch_features(row):    
     match = row['match_id']
     
     try:
        bowl_scorecard = pd.read_csv("../Raw_Data/Scorecard/bowling_{}.csv".format(match))
        bat_scorecard = pd.read_csv("../Raw_Data/Scorecard/batting_{}.csv".format(match))
        bowl_scorecard['overs1'] = bowl_scorecard['overs'].map(lambda x : float(x))
        bowl_scorecard['Balls'] = bowl_scorecard['overs1'].map(lambda x : ball_in_over(x))
        
        
        T1 = bat_scorecard['Team'].iloc[0]
        T2 = bat_scorecard['Team'].iloc[len(bat_scorecard)-1]
       
        
        df = bat_scorecard[bat_scorecard['Team']==T2]
        not_out_cnt = len(df[df['dismissal']=='not out'])
        
        T1_sc = bowl_scorecard[bowl_scorecard['Oppositionteam']==T1]['conceded'].sum()
        T2_sc = bowl_scorecard[bowl_scorecard['Oppositionteam']==T2]['conceded'].sum()
        T2_balls = bowl_scorecard[bowl_scorecard['Oppositionteam']==T2]['Balls'].sum()
        
        if (not_out_cnt == 1):
               score  =  max(T1_sc, T2_sc)
        else :
            score  =  max(T1_sc, (T2_sc*300)/T2_balls)
            
     except:
        score = 0
     return  score
 
df_matches['score'] = df_matches.apply(pitch_features, axis = 1)


def ground_impact(row):
    Pitch = row['Ground']
    odi = row['odi_id']
    try:
        df1 = df_matches[df_matches['Ground'] == Pitch]
        df = df1[df1['odi_id']<odi]
        if len(df) > 20:
            df = df.iloc[len(df)-20:len(df)]
        if (len(df)==0):
            pitch_avg_score = 0
        
        pitch_avg_score = df['score'].mean()
        
        df_all_match = df_matches[df_matches['odi_id']<odi]
        if len(df_all_match) > 100:
            df_all_match = df_all_match.iloc[len(df_all_match)-100:len(df_all_match)]
        if (len(df_all_match)==0):
            in_all_avg_score = 0
        
        in_all_avg_score = df_all_match['score'].mean()
        
        pitch_impact_factor = pitch_avg_score/in_all_avg_score
    except:
        pitch_impact_factor = 0
        
    return pitch_impact_factor 

df_matches['pitch_impact'] = df_matches.apply(ground_impact, axis = 1)
df_matches.to_csv(r'../Processed/df_ground_impact_factor.csv',index= False)



