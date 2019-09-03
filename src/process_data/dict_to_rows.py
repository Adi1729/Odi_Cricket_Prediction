#%reset -f
import pandas as pd
import os
import re
from scipy import stats
os.chdir(r'./Cricket Clairvoyant/src')

import sys
from player_record_v4 import *
import time

#Loading the all match dataframe

df_matches = pd.read_csv("../Raw_Data/Match/all_matches_vF.csv")
lineup =  pd.read_csv("../Raw_Data/Match/match_lineup_vF.csv")

lineup = lineup.drop_duplicates()

lineup['team1'] = lineup['team1'].apply(lambda x: eval(x))
lineup['team2'] = lineup['team2'].apply(lambda x: eval(x))

def match_squad(match_id,index):
    
    team1= list(lineup[lineup['match_id']==match_id]['team1'][index].values())[0]
    team2= list(lineup[lineup['match_id']==match_id]['team2'][index].values())[0]
    player = [x for x,y in team1.items()] + [x for x,y in team2.items()] 
    player_id2 = [y for x,y in team1.items()] +  [y for x,y in team2.items()]
    team = ['T1'] * int(len(player)/2) + ['T2'] * (len(player)- int(len(player)/2)) 

    return pd.DataFrame(data = {'Players':player, 'player_id':player_id2 , 'match_id': match_id, 'team' : team})

squad_11_all_matches = pd.DataFrame()
for rownum,row in lineup.iterrows():
    match_id = row['match_id']
    df = match_squad(match_id,rownum)
    squad_11_all_matches= squad_11_all_matches.append(df)

squad_11_all_matches.to_csv("../Raw_Data/Match/squad_11_all_matches_vF.csv",index=False)

