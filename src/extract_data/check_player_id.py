#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 00:10:13 2019

@author: aditya
"""
import os
import pandas as pd
a=os.listdir('../Raw_Data/Ranking')

a=a[:10]
path  = r'../Raw_Data/Ranking/'
df_main=pd.DataFrame(columns = ['ranking', 'rating', 'player', 'team_abb', 'career best rating', 'team','player_id'])
columns = ['ranking', 'rating', 'player', 'team_abb', 'career best rating', 'team','player_id']
file_issue=[]
size=[]
i=0
file =a[0]

cs = pd.read_csv(r'../Raw_Data/Team/complete_squad_v2.csv')
cs['player'] = cs['player'].apply(name_clean)

new_file  = os.listdir(r'../Raw_Data/Ranking_v2/')
for file in a :
    if file not in new_file:
            
        df= pd.read_csv(path+file)
        try:
            df['team']=df[['team','team_abb']].apply(lambda x : teams_dict[x[1]] if x[1] in teams_dict.keys() else x[0],axis=1) 
            df['player_id']= df[['player','team','player_id']].apply(assign_id,axis=1)
            df.to_csv(r'../Raw_Data/Ranking_v2/%s'%file,index= False)
        except:
            print(file)
            continue
df_main=df_main[df_main.columns.tolist()[7:]]

no_id =  df_main[df_main['player_id'].isnull()]

no_id  = no_id[['player','team','team_abb']].drop_duplicates()


teams_dict={
         'ZIM' : 'Zimbabwe',
         'IRE' : 'Ireland',
         'Namibia' : 'Namibia'
         }

def assign_team(x):
    if x['team_abb'] in teams_dict.keys():
        teams_dict[x['team_abb']]
    return x['team']
    

df['team']=df[['team,team_abb']].map(assign_team) 
   

def name_clean(x):
    player = re.sub('[^a-zA-Z]','',x).strip()   
    return player.lower()
    

def assign_id(x):
    if x[1] in ['Zimbabwe','Ireland'] :
        
        player = re.sub('\.','',x[0]).strip()   
        player = re.sub('[^a-zA-Z]','',player).strip()   
        player = re.sub('\s+','',player).strip()   
        player =player.lower()
        
        try:
            df1 = cs[cs['country']==x[1]]
            player_id = cs[cs['player']==player]['player_id'].values[0]
        except:
            if x[1]!='Others':
                print('Country : %s , Player : %s'%(x[1],player))
            return None
        
        return player_id
    else:
        return x[2]


no_id['player_id']=no_id[['player','team']].apply(assign_id,axis=1)
no_id  = no_id[no_id['team']!='Others']

