'''source: 
    'https://www.researchgate.net/publication/327388396_Analysis_of_Performance_of_Bowlers_Using_Combined_Bowling_Rate'
'''

import sys
sys.path.append(r'C:/Users/Dhawad/Desktop/Cricket Clairvoyant/src/')
from start_scrap import *
from player_record_v4 import *
os.chdir(r'C:/Users/Dhawad/Desktop/Cricket Clairvoyant/src/')
from player_utility import *
import numpy as np
from scipy import stats
from datetime import datetime
import pandas as pd
import statistics

st = time.time()
teams_dict={
         'India':'6',
         'Australia':'2',
         'Bangladesh':'25',
         'England':'1',
         'New Zealand':'5',
         'Pakistan':'7',
         'South Africa':'3',
         'Sri Lanka':'8',
         'West Indies':'4',
         'Afghanistan' :'40',
         'Zimbabwe' : '9',
         'Ireland' : '29',
         'Namibia' : '28'
         }
    

matches = pd.read_csv('../Raw_Data/Match/all_matches_vF.csv')
matches = matches[matches.odi_id>4142]


#form_index_features = ['T1_Bowler1_form_index','T1_Bowler2_form_index',
#         'T1_Bowler3_form_index','T1_Bowler4_form_index',
#         'T1_Bowler5_form_index' , 
#         'T2_Bowler1_form_index','T2_Bowler2_form_index',
#         'T2_Bowler3_form_index','T2_Bowler4_form_index' ,
#         'T2_Bowler5_form_index'  
#         ]

lineup = pd.read_csv(r'../Raw_Data/Match/squad_11_all_matches_vF1.csv')

match_ids = list(matches['match_id'].unique())

#form_file = pd.read_csv('../Processed/form_index.csv')

Teams = list(teams_dict.keys())
Teams = [re.sub('[^a-z]','',team.lower()) for team in Teams]
      
#form_factor_df=pd.DataFrame(data={'T1_player1_form_factor'=[],'T1_player2_form_factor'=[],'T1_player3_form_factor'=[],'T1_player4_form_factor'=[],
#'T1_player5_form_factor'=[],'T2_player1_form_factor'=[],'T2_player2_form_factor'=[],'T2_player3_form_factor'=[],'T2_player4_form_factor'=[],'T2_player5_form_factor'=[]})
#
#

form_factor_df=pd.DataFrame()
error_matches=[]       
for match_idx in range(0,len(match_ids),1):
    
    try:
        #match
        impact = []
        match_id = match_ids[match_idx]
        print("processing ",match_id)
        print(len(match_ids) - match_idx, "matches left")
 
     
    
        match_lineup = lineup[lineup['match_id']==match_id]
        odi_id = matches[matches['match_id']==match_id]['odi_id'].values[0]
        team1 =  matches[matches['match_id']==match_id]['Team 1'].values[0].lower()
        team2 =  matches[matches['match_id']==match_id]['Team 2'].values[0].lower()
        team1 = re.sub('[^a-z]','',team1.lower())
        team2 = re.sub('[^a-z]','',team2.lower())
        
        if team1 not in Teams or team2 not in Teams:
            continue
     
        match_lineup['Teams']=['T1'] * int(len(match_lineup)/2) + ['T2'] *  int(len(match_lineup)/2)
        form_both_teams = []
        pos_df=pd.DataFrame(data={'Pos':[],'Freq':[],'form_factor':[],'player_id':[],'Team':[]})
        for team in ['T1','T2']:
            #team
            
            pos_list =[]
            player_list = []
            count_list=[]
            form_list=[]
            form_list_df=[]
            team_list=[]
            
            
            team_lineup =match_lineup[match_lineup['Teams']==team].reset_index(drop = True)
            for idx,players in enumerate(team_lineup['Players']):
                #players
                player_id = team_lineup['player_id'].iloc[idx]
                try:
                    p= pd.read_csv('../Raw_Data/Players_kn/bowling_%d.csv'%player_id)
                except:
                    p = player_record(player_id = player_id, role_type = 'bowling')
                    p.to_csv('../Raw_Data/Players_kn/bowling_%d.csv'%player_id,index=False)
                odi = matches[matches['match_id']==match_id]['odi_id'].values[0]
                p['odi_id'] = p['odi_id'].astype(int)
                p_filter =p[p['odi_id']<odi]
                p_filter['DNB']= p_filter['Overs'].apply(lambda x : 1 if x in ['DNB','TDNB','sub','absent'] else 0 )
                if len(p_filter) > 20:
                    p_filter = p_filter[len(p_filter)-20:]
                p_filter = p_filter[p_filter['DNB']==0]           
                
                if len(p_filter) ==0:
                    continue
          
    
               
    
                    
                try:
                    res = stats.mode(p_filter[p_filter['Pos']!='-']['Pos'][len(p_filter)-20:len(p_filter)])
                except:
                    res = stats.mode(p_filter[p_filter['Pos']!='-']['Pos'])
                
                try:
                    pos =  int(res[0][0])
                except:
                    continue
                
                if(pos>4):
                    continue
                
                
                p_filter = get_bowler_features(player_df = p_filter)
    
                
                
                def func(x):
                    r=float(x[0])
                    w=float(x[1])
                    o=float(x[2])
                    
                    return(3*r/((w)+(w*r/6*o)+(o)))
                
                p_filter['hm']=p_filter[['Runs','Wkts','Overs']].apply(func,axis=1)
                    
                try:
                    sd =  statistics.stdev(p_filter['hm'].astype(float))
                    mean= statistics.mean(p_filter['hm'].astype(float))
                    count = res[1][0]
                    pos_list.append(pos)
                    count_list.append(count)
                    form_index = (mean /sd)
                    form_list.append(form_index)
                    player_list.append(player_id)
                    team_list.append(team)
                
                except:
                    continue
    #                pos_list.append(pos)
    #                count_list.append(count)
    #                player_list.append(player_id)
    #                team_list.append(team)
    #                form_list.append(np.nan)
    
                
            pos_df = pos_df.append(pd.DataFrame(data = {'Pos':pos_list,
                                                    'Freq':count_list,
                                                    'player_id':player_list,
                                                    'form_factor': form_list,
                                                    'Team': team_list}))      
            pos_df = pos_df.sort_values(by = ['Pos','Freq'], ascending = [True,False]).reset_index(drop=True)
                
        
        
        form_factor={}    
        for team in ['T1','T2']:
            pos_df1= pos_df[pos_df.Team==team].reset_index()
    
            for rownum,row in pos_df1.iterrows():
                if(rownum<5):
                    form_factor['{}_player{}_from_factor'.format(team,rownum+1)]=row['form_factor']
                
        form_factor['match_id']=match_id
        form_factor_df=form_factor_df.append(pd.DataFrame([form_factor]))
        form_factor_df.to_csv('../Processed/form_factor.csv')
    except:
        error_matches.append(match_id)

form_factor_df.to_csv('../Processed/form_factor_submission_final.csv')