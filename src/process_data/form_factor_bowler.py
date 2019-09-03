'''source: 
    'https://www.researchgate.net/publication/327388396_Analysis_of_Performance_of_Bowlers_Using_Combined_Bowling_Rate'
'''

import sys
sys.path.append(r'/home/aditya/Cricket Clairvoyant/src/')
from start_scrap import *
from player_record_v4 import *
os.chdir(r'/home/aditya/Cricket Clairvoyant/src/')
from player_utility import *
import numpy as np
from scipy import stats
import statistics
import time

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
    

matches = pd.read_csv('../Raw_Data/Match/all_matches_v2.csv')

form_index_features = ['T1_Bowler1_form_index','T1_Bowler2_form_index',
         'T1_Bowler3_form_index','T1_Bowler4_form_index',
         'T1_Bowler5_form_index' , 
         'T2_Bowler1_form_index','T2_Bowler2_form_index',
         'T2_Bowler3_form_index','T2_Bowler4_form_index' ,
         'T2_Bowler5_form_index'  
         ]

lineup = pd.read_csv(r'../Raw_Data/Match/squad_11_all_matches.csv')
match_ids = list(lineup['match_id'].unique())
form_file = pd.read_csv('../Processed/form_index.csv')

Teams = list(teams_dict.keys())
Teams = [re.sub('[^a-z]','',team.lower()) for team in Teams]
      
model_df = form_file

for match_idx in range(0,1000,1):
    impact = []
    match_id = match_ids[match_idx]
    print(match_id)
    form_both_teams = []
 
    
    if match_id in form_file['match_id'].tolist() or match_id in [213080,217481,217811,217978,223334,223902,226352,226375
                            ,226381,226382,236963,237222,237574,238169,238170]:
        continue
    
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
    for team in ['T1','T2']:
        
        pos_list =[]
        player_list = []
        count_list=[]
        form_list=[]
        form_list_df=[]
        
        
        team_lineup =match_lineup[match_lineup['Teams']==team].reset_index(drop = True)
        for idx,players in enumerate(team_lineup['Players']):
            player_id = team_lineup['player_id2'].iloc[idx]
            try:
                p= pd.read_csv('../Raw_Data/Players/bowling_%d.csv'%player_id)
            except:
                p = player_record(player_id = player_id, role_type = 'batting')
                p.to_csv('../Raw_Data/Players/bowling_%d.csv'%player_id,index=False)
            odi = matches[matches['match_id']==match_id]['odi_id'].values[0]
            p['odi_id'] = p['odi_id'].astype(int)
            p =p[p['odi_id']<odi]
      
            p_filter = p
            
            if len(p) ==0:
                continue
      
            p = get_bowler_features(player_df = p)
            
            p['bow_avg_sr_eco'] = p["bowl_Avg"] * p["bowl_SR"] * p["bowl_Eco"]
            bat_avg_sr =  p['bow_avg_sr'].iloc[len(p)-1]
            
            p_filter =p_filter[p_filter['odi_id']<odi]
#            p_filter['Runs1']= p_filter['Runs'].astype(str).map(lambda x : re.sub("[^0-9a-z]","",x)).astype(int)
#            p_filter['Runs'] =p_filter['Runs'].astype(str)
#            p_filter['DNB']= p_filter['Runs'].map(lambda x : 1 if x.upper() in ['DNB','TDNB','*','SUB','ABSENT'] else 0 )
#           
#            p_filter = p_filter[p_filter['DNB']==0]            
            
            if len(p_filter) == 0:
                continue            
           
            if len(p_filter) > 20:
                p_filter = p_filter[len(p_filter)-20:]            
           
            p_filter = get_batsman_features(player_df = p_filter[p.columns.tolist()])
            p_filter['bat_avg_sr'] = p_filter['bat_avg'] * p_filter['bat_SR']
            
            rec_avg_sr =  p_filter['bat_avg_sr'].iloc[len(p_filter)-1]
            rec_avg = p_filter['bat_avg'].iloc[len(p_filter)-1]
           
            try:
                res = stats.mode(p_filter['Pos'][len(p_filter)-20:len(p_filter)])
            except:
                res = stats.mode(p_filter['Pos'])
            
            try:
                pos =  int(res[0][0])
                count = res[1][0]
                sd =  statistics.stdev(p_filter['Runs1'].astype(int))
                form_index = (rec_avg /sd) * rec_avg_sr/bat_avg_sr
                form_list.append(form_index)
                pos_list.append(pos)
                count_list.append(count)
                player_list.append(player_id)
                   
            except:
                continue
            
            order_df1 = pd.DataFrame(data = {'Pos':pos_list,
                                            'Freq':count_list,
                                            'player':player_list,
                                            'form_index':form_list})
                
            order_df1 = order_df1.sort_values(by = ['Pos','Freq'], ascending = [True,False]).reset_index(drop=True)
            
            try:
                form_list_df = order_df1['form_index'].to_list()[:5]
            except:
                form_list_df = order_df1['form_index'].to_list()
         
                
        
        while(len(form_list_df)<5):
            try:
                form_list_df.append(sum(form_list_df)/len(form_list_df))
            except:
                form_list_df.append(np.nan)
            
        form_both_teams = form_both_teams + form_list_df 
   
       
    form_dict = dict(zip(form_index_features,form_both_teams))
    form_dict.update({'match_id':match_id})
    
    model_df = model_df.append(pd.DataFrame([form_dict]))
    
    if((match_idx+1)%10==0):
        print(' %d matches ..  Time Taken %0.2f '%(match_idx,st-time.time()))
        model_df.to_csv(r'../Processed/form_index.csv',index= False)   
          