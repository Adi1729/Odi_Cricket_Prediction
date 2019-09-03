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

#lineup = pd.read_csv(r'../Raw_Data/Match/df_match_final_data.csv')
#
#lineup['match_id'] = lineup['match_id'].astype(int)
#lineup['odi_id'] = lineup['odi_id'].astype(int)
#

matches = pd.read_csv('../Raw_Data/Match/all_matches_vF.csv')
abc = pd.read_csv('../Raw_Data/Match/squad_11_all_matches_vF.csv')
lineup = matches.merge(abc,on='match_id',how='inner')

model_bat_bowl_df = pd.read_csv(r'../Processed/df_bat_bowl_features_v2.csv')
model_bat_bowl_df = model_bat_bowl_df.merge(matches[['match_id','odi_id']],on=['match_id'],how='left')

bowlers_impact_columns = ['T1_P1_Bowl_avg','T1_P2_Bowl_avg','T1_P3_Bowl_avg','T1_P4_Bowl_avg','T1_P5_Bowl_avg',
                          'T2_P1_Bowl_avg','T2_P2_Bowl_avg','T2_P3_Bowl_avg','T2_P4_Bowl_avg','T2_P5_Bowl_avg']

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
            
def get_opposition_bowl_strength(row):
  
    cols = [cols for cols in p.columns.tolist() if row['opposition_T_'] in cols]
    
    
    bowl_avg = [row[col] for col in cols if row[col]!=0]
    bowl_avg = [avg for avg in bowl_avg if avg!=0]
    

    if len(bowl_avg)>=3:
        return sum(bowl_avg[:3])
    
    return sum(bowl_avg)    

       
    

impact_factors = ['T1_Batsman1_impact_factor','T1_Batsman2_impact_factor',
                 'T1_Batsman3_impact_factor','T1_Batsman4_impact_factor',
                 'T1_Batsman5_impact_factor',
                 'T2_Batsman1_impact_factor','T2_Batsman2_impact_factor',
                 'T2_Batsman3_impact_factor','T2_Batsman4_impact_factor',
                 'T2_Batsman5_impact_factor'
                 ]

match_ids = list(lineup['match_id'].unique())

Teams = list(teams_dict.keys())
Teams = [re.sub('[^a-z]','',team.lower()) for team in Teams]
Teams = [team for team in Teams if team!='namibia']
impact_file = pd.read_csv('../Processed/impact_factor_batsman.csv')      
model_df = impact_file
random =[225171,64375,65672,65067,65066,65038,64951,64949,64149]
rej_list = [66054,66057,66064,66117,66119,66041,65880,66243,65708,601612,65714,66173,226364,65620,
            566942,226376,702139,238174,710307, 65807, 239907, 667889,239912, 756035, 756039,225250,
            810827]
wc_match_id = matches[matches['odi_id']> 4140]['match_id'].tolist()

file_ids = [ids for ids in model_df.match_id if ids not in wc_match_id]

id_Df = pd.DataFrame(data={'match_id':file_ids})
impact_file = impact_file.merge(id_Df ,on = 'match_id',how= 'inner')

wc_match_id = [ids for ids in wc_match_id if ids not in  [1168517,1168518]]
model_df = impact_file


lineup.rename(columns = {'team':'Team'},inplace= True)
for match_idx in range(len(match_ids)):
    impact = []
    match_id = match_ids[match_idx]
    print(match_id)
    impact_both_teams = []
 
    if match_id not in wc_match_id:
        continue
    
    match_lineup = lineup[lineup['match_id']==match_id]
    odi_id = matches[matches['match_id']==match_id]['odi_id'].values[0]
    team1 =  matches[matches['match_id']==match_id]['Team 1'].values[0].lower()
    team2 =  matches[matches['match_id']==match_id]['Team 2'].values[0].lower()
    team1 = re.sub('[^a-z]','',team1.lower())
    team2 = re.sub('[^a-z]','',team2.lower())
    
    if team1 not in Teams or team2 not in Teams:
        continue
    
    for team in ['T1','T2']:
        
        pos_list =[]
        player_list = []
        count_list=[]
        impact_list = []
        impact_list_df =[]
        
        team_lineup =match_lineup[match_lineup['Team']==team].reset_index(drop = True)
        for idx,players in enumerate(team_lineup['Players']):
            impact =[]    
            player_id = team_lineup['player_id'].iloc[idx]
            player_id = int(player_id)
            
            try:
                p= pd.read_csv('../Raw_Data/Players/batting_%d.csv'%player_id)
            except:
                p = player_record(player_id = player_id, role_type = 'batting')
                p.to_csv('../Raw_Data/Players/batting_%d.csv'%player_id,index=False)
            odi = matches[matches['match_id']==match_id]['odi_id'].values[0]
          
            p['odi_id']  = p['odi_id'].astype(int)  
            p = p[p['odi_id']<odi].reset_index()  
         
            if len(p) ==0:
                continue
            
#'''Getting list of matches for a given player with label as T1 and T2 '''
            player_match = lineup[lineup['player_id']==player_id].reset_index()
           
            p = get_batsman_features(player_df = p)
            p = p.merge(player_match[['odi_id','match_id','Team']],on=['odi_id'],how ='left')
            p['Runs'] =p['Runs'].astype(str)
            p['DNB']= p['Runs'].map(lambda x : 1 if x.upper() in ['DNB','TDNB','*','SUB','ABSENT'] else 0 )
           
            p = p[p['DNB']==0]            
       
        
            p = p.merge(model_bat_bowl_df[['odi_id'] + bowlers_impact_columns],on=['odi_id'],how ='left')
            
#            '''Adding opposotion in terms of T1 and T2'''
            p['opposition_T_'] = p['Team'].apply(lambda x: re.sub(x,'','T1T2') if x in ['T1','T2'] else 'No')
            
#            '''bow strength of opposition'''
            if len(p) ==0:
                continue

            p['bow_avg'] = p.apply(lambda x: get_opposition_bowl_strength(x),axis=1)
           
#            '''removing rows for which no bowling data is present'''
            p = p[p['bow_avg'] >0].reset_index() 
            if len(p) ==0:
                continue

            p['impact_batsman_index'] = p['Runs2']/p['bow_avg']   
            
            p['cum_impact'] = p.impact_batsman_index.cumsum()
            p['impact_runs'] = p['cum_impact'] * 1000/p['cum_runs']
        
       
            
            impact_index =  p['impact_runs'].iloc[len(p)-1]
            
         
            if len(p) ==0:
                continue
         
            try:
                res = stats.mode(p['Pos'][len(p)-20:len(p)])
            except:
                res = stats.mode(p['Pos'])
            
            try:
                pos =  int(res[0][0])
                count = res[1][0]
                impact_list.append(impact_index)
                pos_list.append(pos)
                count_list.append(count)
                player_list.append(player_id)
                   
            except:
                continue
    
            order_df1 = pd.DataFrame(data = {'Pos':pos_list,
                                            'Freq':count_list,
                                            'player':player_list,
                                            'form_index':impact_list})
                
            order_df1 = order_df1.sort_values(by = ['Pos','Freq'], ascending = [True,False]).reset_index(drop=True)
            
            try:
                impact_list_df = order_df1['form_index'].to_list()[:5]
            except:
                impact_list_df = order_df1['form_index'].to_list()
         
                
        
        while(len(impact_list_df)<5):
            try:
                impact_list_df.append(sum(impact_list)/len(impact_list))
            except:
                impact_list_df.append(np.nan)
            
        impact_both_teams = impact_both_teams + impact_list_df 
       
    impact_dict = dict(zip(impact_factors,impact_both_teams))
    impact_dict.update({'match_id':match_id})
    
    model_df = model_df.append(pd.DataFrame([impact_dict]))
    model_df.to_csv(r'../Processed/impact_factor_batsman.csv',index= False)   
        
    if((match_idx+1)%10==0 or (match_idx+1 == len(match_ids))):
        print(' %d matches .. Time Taken : %0.2f'%(match_idx,(time.time()-st)))
        model_df.to_csv(r'../Processed/impact_factor_batsman_vF.csv',index= False)   
model_df.to_csv(r'../Processed/impact_factor_batsman_vF.csv',index= False)   
      