import sys
sys.path.append(r'/home/aditya/Cricket Clairvoyant/src/')
from start_scrap import *
from player_record_v4 import *
os.chdir(r'/home/aditya/Cricket Clairvoyant/src/')
from player_utility import *
import numpy as np
from scipy import stats

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
    
def get_bowler_impact(player_id = None, odi_id = None):

    try:
        p = pd.read_csv(r'../Raw_Data/Players/bowling_%d.csv'%player_id)    
    except:
        p = player_record(player_id = player_id, role_type = 'bowling')
        p.to_csv(r'../Raw_Data/Players/bowling_%d.csv'%player_id,index = False)
        
    p = p[p['odi_id']<odi_id]
    
    p = get_bowler_features(player_df = p)
    p['Weighted_Wickets'] = p['Wkts']
   
    for idx in range(len(p)):
        try:
            w_o =eval(p['Wicket_Order'].iloc[idx])        
            s=0
            
            for w in w_o :
                s = s + batting_order_weight[int(w)]
            p['Weighted_Wickets'].iloc[idx] = s 
        except:
            continue
        
    p['cum_weights'] = p.Weighted_Wickets.cumsum()
    p['impact_score'] =np.square(p['cum_weights'] * 1000/ (p['cum_Wkts'] * p['bowl_SR']))
    
    return p['impact_score'].iloc[len(p)-1]


matches = pd.read_csv('../Raw_Data/Match/all_matches_vF.csv')
wc_match_id = matches[matches['odi_id']> 4140]['match_id'].tolist()

impact_factors = ['T1_Bowler1_impact_factor','T1_Bowler2_impact_factor',
         'T1_Bowler3_impact_factor','T1_Bowler4_impact_factor',
         'T2_Bowler1_impact_factor','T2_Bowler2_impact_factor',
         'T2_Bowler3_impact_factor','T2_Bowler4_impact_factor'
         ]



lineup = pd.read_csv(r'../Raw_Data/Match/squad_11_all_matches_vF.csv')
match_ids = list(lineup['match_id'].unique())
impact_file = pd.read_csv('../Processed/impact_factor_bowler_v4.csv')

Teams = list(teams_dict.keys())
Teams = [re.sub('[^a-z]','',team.lower()) for team in Teams]
Teams = [team for team in Teams if team!='namibia']
      
model_df = impact_file
random =[225171,64375,65672,65067,65066,65038,64951,64949,64149]

rej_list = [1152841,1169330]

#match_ids = [1144529,1144530,1144531,1144532,1144533]
file_ids = [ids for ids in model_df.match_id if ids not in wc_match_id]

id_Df = pd.DataFrame(data={'match_id':file_ids})
impact_file = impact_file.merge(id_Df ,on = 'match_id',how= 'inner')

wc_match_id = [ids for ids in wc_match_id if ids not in  [1168517,1168518]]
model_df = impact_file


for match_idx in range(len(match_ids)):
    impact = []
    match_id = match_ids[match_idx]
    print(match_id)
    impact_both_teams = []
 
    
#    if match_id in impact_file['match_id'].tolist() or match_id in rej_list :
    if match_id not in wc_match_id :
        continue

    match_lineup = lineup[lineup['match_id']==match_id]
    odi_id = matches[matches['match_id']==match_id]['odi_id'].values[0]
    team1 =  matches[matches['match_id']==match_id]['Team 1'].values[0].lower()
    team2 =  matches[matches['match_id']==match_id]['Team 2'].values[0].lower()
    team1 = re.sub('[^a-z]','',team1.lower())
    team2 = re.sub('[^a-z]','',team2.lower())
    
    if team1 not in Teams or team2 not in Teams:
        continue

    
    match_lineup['Teams']=['T1'] * int(len(match_lineup)/2) + ['T2'] *  (len(match_lineup) - int(len(match_lineup)/2))
    
    
    for team in ['T1','T2']:
        
        pos_list =[]
        player_list = []
        count_list=[]
        
        team_lineup =match_lineup[match_lineup['Teams']==team].reset_index(drop = True)
        for idx,players in enumerate(team_lineup['Players']):
            impact =[]    
            player_id = team_lineup['player_id'].iloc[idx]
            try:
                p= pd.read_csv('../Raw_Data/Players/bowling_%d.csv'%player_id)
            except:
                p = player_record(player_id = player_id, role_type = 'bowling')
                p.to_csv('../Raw_Data/Players/bowling_%d.csv'%player_id,index=False)
            odi = matches[matches['match_id']==match_id]['odi_id'].values[0]
            p_filter =p[p['odi_id']<odi]
            p_filter["Wkts"] = p_filter.Wkts.map(lambda x : str(x).replace('-','0')).astype(int)
            
            if sum(p_filter['Wkts']) >0:
                try:
                    res = stats.mode(p_filter['Pos'][len(p_filter)-20:len(p_filter)])
                except:
                    res = stats.mode(p_filter['Pos'])
                try:
                    pos =  int(res[0][0])
                    count = res[1][0]
                    pos_list.append(pos)
                    count_list.append(count)
                    player_list.append(player_id)
                except:
                    pass
        
        order_df1 = pd.DataFrame(data = {'Pos':pos_list,
                                        'Freq':count_list,
                                        'player':player_list})
            
        order_df1 = order_df1.sort_values(by = ['Pos','Freq'], ascending = [True,False]).reset_index(drop=True)
        
        for order_idx in range(0,min(4,order_df1.shape[0]),1):
            
            player_id = order_df1['player'].iloc[order_idx]
            q=get_bowler_impact(player_id = player_id, odi_id = odi_id)
            impact.append(q)
            
        while(len(impact)<4):
            try:
                impact.append(sum(impact)/len(impact))
            except:
                impact.append(0)
            
        impact_both_teams = impact_both_teams + impact 
       
    impact_dict = dict(zip(impact_factors,impact_both_teams))
    impact_dict.update({'match_id':match_id})
    
    model_df = model_df.append(pd.DataFrame([impact_dict]))
    print(' %d matches .. '%match_idx)
    model_df.to_csv(r'../Processed/impact_factor_bowler_vF.csv',index= False)   
#
#    if((match_idx+1)%10==0):
#        print(' %d matches .. '%match_idx)
#        model_df.to_csv(r'../Processed/impact_factor_batsman_v2.csv',index= False)   
#          
    
model_df.to_csv(r'../Processed/impact_factor_bowler_vF.csv',index= False)  

model_df.match_id.value_counts()
 