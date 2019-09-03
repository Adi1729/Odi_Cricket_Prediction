''' output: Returns playing XI for given match_id
    input: match_id is taken from file 'all_matches csv'

Data would be stored in below format(sample):
   
    column :  'team1'
    value: {'Australia': {'WM Lawry': '1150', 'KR Stackpole': '1278', 'IM Chappell': '1243', 
                          'KD Walters': '1277', 'IR Redpath': '1215', 'GS Chappell': '1364', 
                          'RW Marsh': '1361', 'AA Mallett': '1319', 'GD McKenzie': '1152',
                          'AN Connolly': '1207', 'AL Thomson': '1363'}}
    column : 'team2'
    value: {'England'  : {'G Boycott': '1228', 'JH Edrich': '1203', 'KWR Fletcher': '1316', 
                          "BL D'Oliveira": '1283', 'JH Hampshire': '1334', 'MC Cowdrey': '998',
                          'R Illingworth': '1081', 'APE Knott': '1301', 'JA Snow': '1270', 
                          'K Shuttleworth': '1362', 'P Lever': '1365'}}	
    
    column: 'match_id'
    value: 64148

'''


import sys
sys.path.append(r'/home/aditya/Cricet Clairvoyant/src/')
from start_scrap import *


def get_lineup(match_id):
        
    match = Match(match_id)
    
    team1 = match.team_1
    team2 = match.team_2
    team2['team_general_name']
    
    player_id =[]
    players = []

    for player in team1['player']:
        players.append(player['card_long'])
        player_id.append(player['object_id'])
    
    team1_dict={}    
    team1_dict[team1['team_general_name']]= dict(zip(players,player_id))
    
    team2 = match.team_2
    team2['team_general_name']
    
    player_id =[]
    players = []
    for player in team2['player']:
        players.append(player['card_long'])
        player_id.append(player['object_id'])
    
    team2_dict={}    
    team2_dict[team2['team_general_name']]= dict(zip(players,player_id))
    
    
    df= pd.DataFrame(data={'team1':[team1_dict],
                       'team2':[team2_dict]})
    df['match_id']=match_id
    
    return df

if __name__ =='__main__':

    dirname = os.path.dirname(__file__)
    cs = pd.read_csv(os.path.join(dirname, '../Raw_Data/Match/all_matches_vF.csv'))
    cs['match_id']=cs['match_id'].astype('int')
    lineup=pd.DataFrame()
    df= pd.read_csv(os.path.join(dirname, '../Raw_Data/Match/match_lineup_v5.csv'))
    df['match_id']=df['match_id'].astype('int')
    lineup=df
    
    
   
#    cs=pd.read_csv(r'/home/aditya/Cricket Clairvoyant/Raw_Data/Match/all_matches.csv')
    st_time = time.time()
   
    for idx,match_id in enumerate(cs['match_id']):
        if((idx+1)%10==0 or len(cs)==(idx+1)):
            print('Lineup collected for %d matches..   Time Taken: %.2f min'%(idx+1,(time.time()-st_time)/60))
            lineup.to_csv(os.path.join(dirname, '../Raw_Data/Match/match_lineup_vF.csv'),index= False)   
#        if cs['match_id'].iloc[idx] in df['match_id'].tolist():
#                continue
        if cs['odi_id'].iloc[idx]< 4140:
             continue
     
        try:
            lineup=lineup.append(get_lineup(match_id))
        except:
            print('##### %d'%match_id)
            continue
    
    lineup.to_csv(os.path.join(dirname, '../Raw_Data/Match/match_lineup_vF.csv'),index= False)            
    print('\n Total Time taken : %d secs\n'%(time.time()-st_time))
        
        
