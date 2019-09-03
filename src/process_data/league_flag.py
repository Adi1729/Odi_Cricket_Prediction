
''' 
 input: match_id


Batting :
 Data would be stored in below format(sample):
   
        column :  'name'
        value: players name who have batted in a match
        
        column: 'dismissal'
        value : dismissal details if out else not out 
        
        column: 'WicketTakenBy'
        value:  last name of a bolwer who dismissed
        
        column : 
        statstype : batting
    
        other columns: 
             'Team',
             'Oppositionteam',
             'WicketOrder',
             'runs',
             'ballsFaced',
             'minutes',
             'fours',
             'sixes',
             'strikeRate',
             'city',
             'date',
             'MatchId',
             'StatsType'
             
             
    function -> BowlingScorecard
    
    
    

'''



'''
bowling_scorecard

columns:
    
   'name', 'Team', 'Oppositionteam', 'overs', 'maidens', 'conceded',
   'wickets', 'economyRate', 'dots', 'foursConceded', 'sixesConceded',
   'wides', 'noballs', 'city', 'date', 'MatchId', 'BowlerLastName',
   'StatsType', 'WicketOrder'
'''
        

import sys
sys.path.append(r'/home/aditya/Cricket Clairvoyant/src')
from start_scrap import *
###------------------------------##------------------------------------
#This functions helps to expand the list of dictonaries to columns in a dataframe.
def flatten(js):
    return pd.DataFrame(js).set_index(['text','name']).squeeze()

###------------------------------##------------------------------------
### getting the batting scorecard based 
    
def Match_info(x):
        dfbat = pd.DataFrame()
        dfbowl = pd.DataFrame()
        for i in range(10):
            try:
                x1 = Match(x)
                break
            except:
                print('iteration %d'%i)
                continue
#        time.sleep(2)        
#        
#        dfbowl_wicket_order.to_csv("home/aditya/Cricket Clairvoyant/Raw Data/Match/bowling_stats_match_id_"+x+".csv")
        try:
            description = x1.description
        except:
            description = ''
        
        try:
            title = x1.title
        except:
            title = ''
      
        try:
            team1 = x1.team_1_innings['runs'] 
        except:
            team1 =''
    
        try:
            team2  = x1.team_2_innings['runs']
        except:
            team2=''  
        
        return [description,title,team1,team2]


if __name__ =='__main__':
 
    info_label =['Description','Title','T1_Score','T2_Score']

    dirname = os.path.dirname(__file__)
    all_match = pd.read_csv(os.path.join(dirname, '../Raw_Data/Match/all_matches_v2.csv'))
#    all_match = pd.read_csv(r'../Raw_Data/Match/all_matches.csv')
#    match_not_found = pd.read_csv(os.path.join(dirname, '../Raw_Data/Match/match_not_found.csv'))
    Match_df = pd.read_csv(os.path.join(dirname, '../Raw_Data/Match/Match_info.csv'))

    st_time = time.time()
  
    print('\nScrapping stats for %d matches...\n\n'%(all_match.shape[0]))
    
    start_idx = input('Enter starting index : ')
#        all_match.shape[0]
    for idx in range(len(all_match)):
        
        match_id = all_match['match_id'].iloc[idx] 
       
        if match_id in Match_df['match_id'].tolist():
            continue
        print(match_id)
        if((idx +1)% 10 == 0):
            print('%d matches processed : Time Taken : %.2f'%(idx+1,(time.time()-st_time)/60))
        
        info = Match_info(all_match['match_id'].iloc[idx])
        
        dict_match_info = dict(zip(info_label,info))
        dict_match_info.update({'match_id':all_match['match_id'].iloc[idx]})
        Match_df = Match_df.append(pd.DataFrame([dict_match_info]))
       
        Match_df['Description'] = Match_df['Description'].astype(str)
        Match_df['multi_team'] = Match_df['Description'].apply(lambda x : int('tour' not in x.lower()))
           
        if((idx+1)% 10 ==0 or idx == (len(all_match)-1)):
             Match_df.to_csv(os.path.join(dirname, '../Raw_Data/Match/Match_info.csv'),index= False)
             
                 
        
    print('\nTime taken : %d secs\n'%(time.time()-st_time))
    

  