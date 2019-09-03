
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

def Scorecard(x):
        dfbat = pd.DataFrame()
        dfbowl = pd.DataFrame()
        for i in range(10):
            try:
                x1 = Match(x).html
                break
            except:
                print('iteration %d'%i)
                continue
        time.sleep(2)
        
        try:
            x2 = json.loads(x1.find_all('script')[13].get_text().replace("\n", " ").replace('window.__INITIAL_STATE__ =','').replace('&dagger;','wk').replace('&amp;','').replace('wkts;','wkts,').replace('wkt;','wkt,').strip().replace('};', "}};").split('};')[0])
        except:
            print('index number : 12')
            x2 = json.loads(x1.find_all('script')[12].get_text().replace("\n", " ").replace('window.__INITIAL_STATE__ =','').replace('&dagger;','wk').replace('&amp;','').replace('wkts;','wkts,').replace('wkt;','wkt,').strip().replace('};', "}};").split('};')[0])
            
        df1bat = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['1']['batsmen'])
        d1title = x2['gamePackage']['scorecard']['innings']['1']['title']
        df1bat['Team'] = ' '.join(d1title.split(' ')[:-1])
        df2bat = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['2']['batsmen'])
        d2title = x2['gamePackage']['scorecard']['innings']['2']['title']
        df2bat['Team'] = ' '.join(d2title.split(' ')[:-1])
        df1bat['Oppositionteam'] = ' '.join(d2title.split(' ')[:-1])
   
        df1bat['WicketOrder'] = df1bat.index + 1
        df2bat['Oppositionteam'] =  ' '.join(d1title.split(' ')[:-1])
        df2bat['WicketOrder'] = df2bat.index + 1
        
        Finaldf_bat = pd.concat([df1bat.drop(['captain','commentary','runningScore','runningOver', 'stats','hasVideoId','href','isNotOut','roles','trackingName'], axis=1),
           df1bat.stats.apply(flatten)], axis=1).append(pd.concat([df2bat.drop(['captain','commentary','runningScore','runningOver', 'stats','hasVideoId','href','isNotOut','roles','trackingName'], axis=1),
                                                               df2bat.stats.apply(flatten)], axis=1))
        Finaldf_bat['city'] = Match(x).town_name
        Finaldf_bat['date'] = Match(x).date
        Finaldf_bat['MatchId'] = x
        Finaldf_bat['WicketTakenBy'] = Finaldf_bat['shortText']
        Finaldf_bat['WicketTakenBy'] = Finaldf_bat['WicketTakenBy'].str.extract(r'\b(\w+)$', expand=True)
        Finaldf_bat['StatsType']='Batting'

        dfbat=pd.concat([dfbat,Finaldf_bat])
        cols = dfbat.columns.tolist()
        cols= [col[1] if len(col)==2 else col for col in dfbat.columns.tolist()]
        dfbat.columns = cols      
        dfbat.rename(columns = {'shortText':'dismissal'},inplace = True)     


        df1bowl = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['1']['bowlers'])
        d1title = x2['gamePackage']['scorecard']['innings']['1']['title']
        df2bowl = pd.DataFrame(x2['gamePackage']['scorecard']['innings']['2']['bowlers'])
        d2title = x2['gamePackage']['scorecard']['innings']['2']['title']
        df1bowl['Team'] = d2title.split(' ')[0]
        df2bowl['Team'] = d1title.split(' ')[0]
        df1bowl['Oppositionteam'] = ' '.join(d1title.split(' ')[:-1])
        df2bowl['Oppositionteam'] = ' '.join(d2title.split(' ')[:-1])
        
        Finaldf_bowl = pd.concat([df1bowl.drop(['captain','stats','hasVideoId','href','roles','trackingName'], axis=1),
                       df1bowl.stats.apply(flatten)], axis=1).append(pd.concat([df2bowl.drop(['captain','stats','hasVideoId','href','roles','trackingName'], axis=1),
                                                               df2bowl.stats.apply(flatten)], axis=1))
        Finaldf_bowl['city'] = Match(x).town_name
        Finaldf_bowl['date'] = Match(x).date
        Finaldf_bowl['MatchId'] = x
        Finaldf_bowl['BowlerLastName'] = Finaldf_bowl['name'].str.extract(r'\b(\w+)$', expand=True)
        Finaldf_bowl['StatsType']='Bowling'
        dfbowl=pd.concat([dfbowl,Finaldf_bowl])
        
        
        wiket_order_df=pd.merge(dfbowl, dfbat, left_on='BowlerLastName', right_on='WicketTakenBy', how='left')
        wiket_order_df_2=pd.DataFrame(wiket_order_df.groupby('name_x')['WicketOrder'].apply(list))
        wiket_order_df_2['name']=wiket_order_df_2.index    
        dfbowl_wicket_order=pd.merge(dfbowl, wiket_order_df_2, on='name', how='left')
        
        cols = dfbowl_wicket_order.columns.tolist()
        cols= [col[1] if len(col)==2 else col for col in dfbowl_wicket_order.columns.tolist()]
        dfbowl_wicket_order.columns = cols      
   

#        
#        dfbowl_wicket_order.to_csv("home/aditya/Cricket Clairvoyant/Raw Data/Match/bowling_stats_match_id_"+x+".csv")
        
        return dfbat, dfbowl_wicket_order


if __name__ =='__main__':
    
    dirname = os.path.dirname(__file__)
    all_match = pd.read_csv(os.path.join(dirname, '../Raw_Data/Match/all_matches_vF.csv'))
#    all_match = pd.read_csv(r'../Raw_Data/Match/all_matches.csv')
#    match_not_found = pd.read_csv(os.path.join(dirname, '../Raw_Data/Match/match_not_found.csv'))

    st_time = time.time()
  
    print('\nScrapping stats for %d matches...\n\n'%(all_match.shape[0]))
    
    start_idx = input('Enter starting index : ')
#        all_match.shape[0]
    for idx in range(all_match.shape[0]):
        
        if all_match['odi_id'].iloc[idx] < 4142:
            continue
        
        if((idx +1)% 10 == 0):
            print('%d matches processed : %d left   Time Taken : %.2f'%(idx+1,all_match.shape[0]-(idx+1),(time.time()-st_time)/60))
#        if(all_match['Winner'].iloc[idx]=='no result'):
#            continue
#           
        match_id = all_match['match_id'].iloc[idx]
#        match_id = match_not_found['match_id'].iloc[idx]
#        print(match_id)
        bat_file ='batting_%s.csv'%(match_id)
        bowl_file ='bowling_%s.csv'%(match_id)
        
        try:
            if bowl_file not in os.listdir(os.path.join(dirname, '../Raw_Data/Match/Scorecard2/')):
                bat_df,bowl_df =Scorecard(match_id)
                bat_df.to_csv(os.path.join(dirname, '../Raw_Data/Match/Scorecard2/batting_%s.csv'%(match_id)),index= False)
                bowl_df.to_csv(os.path.join(dirname, '../Raw_Data/Match/Scorecard2/bowling_%s.csv'%(match_id)),index= False)
        except:
            print('######################### %d'%match_id)
            continue
        
    print('\nTime taken : %d secs\n'%(time.time()-st_time))
    
    

    
##
#bat_df = dfbat
#bowl_df =  dfbowl_wicket_order
#bat_df.to_csv(r'../Raw_Data/Match/Scorecard/batting_%s.csv'%(match_id),index= False)
#bowl_df.to_csv('../Raw_Data/Match/Scorecard/bowling_%s.csv'%(match_id),index= False)
##         