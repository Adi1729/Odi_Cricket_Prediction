squad = pd.read_csv('../Raw_Data/Team/squad.csv')

matches = pd.read_csv('../Raw_Data/Match/world_cup_fixtures.csv')
squad['Playing_11'] = squad['Playing_11'].astype(str)

squad['Playing_11'] = squad['Playing_11'].apply(lambda x : x.lower())
squad = squad[squad['Playing_11']=='in'].reset_index()
squad['country'] = squad['country'].apply(lambda x :  re.sub('[^a-z]','',x.lower()))
squad['player_id'] = squad['player_id'].astype(int)

from collections import OrderedDict
main_df = pd.DataFrame()

for idx in range(len(matches)):

    team1 = matches['team1'].iloc[idx]
    team2 = matches['team2'].iloc[idx]
    team1 = re.sub('[^a-z]','',team1.lower())
    team2 = re.sub('[^a-z]','',team2.lower())
    
    player_lineup  = squad[squad['country']==team1]['player'].tolist() +  squad[squad['country']==team2]['player'].tolist()
    player_id_lineup  = squad[squad['country']==team1]['player_id'].tolist() +  squad[squad['country']==team2]['player_id'].tolist() 
    match_id = [matches['match_id'].iloc[idx]] * 22
    team = ['T1'] * 11 + ['T2'] * 11
    country = squad[squad['country']==team1]['country'].tolist() + squad[squad['country']==team2]['country'].tolist()
    
    try:
            
        df= pd.DataFrame({'Players':player_lineup,
                      'player_id2':player_id_lineup,
                      'match_id':match_id,
                      'team':team,
                      'country':country})
           
        main_df= main_df.append(df)
    except:
        print(team1)
        print(team2)
        
 
    main_df.to_csv(r'../Raw_Data/Match/world_cup_lineup.csv',index= False)