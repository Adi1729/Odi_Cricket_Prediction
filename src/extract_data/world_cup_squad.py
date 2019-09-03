
'''
####
        Purpose of code:
        1. This code generates world cup squad for all 10 teams in world cup.
        2. Features captured:  Player Name, ID, Age, Playing Role, Status : Captain/StandBy
        
###
'''

#import packaages

import start_scrap

def wc_squad():
        
    main_url = 'http://www.espncricinfo.com/ci/content/squad/index.html?object=1144415'
    
    url =main_url
    html  = urllib.request.urlopen(url, context = ctx).read()
    
    bs = BeautifulSoup(html,'lxml')
    teams=bs.find_all('a')
    
    team_squad =[]
    teams_link=[]
    
    #extracting teams playing in world cup
    for team in teams:
        if 'Squad' in team.text:
            print(idx)
            team_squad.append(re.sub('Squad','',team.text).strip())
            teams_link.append(team.get('href'))
    
    parent_url ='http://www.espncricinfo.com'
    
    squad =[]
    link=[]
    team=[]
    
    #extracting player names in world cup squad
    team_idx = 3
    for team_idx,team_link in enumerate(teams_link[1:]):
            
        team_url = parent_url + team_link
        html  = urllib.request.urlopen(team_url, context = ctx).read()
        bs = BeautifulSoup(html,'lxml')
        players= bs.find_all('div',{'class':'large-13 medium-13 small-13 columns'})
    
        
        for idx,player in enumerate(players):
            squad.append(re.sub('\n|\r|\t','',player.a.get_text()))
            link.append(player.a.get('href'))
            team.append(team_squad[team_idx+1])
            player_details = player[idx].find_all('span')
            
            for details in player_details:
                info = details.text
                
                if ':' in info:
                    
                    
                    
            
            
        player_id = [re.findall('\d+',player_link)[0] for player_link in link]


    squad=pd.DataFrame(list(zip(squad,player_id,team)),columns =['player','player_id','country'])


    a=bs.find_all('div',{'class':'large-13 medium-13 small-13 columns'})
    for w in q:
        print(w.text)
            


    return squad


if __name__ =='__main__':
    
    dirname = os.path.dirname(__file__)
    st_time = time.time()
    squad = wc_squad()
    squad.to_csv(os.path.join(dirname, '../Raw_Data/Team/squad.csv'),index= False)
    print('Time taken : %d secs'%(time.time()-st_time))
    
    

'''
To Do 
Get age, playing roles, batting , bwoling, or if standy , captain, wicket keeper
Data Dictionary
'''