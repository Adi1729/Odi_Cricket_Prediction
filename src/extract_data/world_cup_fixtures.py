

'''
####
        Purpose of code:
        1. This code generates world cup fixtures for all 10 teams in world cup. : 45 matches
        2. Features captured:  Player Name, ID, Age, Playing Role, Status : Captain/StandBy
        
###
'''

#import packaages

import start_scrap

def wc_squad():
        
    main_url = 'http://www.espncricinfo.com/scores/series/8039/season/2019/icc-cricket-world-cup'
    
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
        players= bs.find_all('div',{'class':'cscore_link cscore_link--button'})
    
        
        for idx,player in enumerate(players):
            squad.append(re.sub('\n|\r|\t','',player.a.get_text()))
            link.append(player.a.get('href'))
            team.append(team_squad[team_idx+1])
            match_sched = players[0].find_all({'div':'cscore_overview'})
            match=  players[0].find_all({'a':'cscore_details'})
            
        for i in range(0,len(match_sched)):
            try:
                match_sched[i].find_all({'div':'cscore_info'}).text
                print(i)
            except:
                pass
            
            for details in player_details:
                info = details.text
                
                if ':' in info:
                    
                    

def extract_tournament_stats(seasons,id_,tournament):
    
    main_url = 'http://www.espncricinfo.com/scores/series/8039/season/2019/icc-cricket-world-cup'
    data_list = []
    
    url =main_url
    html  = urllib.request.urlopen(url, context = ctx).read()
    bs = BeautifulSoup(html,'lxml')
    a_container=bs.find_all('div',{'class' :'cscore_link cscore_link--button'})

    
    for idx,matches in enumerate(a_container):
        temp_data=OrderedDict()
        temp_data['Match_info']= a_container[idx].find('div', {'class':'cscore_info-overview'}).text    
        temp_data['Match_info']= a_container[idx].find('div', {'class':'cscore_info-overview'}).text    
        temp_data['Match_info']= a_container[idx].find('div', {'class':'cscore_info-overview'}).text    
      
        temp_data['match_link'] =a_container[idx].find('a', {'class':'cscore_details'}).get('href')
        home = a_container[idx].find('li', {'class':'cscore_item cscore_item--home'})
        temp_data['team1'] = home.find('span', {'class':'cscore_name cscore_name--long'}).text
#            temp_data['score1'] = home.find('div', {'class':'cscore_score'}).text
        away = a_container[idx].find('li', {'class':'cscore_item cscore_item--away'})
        temp_data['team2'] = away.find('span', {'class':'cscore_name cscore_name--long'}).text
#            temp_data['score2'] = away.find('div', {'class':'cscore_score'}).text
        data_list.append(temp_data)

        
    df=pd.DataFrame(data_list)

    df['Match_No']= df['Match_info'].apply(lambda x : x.split(',')[0])
    df['Ground']= df['Match_info'].apply(lambda x : x.split(',')[1].split('at')[1].strip())
    df['Date']= df['Match_info'].apply(lambda x : x.split(',')[2].strip())
    df['match_id'] =df['match_link'].apply(lambda x : re.findall('\d+',x)[1])       
            

    df.to_csv('../Raw_Data/Match/world_cup_fixtures_vF.csv',index= False)
    

squad
'''
To Do 
Get age, playing roles, batting , bwoling, or if standy , captain, wicket keeper
Data Dictionary
'''