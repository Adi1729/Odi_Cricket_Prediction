from start_scrap import *

def ranking(role_type):
     
        
        for year in range(2019,2020):
    
            print('Scrapping %s ranking for year : %d..'%(role_type,year))
            
            for month in range(1,8):
                
#                if 'ranking_%s_%d_%d.csv'%(role_type,month,year) in os.listdir(os.path.join(dirname, '../Raw_Data/Ranking/')):
#                    continue
    
                output_rows = []

                main_url = 'http://www.relianceiccrankings.com/datespecific/odi/?stattype=%s&day=07&month=%d&year=%d'  
                url = main_url%(role_type,month,year)                
                html  = urllib.request.urlopen(url, context = ctx).read()
                bs = BeautifulSoup(html,'lxml')

                try:
                    header= [th.get_text() for th in bs.find_all('tr')[1].find_all('th')]
                except: 
                    continue

                header = ['ranking','rating','player','team_abb','career best rating']

                for table_row in bs.findAll('tr'):
                    columns = table_row.findAll('td')
                    output_row = []
                    for idx,column in enumerate(columns):
                        try:                   
                            if column.get('title').isupper():
                                output_row.append(column.get('title'))
                            else:
                                output_row.append(column.text)
                        except:
                            pass

                    if(len(header)==len(output_row)):
                        output_rows.append(output_row)
               
                
                df = pd.DataFrame(columns=header, data=output_rows)
                df['team']=df['team_abb'].apply(lambda x :country_dict[x] if x in country_dict.keys() else 'Others')
                df['player_id']=df[['player','team']].apply(assign_id,axis=1)
                df.to_csv(('../Raw_Data/Ranking2/ranking_%s_%d_%d.csv'%(role_type,month,year)),index= False)
 
#                df.to_csv(os.path.join(dirname, '../Raw_Data/Ranking2/ranking_%s_%d_%d.csv'%(role_type,month,year)),index= False)


def name_clean(x):
    player = re.sub('\.','',x).strip()   
    player = re.sub('[^a-zA-Z]','',player).strip()   
    player = re.sub('\s+','',player).strip()   
    return player.lower()
    
def assign_id(x):
    player = re.sub('\.','',x[0]).strip()   
    player = re.sub('[^a-zA-Z]','',player).strip()   
    player = re.sub('\s+','',player).strip()   
    player =player.lower()
    
    try:
        df1 = cs[cs['country']==x[1]]
        player_id = cs[cs['player']==player]['player_id'].values[0]
    except:
        if x[1]!='Others':
            print('Country : %s , Player : %s'%(x[1],player))
        return None
    
    return player_id
    
if __name__ =='__main__':
    
    
    dirname = os.path.dirname(__file__)
    cs = pd.read_csv(os.path.join(dirname, '../Raw_Data/Team/complete_squad_v2.csv'))
    cs = pd.read_csv(r'/home/aditya/Cricket Clairvoyant/Raw_Data/Team/complete_squad_v2.csv')
    
    cs['player'] = cs['player'].apply(name_clean)

    st_time = time.time()

    role_type = 'bowling'
    ranking(role_type)
    ranking(role_type).to_csv(('../Raw_Data/Ranking2/ranking_%s_%d_%d.csv'%(role_type,month,year)),index= False)
   
       
    print('\nTime taken : %d secs\n'%(time.time()-st_time))


