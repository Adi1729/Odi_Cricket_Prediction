'''
####
        Purpose of code:
        1. This code will generate bowling, batting and fielding stats for players .
        2. Input : Player ID
        
###
'''

import urllib
from bs4 import BeautifulSoup
import ssl
import pandas as pd
from collections import OrderedDict
import re
import os
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def get_wicket_order(player_df = None, player_id = None):
        
    squad = pd.read_csv('../Raw_Data/Team/complete_squad_v2.csv')
    match = pd.read_csv('../Raw_Data/Match/all_matches.csv')
    match['odi_id'] =match['odi_id'].astype('int')
    player_df['odi_id'] =player_df['odi_id'].astype('int')
    
    player_df=player_df.merge(match[['odi_id','match_id']],on=['odi_id'],how='left')
    player_name = squad[squad['player_id']==player_id]['player'].values[0]
    player_name = player_name.lower()
    player_name = re.sub('[^a-z]','',player_name)
    
    player_df['Wicket_Order']=None
    for idx in range(player_df.shape[0]):
        
        match_id  = player_df['match_id'].iloc[idx] 
        over  = player_df['Overs'].iloc[idx]
        
        if re.findall('\d+',over):
           try:
                df = pd.read_csv('../Raw_Data/Match/Scorecard/bowling_%d.csv'%match_id)
                df['name']=df['name'].apply(lambda x: x.lower())
                df['clean_name'] = df['name'].apply(lambda x : re.sub('[^a-z]','',x))
                wicket_order=df[df['clean_name']==player_name]['WicketOrder'].values[0]
                player_df['Wicket_Order'].iloc[idx] =wicket_order
           except:
                pass
        else:
            continue
    return player_df['Wicket_Order']            

def player_record(player_id = None,role_type = 'batting'):
        
    main_url = 'http://stats.espncricinfo.com/ci/engine/player/%s.html?class=2;template=results;type=%s;view=innings'
    
    url =main_url%(player_id,role_type)
    html  = urllib.request.urlopen(url, context = ctx).read()
    
    bs = BeautifulSoup(html,'lxml')
    
    try :
        header= [th.get_text() for th in bs.find_all('thead')[1].find_all('th')]
    except:
        return pd.DataFrame()
    
    header[-1]='odi_id'
    output_rows = []
    for table_row in bs.findAll('tr'):
        columns = table_row.findAll('td')
        output_row = []
        for column in columns:
            output_row.append(column.text)
        if(len(header)==len(output_row)):
            output_rows.append(output_row)
    
    df = pd.DataFrame(columns=header, data=output_rows)
    df['filter'] = df['odi_id'].apply(lambda x: 1 if 'odi' in x.lower() else 0)   # converting 'odi # 1234' -> '1234'
    df = df[df['filter']==1]
    df = df.drop(columns = 'filter')
    df['odi_id'] = df['odi_id'].apply(lambda x : re.findall('\d+',x)[0])
    df['Opposition'] = df['Opposition'].apply(lambda x: x[2:]) # coverting 'v India' -> 'India'
    
    if role_type == 'bowling':
        df['Wicket_Order'] = get_wicket_order(player_df = df, player_id = player_id)
    
    return df

if __name__ =='__main__':
    
    dirname = os.path.dirname(__file__)
    squad = pd.read_csv(os.path.join(dirname, '../Raw_Data/Team/squad1.csv'))
    st_time = time.time()
    role_type = 'batting'
    print('\nScrapping %s stats for %d players...\n\n'%(role_type,squad.shape[0]))
        
    for idx in range(squad.shape[0]):
        if((idx +1)% 10 == 0):
            print('%d players processed : %d left'%(idx+1,squad.shape[0]-(idx+1)))
        player_id = squad['player_id'].iloc[idx]
        df = player_record(player_id = player_id , role_type = role_type)
        if len(df)==0:
            print('\n%s , %s : %s record not found\n'%(squad['player'].iloc[idx],squad['country'].iloc[idx],role_type))
        else:
            df.to_csv(os.path.join(dirname, '../Raw_Data/%s_%s.csv'%(role_type,player_id)),index= False)
    print('\nTime taken : %d secs\n'%(time.time()-st_time))
    
