'''http://www.espncricinfo.com/ci/content/player/index.html
http://stats.espncricinfo.com/wi/engine/records/team/match_results.html?class=2;id=1995;type=year
'''

import time
import os
import urllib
from bs4 import BeautifulSoup
import ssl
import pandas as pd
from collections import OrderedDict
import re
import os


def all_match_list():
        
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    output_rows = []
        
    for year in range(1971,2020):
        
        print('Scrapping matches for %d..\n'%year)
        main_url = 'http://stats.espncricinfo.com/wi/engine/records/team/match_results.html?class=2;id=%d;type=year'  
        url = main_url%year
        html  = urllib.request.urlopen(url, context = ctx).read()
        
        bs = BeautifulSoup(html,'lxml')
        header= [th.get_text() for th in bs.find_all('thead')[0].find_all('th')]
        header[-1]='odi_id'
        header.append('match_link')
        
        for table_row in bs.findAll('tr'):
            columns = table_row.findAll('td')
            output_row = []
            for idx,column in enumerate(columns):
                output_row.append(column.text)
                
                if(idx==(len(header)-2)):
                    output_row.append(column.a.get('href'))
                    
            if(len(header)==len(output_row)):
                output_rows.append(output_row)
   
        
    df = pd.DataFrame(columns=header, data=output_rows)
    
    return df


if __name__ =='__main__':
    
    dirname = os.path.dirname(__file__)
    st_time = time.time()
    
    df = all_match_list()
    df['odi_id'] = df['odi_id'].apply(lambda x: re.findall('\d+',x)[0])   # converting 'odi # 1234' -> '1234'
    df['match_id'] = df['match_link'].apply(lambda x: re.findall('\d+',x)[0])   # converting 'odi # 1234' -> '1234'
   
    df.to_csv(os.path.join(dirname, '../Raw_Data/Match/all_matches_vF.csv'),index= False)
    print('\nTime taken : %d secs\n'%(time.time()-st_time))
   
    print('File Generated : %s ,  Path : %s\n\n ' %('all_matches.csv',r'../Raw_Data/Match/'))
   
    

    




