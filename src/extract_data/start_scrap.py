#use for scraping web data
import time
import os
import urllib
from bs4 import BeautifulSoup
import ssl
import pandas as pd
from collections import OrderedDict
import re
import os


ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#use in match scorecard.py file
from espncricinfo.summary import Summary
from espncricinfo.match import Match 
from espncricinfo.series import Series

import json
import requests
from espncricinfo.exceptions import MatchNotFoundError, NoScorecardError


country_dict={
        'IND':'India',
        'SL':'Sri Lanka',
        'ENG':'England',
        'SA':'South Africa',
        'NZ':'New Zealand',
        'BAN':'Bangladesh',
        'WI':'West Indies',
        'AFG':'Afghanistan',
        'PAK':'Pakistan',
        'AUS':'Australia',
        
        }

#gives weights to each wicket depending on position

batting_order_weight={
    1 : 1.5,
    2 : 1.4,
    3 : 1.5,
    4 : 1.3,
    5 : 1.2,
    6 : 0.9,
    7 : 0.7,
    8 : 0.5,
    9 : 0.5,
    10: 0.3,
    11: 0.2
}

