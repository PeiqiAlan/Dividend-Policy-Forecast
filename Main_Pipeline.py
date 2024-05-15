import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from Data_Pipeline_ETL import Data_Extractor

tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
ticker_table = tables[0]
company_tick = ticker_table['Symbol'].tolist()
#company_tick = ['AAPL', 'APA', 'ACGL', 'ANET', 'AXON', 'GOOGL', 'BRK.B', 'BXP']
existing_company = []
dataset = []
dictionary_company_data = {}


'''

E1 = Data_Extractor('BRK.B', 'annual', offset_limit=300)
Data_all = E1.Main_Pipeline()

'''

for tick in company_tick:
    print('Now processing for company ', tick)
    if tick in existing_company:
        continue

    E1 = Data_Extractor(tick, 'annual', offset_limit=300, start_yr=2010)
    Data_all = E1.Main_Pipeline()
    dictionary_company_data[tick] = Data_all.three_cluster

    if len(dataset) == 0:
        dataset = Data_all.three_cluster
    elif len(dictionary_company_data[tick]) == 0:
        pass
    else:
        dataset = pd.concat([dataset, Data_all.three_cluster])

    existing_company.append(tick)

print('Loading Complete')
# Save data to disk
dataset.to_csv("Stock_data.csv", index=False)