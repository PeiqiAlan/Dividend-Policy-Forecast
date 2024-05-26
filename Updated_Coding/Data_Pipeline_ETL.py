import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Get the current working directory
current_directory = os.getcwd()
print("Current directory:", current_directory)

path_env = r"E:\United Commonwealth of Planets\Cabinet\Department of Development\Dividend_Policy\Dividend_Policy\.env.txt"
load_dotenv(path_env)  # Load environment variables from the file

API_KEY_FMP = os.environ.get('API_KEY_FMP')  # Retrieve the value of the environment variable 'API_KEY_FMP'

class Data_collection():
    def __init__(self, frame_original, three_cluster_with_numerical, two_cluster):
        self.list_company = []
        self.origin = frame_original
        self.two_cluster = two_cluster
        self.three_cluster = three_cluster_with_numerical

class Data_Extractor():

    def __init__(self, ticker, frequency, offset_limit, start_yr=2013, end_yr=2022):
        self.company = ticker
        self.frequency = frequency
        self.limit = offset_limit
        self.BASE_URL = "https://financialmodelingprep.com/api/v3"
        self.start_year = start_yr
        self.end_year = end_yr
        self.year_list = list(range(self.start_year, self.end_year))
        self.Ratio_column_name = None

    def extract_data_dividend(self):
        endpoint_url = f"{self.BASE_URL}/historical-price-full/stock_dividend/{self.company}?apikey={API_KEY_FMP}"
        response = requests.get(endpoint_url)  # Get the response
        if response.status_code == 250:  # Check the status of the response, 429 means the API limit has been reached
            print("FMP API limit reached")  # Print the result for easy debugging later
        response_dict = response.json()
        dividends = pd.DataFrame(response_dict['historical'])
        # Data Transformation
        if dividends.shape == (0, 0):  # Handle the case where the company never issued any dividend in the past
            dividends = pd.DataFrame({
                "Year": list(range(self.start_year - 1, self.end_year + 1)),
                "adjDividend": [0.0] * len(list(range(self.start_year - 1, self.end_year + 1))),
                "Company": self.company
                # We are obtaining 2 more years' data
            })
        else:
            # Extract year data from the date column
            dividends['Year'] = pd.to_datetime(dividends['date']).dt.year
            # Aggregate the dividend paid by year
            dividends = dividends.groupby("Year").agg({"adjDividend": "sum"}).reset_index()
            # Create a new DataFrame with all years from start to end - So that we don't omit years without dividends
            all_years = pd.DataFrame({'Year': list(range(self.start_year - 1, self.end_year + 1))})
            # Merge the two DataFrames on the year column and fill missing values with 0.0
            dividends = all_years.merge(dividends, on='Year', how='left').fillna(0.0)
            dividends['Company'] = self.company

        return dividends

    def extract_data_cashflow(self):
        url2 = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{self.company}?period={self.frequency}&limit={self.limit}&apikey={API_KEY_FMP}"
        response_cash = requests.get(url2)
        if response_cash.status_code == 250:  # Check the status of the response, 429 means the API limit has been reached
            print("FMP API limit reached")  # Print the result for easy debugging later
        dict_cash = response_cash.json()
        cash_flow = pd.DataFrame(dict_cash)

        return cash_flow

    def get_sector_info_qualitative(self):
        # Engineer some other predictors
        predictors = pd.DataFrame({"year": list(range(self.start_year - 1, self.end_year))})  # We include one more year before
        # the first year to calculate changes
        company_data_raw = yf.Ticker(self.company)

        if company_data_raw is not None:
            company_data = company_data_raw.info
            predictors["industry"] = company_data.get('industry', 'Others')
            predictors["sector"] = company_data.get('sector', 'Others')
        else:
            predictors["industry"] = 'Others'
            predictors["sector"] = 'Others'

        predictors.rename(columns={'year': 'Year'}, inplace=True)
        predictors['Company'] = self.company

        return predictors



    def engineer_dps_change(self, frame, num_cluster=3):

        if num_cluster == 3:
            frame['next_year_dividend'] = frame['adjDividend'].shift(-1)
            conditions1 = [
                frame['adjDividend'] == frame['next_year_dividend'],
                frame['adjDividend'] != frame['next_year_dividend'],
            ]

            conditions2 = [
                frame['adjDividend'] < frame['next_year_dividend'],
                frame['adjDividend'] > frame['next_year_dividend'],
            ]

            choices1 = ['constant', 'changed']
            choices2 = ['increased', 'decreased']

            # Create the target column 'dps_change' based on the conditions
            frame['dps_change_next_year'] = np.select(conditions1, choices1, default=np.nan)
            # Now map the second set of conditions to the column with choices2,
            # but only for entries where the 'dps_change_next_year' is 'changed'
            frame['dps_change_next_year'] = np.where(frame['dps_change_next_year'] == 'changed',
                                                     np.select(conditions2, choices2, default=np.nan),
                                                     frame['dps_change_next_year'],)
        elif num_cluster == 2:
            frame['next_year_dividend'] = frame['adjDividend'].shift(-1)

            conditions = [
                frame['adjDividend'] <= frame['next_year_dividend'],
                frame['adjDividend'] > frame['next_year_dividend']
            ]

            choices = ['constant/increased', 'decreased']

            # Create the target column 'dps_change' based on the conditions
            frame['dps_change_next_year'] = np.select(conditions, choices, default=np.nan)

        return frame

    def engineer_dps_change_numeric(self, frame):
        frame['dps_growth_absolute'] = frame['next_year_dividend'] - frame['adjDividend']
        frame['dps_growth_rate'] = np.where((frame['next_year_dividend'] == 0) & (frame['adjDividend'] == 0), 0,  # If both are 0 then change is 0
            np.where(frame['next_year_dividend'] != 0, ((frame['next_year_dividend'] / frame['adjDividend']) - 1), 999)  # If last year dividend is 0 then return 999
        )

        return frame

    def Ratio_predictors(self):
        BASE_URL = 'https://financialmodelingprep.com/api/v3'
        company_tick = self.company

        endpoint_ratios = f"{BASE_URL}/ratios/{company_tick}?apikey={API_KEY_FMP}"
        endpoint_yield = f"{BASE_URL}/key-metrics/{company_tick}?apikey={API_KEY_FMP}"


        end_response = requests.get(endpoint_ratios)
        if end_response.status_code == 429:
            print("FMP API limit reached")
        #print(type(end_response))

        end_response2 = requests.get(endpoint_yield)
        if end_response2.status_code == 429:
            print("FMP API limit reached")
        #print(type(end_response2))

        # Convert json to dictionary object and then a Pandas Dataframe
        response_dicts = end_response.json()
        payout = pd.DataFrame(response_dicts)

        response_dicts3 = end_response2.json()
        div_yield = pd.DataFrame(response_dicts3)

        if not self.Ratio_column_name:
            endpoint_ratios_trial = f"{BASE_URL}/ratios/AAPL?apikey={API_KEY_FMP}" # AAPL always has the information, this is to extract a list of column names for the zeros placeholder dataframe in the later if statement
            endpoint_yield_trial = f"{BASE_URL}/key-metrics/AAPL?apikey={API_KEY_FMP}"
            end_response_trial = requests.get(endpoint_ratios_trial)
            if end_response_trial.status_code == 429:
                print("FMP API limit reached")
            # print(type(end_response))

            end_response2_trial = requests.get(endpoint_yield_trial)
            if end_response2_trial.status_code == 429:
                print("FMP API limit reached")
            # print(type(end_response2))

            # Convert json to dictionary object and then a Pandas Dataframe
            response_dicts_trial = end_response_trial.json()
            payout_trial = pd.DataFrame(response_dicts_trial)

            response_dicts3_trial = end_response2_trial.json()
            div_yield_trial = pd.DataFrame(response_dicts3_trial)
            Ratios_Frame_trial = payout_trial.merge(div_yield_trial, on=['symbol', 'calendarYear'], how='left').fillna(0.0)
            Ratios_Frame_trial = Ratios_Frame_trial.loc[:, ~Ratios_Frame_trial.columns.isin(['date_x', 'period_x', 'date_y', 'period_y'])].copy()

            self.Ratio_column_name = Ratios_Frame_trial.columns

        # Merge the two DataFrames on the year column and fill missing values with 0.0
        if payout.shape == (0,0) or div_yield.shape == (0,0):
            year_list = list(range(self.start_year - 1, self.end_year + 1))
            n_rows = len(year_list) # Under the situation where there is no available information, the number of rows stay the same as our yearly data
            Ratios_Frame = pd.DataFrame(np.zeros(shape=(n_rows, len(self.Ratio_column_name)))) # If there is no information, fill with zeros as placeholders
            Ratios_Frame.columns = self.Ratio_column_name
            Ratios_Frame = Ratios_Frame.loc[:, ~Ratios_Frame.columns.isin(['date_x', 'period_x', 'date_y', 'period_y'])].copy()
            Ratios_Frame.rename(columns={'symbol': 'Company', 'calendarYear': 'Year'}, inplace=True)
            Ratios_Frame['Company'] = self.company
            Ratios_Frame['Year'] = list(range(self.start_year - 1, self.end_year + 1))
        else:
            Ratios_Frame = payout.merge(div_yield, on=['symbol', 'calendarYear'], how='left').fillna(0.0)
            Ratios_Frame = Ratios_Frame.loc[:, ~Ratios_Frame.columns.isin(['date_x', 'period_x', 'date_y', 'period_y'])].copy()
            Ratios_Frame.rename(columns={'symbol': 'Company', 'calendarYear': 'Year'}, inplace=True)
            Ratios_Frame['Year'] = Ratios_Frame['Year'].astype('int64')

        return Ratios_Frame

    def Main_Pipeline(self):
        dividend1 = self.extract_data_dividend()
        data_cluster3 = self.engineer_dps_change(dividend1, 3)
        data_cluster3 = self.engineer_dps_change_numeric(data_cluster3)
        Data_Ratio = self.Ratio_predictors()
        Data_aggregate = data_cluster3.merge(Data_Ratio, on=['Year', 'Company'], how='left').fillna(np.nan)
        Data_sector = self.get_sector_info_qualitative()
        Data_aggregate = Data_aggregate.merge(Data_sector, on=['Year', 'Company'], how='left').fillna(np.nan)

        # Drop the last row as usually there is nan for growth in the most recent year due to lack of data
        # NAN problem shouldn't arise elsewhere
        if Data_aggregate.iloc[-1,:].isna().sum() > 0:
            Data_aggregate.drop(Data_aggregate.index[-1], inplace=True)
        #if Data_aggregate.iloc[-1,:].isna().sum()
        Data_all = Data_collection(dividend1, Data_aggregate, np.nan)

        return Data_all


'''
# Construction Case
tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
ticker_table = tables[0]
company_tick = ticker_table['Symbol'].tolist()
#company_tick = ['AAPL', 'APA', 'ACGL', 'ANET', 'AXON', 'GOOGL', 'BRK.B', 'BXP']
existing_company = []
dataset = []
dictionary_company_data = {}




E1 = Data_Extractor('BRK.B', 'annual', offset_limit=300)
Data_all = E1.Main_Pipeline()



for tick in company_tick:
    print(tick)
    if tick in existing_company:
        continue

    E1 = Data_Extractor(tick, 'annual', offset_limit=300)
    Data_all = E1.Main_Pipeline()
    dictionary_company_data[tick] = Data_all.three_cluster

    if len(dataset) == 0:
        dataset = Data_all.three_cluster
    elif len(dictionary_company_data[tick]) == 0:
        pass
    else:
        dataset = pd.concat([dataset, Data_all.three_cluster])

    existing_company.append(tick)


'''
