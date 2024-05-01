import os
from dotenv import load_dotenv
import requests
import pandas as pd

# Get the current working directory
current_directory = os.getcwd()
print("Current directory:", current_directory)

path_env = r"YOUR_PATH_HERE\.env.txt"
load_dotenv(path_env)  # Load environment variables from the file

API_KEY_FMP = os.environ.get('API_KEY_FMP')  # Retrieve the value of the environment variable 'API_KEY_FMP'
#print(API_KEY_FMP)

BASE_URL = "https://financialmodelingprep.com/api/v3"
company_tick = "YOUR_COMPANY_TICKER_NAME" # Demo only

endpoint_url = f"{BASE_URL}/historical-price-full/stock_dividend/{company_tick}?apikey={API_KEY_FMP}"


class Data_Extractor:
    def __init__(self, ticker):
        self.company = ticker

    def extract_data(self):
        response = requests.get(endpoint_url)  # Get the response
        if response.status_code == 250:  # Check the status of the response, 429 means the API limit has been reached
            print("FMP API limit reached")  # Print the result for easy debugging later
        response_dict = response.json()
        dividends = pd.DataFrame(response_dict['historical'])
        return dividends


E1 = Data_Extractor(company_tick)
dividend1 = E1.extract_data()
print(dividend1)
