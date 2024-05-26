import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

path_env = r"E:\United Commonwealth of Planets\Cabinet\Department of Development\Dividend_Policy\Dividend_Policy\.env.txt"
load_dotenv(path_env)  # Load environment variables from the file

API_KEY_FMP = os.environ.get('API_KEY_FMP')  # Retrieve the value of the environment variable 'API_KEY_FMP'

class Label_encoding():
    def __init__(self, df_dummy, start_yr=2010, end_yr=2022):
        self.BASE_URL = "https://financialmodelingprep.com/api/v3"
        self.start_year = start_yr
        self.end_year = end_yr
        self.DataFrame = df_dummy

    def label_encode_sector_pe(self):
        #list_industry = list(self.DataFrame.industry.unique())
        list_year = list(self.DataFrame.Year.unique())
        PE_mean_sector = self.DataFrame.groupby(['Year', 'sector'])['peRatio'].mean().reset_index()
        PE_mean_sector_sort = []

        for yr in list_year:
            PE_mean_sector_sort_loop = PE_mean_sector.loc[PE_mean_sector['Year'] == yr].sort_values(by='peRatio', ascending=False).copy()
            PE_mean_sector_sort_loop['sector_ranking'] = PE_mean_sector_sort_loop['peRatio'].rank(ascending=True)
            PE_mean_sector_sort_loop['sector_ranking'] = PE_mean_sector_sort_loop['sector_ranking'] - 1
            if len(PE_mean_sector_sort) == 0:
                PE_mean_sector_sort = PE_mean_sector_sort_loop
            else:
                PE_mean_sector_sort = pd.concat([PE_mean_sector_sort, PE_mean_sector_sort_loop])

        return PE_mean_sector_sort[['Year','sector','sector_ranking']].copy()


    def label_encode_industry_pe(self):
        list_year = list(self.DataFrame.Year.unique())
        PE_mean_industry = self.DataFrame.groupby(['Year', 'industry'])['peRatio'].mean().reset_index()
        PE_mean_industry_sort = []
        for yr in list_year:
            PE_mean_industry_sort_loop = PE_mean_industry.loc[PE_mean_industry['Year'] == yr].sort_values(by='peRatio', ascending=False).copy()
            PE_mean_industry_sort_loop['industry_ranking'] = PE_mean_industry_sort_loop['peRatio'].rank(ascending=True)
            PE_mean_industry_sort_loop['industry_ranking'] = PE_mean_industry_sort_loop['industry_ranking'] - 1
            if len(PE_mean_industry_sort) == 0:
                PE_mean_industry_sort = PE_mean_industry_sort_loop
            else:
                PE_mean_industry_sort = pd.concat([PE_mean_industry_sort, PE_mean_industry_sort_loop])

        PE_mean_industry_sort['industry_ranking'] = PE_mean_industry_sort['industry_ranking'].fillna(0.0).copy() # Because the lower the ranking, the lesser the PE ratios.
                                                                            # A lack of data means highly suspectable, you shouldn't be making investments in the first place

        return PE_mean_industry_sort[['Year', 'industry', 'industry_ranking']].copy()

        # Define a dataframe of industries and year and PEs

class Count_nan():
    def __init__(self, df_dummy, threshold_dummy, threshold_col):
        self.df = df_dummy
        self.threshold_nan = threshold_dummy
        self.threshold_col_nan = threshold_col

    list_company_nan_values = []

    def special_case_target(self):
        self.df['dps_growth_rate'] = self.df['dps_growth_rate'].replace(np.inf, 999).copy()

    def count_nans_and_inf(self):
        # Replace inf values with NaN for easier counting
        df_replaced = self.df.replace([np.inf, -np.inf], np.nan)

        # Group by 'company' and apply a function to count NaNs
        result = df_replaced.groupby('Company').apply(lambda x: x.isna().sum())

        companies_nan_rows = result[result >= self.threshold_nan].copy()
        list_companies = companies_nan_rows.loc[companies_nan_rows.sum(axis=1) > 0].index.tolist()

        return list_companies

    def remove_nans(self, list_rows):
        df_nan = self.df.loc[self.df['Company'].isin(list_rows)].copy()
        percentage_of_nan_col = df_nan.isna().mean(axis=1)
        list_index_remove = percentage_of_nan_col[percentage_of_nan_col > self.threshold_col_nan].index.tolist()
        df_filtered = self.df[~self.df.index.isin(list_index_remove)].copy()
        return df_filtered



class Feature_selection():
    def __init__(self, df_dummy, threshold_dummy):
        self.df = df_dummy
        # Correlation matrix
        self.threshold = threshold_dummy

    def Column_ranking_correlation(self):
        df_numerical = self.df.iloc[:, 7:].copy()
        df_numerical = df_numerical.drop(columns=['industry', 'sector']).copy()
        corr_matrix = df_numerical.corr()
        # Initializing a list to hold the tuples (col1, col2, correlation)
        correlations = []

        corr_matrix_TF = corr_matrix.abs() > self.threshold
        corr_matrix_TF_Series = corr_matrix_TF.stack()
        correlation_higher_threshold = corr_matrix_TF_Series[corr_matrix_TF_Series].reset_index()
        correlation_higher_threshold.columns = ['Row', 'Column', 'Correlation_Higher']

        # To filter out the ones due to cross-correlations
        correlation_higher_threshold_unique = correlation_higher_threshold.loc[correlation_higher_threshold['Row'] != correlation_higher_threshold['Column']].copy()


        return correlation_higher_threshold_unique


class Check_data_imbalance():
    def __init__(self, df_dummy):
        self.df = df_dummy

    def Check_Imbalance_Targets(self):
        df = self.df
        df_count = df.groupby('dps_change_next_year').count()

        # Interactive mode on
        plt.ion()
        plt.figure(figsize=(10,6))
        bar_labels = df_count.index.tolist()
        bar_colors = ['tab:red', 'tab:blue', 'tab:orange']
        plt.bar(df_count.index.tolist(), df_count['Year'].tolist(), label=bar_labels, color=bar_colors)

        # Add titles and labels
        plt.title('Distribution of Targets')
        plt.xlabel('Categorical Targets')
        plt.ylabel('Frequency')
        plt.legend()

        # Save the chart to a directory
        output_path = 'Distribution_of_Targets.png'
        plt.savefig(output_path)





'''
df = pd.read_csv("Stock_data.csv")

label_encoder = Label_encoding(df)
Encoder_industry = label_encoder.label_encode_industry_pe()
Encoder_sector = label_encoder.label_encode_sector_pe()
'''
