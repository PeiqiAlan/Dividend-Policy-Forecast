import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
from Data_Pipeline_ETL import Data_Extractor
from EDA import Label_encoding, Count_nan, Feature_selection, Check_data_imbalance
from Machine_Learning_Pipeline import Train_val_test_split, Feature_importance, Classification_Model, Hyperparameter_tuning, Regression_models


import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, roc_auc_score, auc

import joblib



def Fetch_new_stock_data(newly_added_tickers):
    existing_company = []
    dataset = []
    dictionary_company_data = {}

    for tick in newly_added_tickers:
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
    df_ref = pd.read_csv("Stock_data.csv")
    columns_ref = list(df_ref.columns)
    if list(dataset.columns) == list(columns_ref):
        # Append the new data to the existing CSV file
        dataset.to_csv("Stock_data.csv", mode='a', header=False, index=False)
        print(f"Data appended to Stock_data.csv")
        return dataset

    else:
        print("Error: Columns do not match. The new data was not appended.")
        sys.exit(1)


def Fetch_stock_data():
    if "Stock_data.csv" not in os.listdir():

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
        return dataset
    else:
        pass


def Preprocess_New_Data(New_data):
    # Get list of companies for reference
    list_company = New_data['Company'].unique()

    # Label encoding sectors given industry pe ratios
    label_encoder = Label_encoding(New_data)
    Encoder_industry = label_encoder.label_encode_industry_pe()
    Encoder_sector = label_encoder.label_encode_sector_pe()
    ## Merge data with label-encoded values for industries and sectors
    New_data = New_data.merge(Encoder_sector, on=['Year', 'sector'], how='left')
    New_data = New_data.merge(Encoder_industry, on=['Year', 'industry'], how='left')

    # Preprocessing data: Replacing nans
    count_nan = Count_nan(df_dummy=New_data, threshold_dummy=5, threshold_col=0.5)
    count_nan.special_case_target()
    list_company_nans = count_nan.count_nans_and_inf()
    df_filtered_new = count_nan.remove_nans(list_company_nans)
    df_filtered_new = df_filtered_new.interpolate(method='linear', axis=0, inplace=True).copy()
    df_filtered_new.to_csv("Filtered_data.csv", mode='a', header=False, index=False)



def Preprocess_Data():

    if "Filtered_data.csv" not in os.listdir():
        df = pd.read_csv("Stock_data.csv")

        # Get list of companies for reference
        list_company = df['Company'].unique()

        # Label encoding sectors given industry pe ratios
        label_encoder = Label_encoding(df)
        Encoder_industry = label_encoder.label_encode_industry_pe()
        Encoder_sector = label_encoder.label_encode_sector_pe()
        ## Merge data with label-encoded values for industries and sectors
        df = df.merge(Encoder_sector, on=['Year', 'sector'], how='left')
        df = df.merge(Encoder_industry, on=['Year', 'industry'], how='left')

        # Preprocessing data: Replacing nans
        count_nan = Count_nan(df_dummy=df, threshold_dummy=5, threshold_col=0.5)
        count_nan.special_case_target()
        list_company_nans = count_nan.count_nans_and_inf()
        df_filtered = count_nan.remove_nans(list_company_nans)
        df_filtered = df_filtered.interpolate(method='linear', axis=0).copy()
        df_filtered.to_csv("Filtered_data.csv", index=False)

    elif "Filtered_data.csv" in os.listdir():

        df_filtered = pd.read_csv("Filtered_data.csv")
        # Get list of companies for reference
        list_company_filtered = df_filtered['Company'].unique()
        # Prompt the user to input company names separated by commas
        company_names_str = input("Enter company names separated by commas (Press Enter if there is no new company): ")

        if not company_names_str.strip():
            print("No new company entered")
            pass
        else:
            # Split the input string into a list of company names
            company_names_list = company_names_str.split(',')

            # Remove leading and trailing whitespaces from each company name
            company_names_list = [name.strip() for name in company_names_list]

            print("Company names as list:", company_names_list)

            # Remove leading and trailing whitespaces from each ticker
            company_tickers = [ticker.strip() for ticker in company_names_list]

            # Initialize a list to store newly added tickers
            newly_added_tickers = []

            # Check each newly inputted ticker if it exists in list_company
            for ticker in company_tickers:
                if ticker not in list_company_filtered:
                    newly_added_tickers.append(ticker)

            if len(newly_added_tickers) == 0:
                print("Company ticker already included")
                pass
            elif len(newly_added_tickers) != 0:
                print("Newly added tickers:", newly_added_tickers)
                New_data = Fetch_new_stock_data(newly_added_tickers)
                Preprocess_New_Data(New_data)
            else:
                print("No New Company Ticker Entered")
                pass



def Remove_features_based_on_correlation(df_filtered_dummy):
    Feature1 = Feature_selection(df_filtered_dummy, 0.9)
    column_info = Feature1.Column_ranking_correlation()

    # Remove highly correlated columns
    columns_to_remove = column_info["Column"].unique()
    df_filtered_dummy.drop(columns_to_remove, axis=1, inplace=True)

    return df_filtered_dummy


def Barplot_feature_importane(df_importance, png_name):
    plt.ion()
    plt.figure(figsize=(16,8))
    ranking = df_importance.sort_values('0')
    sns.barplot(x=ranking['index'], y=ranking['0'])
    plt.xlabel('Labels')
    plt.ylabel('Values')
    plt.title('Feature Importance')
    plt.xticks(rotation=90)
    plt.show()

    # Save the chart to a directory
    output_path = png_name
    plt.savefig(output_path)


    pass


# The Pipeline is executed herein:
print("************ The Pipeline Commences ***************")

Fetch_stock_data()
Preprocess_Data()
df_filtered = pd.read_csv("Filtered_data.csv")
df_filtered_final = Remove_features_based_on_correlation(df_filtered)
df_filtered_final.drop(['industry', 'sector'], axis=1, inplace=True)

# EDA: Target Data Imbalance
Check_imbalance = Check_data_imbalance(df_filtered_final)
Check_imbalance.Check_Imbalance_Targets()

'''
In machine learning, oversampling is a technique used to handle imbalanced datasets. Here's a simple explanation in layman's terms:

What is an Imbalanced Dataset?
An imbalanced dataset is one where some classes (categories or labels) are much less frequent than others. For example, if you're trying to detect fraud in financial transactions, the number of fraudulent transactions (frauds) is typically much smaller compared to the number of non-fraudulent transactions (non-frauds).

What is Oversampling?
Oversampling is a way to balance the dataset by artificially increasing the number of examples in the less frequent class. This helps the machine learning model learn to recognize the minority class better. Here’s how it works:

'''


# Step 1: Divide into Training, Testing, Validation Data:
# This step contains oversampling after spliting into three datasets
data_divide = Train_val_test_split(df_dummy=df_filtered_final)
Train_cate, Train_num, Valid_cate, Valid_num, Test_cate, Test_num = data_divide.Train_val_test_split()

# Step 2: Feature Variance:
if "Feature_Importance_Classification.csv" not in os.listdir():
    RForest_c = RandomForestClassifier(max_features=None)
    feature_pca_cate = Feature_importance(Train_cate.X_std, Train_cate.y, Test_cate.X_std.columns, RForest_c)
    df_importance_cate = feature_pca_cate.feature_importance()
    df_importance_cate.reset_index(inplace=True)
    df_importance_cate.to_csv("Feature_Importance_Classification.csv", index=False)
else:
    df_importance_cate = pd.read_csv("Feature_Importance_Classification.csv")
    Barplot_feature_importane(df_importance_cate, "Feature_Importance_Classification.png")

if "Feature_Importance_Regression.csv" not in os.listdir():
    RForest_num = RandomForestRegressor(max_features=None)
    feature_pca_num = Feature_importance(Train_num.X_std, Train_num.y, Test_cate.X_std.columns, RForest_num)
    df_importance_num = feature_pca_num.feature_importance()
    df_importance_num.reset_index(inplace=True)
    df_importance_num.to_csv("Feature_Importance_Regression.csv", index=False)
else:
    df_importance_num = pd.read_csv("Feature_Importance_Regression.csv")
    Barplot_feature_importane(df_importance_num, "Feature_Importance_Numerical.png")


# Step 3: Classification Problems

# Logistic Regression
model1 = LogisticRegression(multi_class='ovr', solver='liblinear')
LRC = Classification_Model(model1, Train_cate.X_std, Valid_cate.X_std, Train_cate.y, Valid_cate.y)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
accuracy1, bund1, y_valid_ref1 = LRC.Classification()

# Decision Tree
model2 = DecisionTreeClassifier()
DCT = Classification_Model(model2, Train_cate.X_std, Valid_cate.X_std, Train_cate.y, Valid_cate.y)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
accuracy2, bund2, y_valid_ref2  = DCT.Classification()

# = RandomForestClassifier()
model3 = RandomForestClassifier()
RFC = Classification_Model(model3, Train_cate.X_std, Valid_cate.X_std, Train_cate.y, Valid_cate.y)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
accuracy3, bund3, y_valid_ref3  = RFC.Classification()

# KNeighborsClassifier
model5 = KNeighborsClassifier(n_neighbors=3)
KNN = Classification_Model(model5, Train_cate.X_std, Valid_cate.X_std, Train_cate.y, Valid_cate.y)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
accuracy5, bund5, y_valid_ref5  = RFC.Classification()



    # Step 3: Classification Problems
if 'best_model_params.txt' not in os.listdir():
    # Grid Search RandomForestClassifier()
    grid_RFC = {
        'n_estimators': [100, 120, 130, 160],
        'criterion': ['gini', 'entropy', 'log_loss'],
    }

    Tuning_RFC = Hyperparameter_tuning(model3, grid_RFC)
    RFC_tuned = Tuning_RFC.grid_search()

    # Use the best model for prediction
    RFC_tuned_classifier = Classification_Model(RFC_tuned, Train_cate.X_std, Valid_cate.X_std, Train_cate.y, Valid_cate.y)
    accuracy_rfc_final, bund_rfc_final, y_valid_ref_final = RFC_tuned_classifier.Classification()


    # Get the best estimator
    best_model = RFC_tuned.best_estimator_

    # Save the best model parameters
    best_model_params = RFC_tuned.best_params_

    # Save the best model to a file
    joblib.dump(best_model, 'best_model.pkl')

    # Save the best model parameters to a file (optional)
    with open('best_model_params.txt', 'w') as f:
        f.write(str(best_model_params))

else:
    # Load the best model from the file
    loaded_model = joblib.load('best_model.pkl')

    # Optionally, load the best model parameters from the text file
    with open('best_model_params.txt', 'r') as f:
        loaded_model_params = eval(f.read())

    # Use the best model for prediction

    RFC_tuned_classifier = Classification_Model(loaded_model, Train_cate.X_std, Valid_cate.X_std, Train_cate.y, Valid_cate.y)
    accuracy_rfc_final, bund_rfc_final, y_valid_ref_final = RFC_tuned_classifier.Classification()

    # With important features only
    #list_cols_imp_cate = df_importance_cate.sort_values(by='0', ascending=False).iloc[:10,:]['index'].tolist()
    #index_cols = df_importance_cate.sort_values(by='0', ascending=False).iloc[:10,:].index.tolist()
    #RFC_tuned_classifier_partial = Classification_Model(loaded_model, Train_cate.X_std[:,index_cols].copy(), Valid_cate.X_std[:,index_cols].copy(), Train_cate.y, Valid_cate.y)
    #accuracy_rfc_final_partial, bund_rfc_final_partial, y_valid_ref_final_partial = RFC_tuned_classifier_partial.Classification()

# Regression Models
# Logistic Regression
model_LR1 = LinearRegression()
LR = Regression_models(model_LR1, Train_num.X_std, Valid_num.X_std, Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
df_ref_LR1, MSE_LR1 = LR.regression_prediction()

# With important features only
list_cols_imp_num = df_importance_num.sort_values(by='0', ascending=False).iloc[:10,:]['index'].tolist()
index_cols_num = df_importance_num.sort_values(by='0', ascending=False).iloc[:10,:].index.tolist()
LR_a = Regression_models(model_LR1, Train_num.X_std[:,index_cols_num].copy(), Valid_num.X_std[:,index_cols_num].copy(), Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
df_ref_LRa, MSE_LRa = LR_a.regression_prediction()

'''
#
model12 = BayesianRidge()
BR = Regression_models(model12, Train_num.X_std, Valid_num.X_std, Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
BR.regression_prediction()

BR_a = Regression_models(model12, Train_num.X_std[:,index_cols_num], Valid_num.X_std[:,index_cols_num], Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
BR_a.regression_prediction()


# KNeighbors
model_KR = KNeighborsRegressor()
KR = Regression_models(model_KR, Train_num.X_std, Valid_num.X_std, Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
KR.regression_prediction()

KR_a = Regression_models(model_KR, Train_num.X_std[:,index_cols_num], Valid_num.X_std[:,index_cols_num], Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
KR_a.regression_prediction()


# Ramdom Forest Regressor
model_RFR = RandomForestRegressor()
RFR = Regression_models(model_RFR, Train_num.X_std, Valid_num.X_std, Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
RFR.regression_prediction()

RFR = Regression_models(model_RFR, Train_num.X_std[:,index_cols_num], Valid_num.X_std[:,index_cols_num], Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
RFR.regression_prediction()

# SGD Regressor
model_SGD =SGDRegressor()
SGD = Regression_models(model_SGD, Train_num.X_std, Valid_num.X_std, Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
SGD.regression_prediction()
'''
# Gridsearch for regression models
#model_LSO = Lasso(alpha = 0.4)
#LSO = Regression_models(model_LSO, Train_num.X_std, Valid_num.X_std, Train_num.y, Valid_num.y, 3.5)
# Valid_cate.y.reset_index().groupby('dps_change_next_year').count()
#df_ref_LSO, MSE_LSO = LSO.regression_prediction()

# Gridsearch for regression models
# TO BE COMPLETE

# Prediction
# Classification
scaler_test = Test_cate.scaler
X_std_test = scaler_test.transform(Test_cate.X_std)
X_test_indexed = pd.DataFrame(X_std_test, index=Test_cate.X_std.index)
y_test = loaded_model.predict(X_test_indexed)
accuracy_final = accuracy_score(Test_cate.y['dps_change_next_year'], y_test)
#roc_auc_score(Test_cate.y['dps_change_next_year'], loaded_model.predict_proba(X_test_indexed))
pass