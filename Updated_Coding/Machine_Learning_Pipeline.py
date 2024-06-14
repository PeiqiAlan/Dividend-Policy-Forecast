import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score, auc
from sklearn.utils import shuffle



class Data_bund():
    def __init__(self, X, y, scaler_fitted):
        self.X_std = X
        self.y = y
        self.scaler = scaler_fitted



class Train_val_test_split():
    def __init__(self, df_dummy, val_portion=0.25, last_year=2023):
        self.df1 = df_dummy
        self.val_portion = val_portion
        self.last_year = last_year
        self.scaler1 = StandardScaler() # For clustering
        self.scaler2 = StandardScaler() # For numerical

    def Oversampling(self):
        pass

    def Train_val_test_split(self):
        df_train = self.df1.loc[self.df1['Year'] < (self.last_year-2)].copy()
        df_valid = self.df1.loc[self.df1['Year'].isin([self.last_year-2, self.last_year-1])].copy()
        df_test = self.df1.loc[self.df1['Year'] == self.last_year].copy()

        # Split between X and y targets
        # Train data:
        #X_train = df_train.drop(columns=['dps_change_next_year', 'dps_growth_absolute', 'dps_growth_rate', 'next_year_dividend']).copy()
        X_train = df_train.drop(columns=['dps_change_next_year', 'dps_growth_rate', 'dps_growth_absolute', 'next_year_dividend']).copy()
        X_train.drop(labels='Year', axis=1, inplace=True)
        X_train.set_index('Company', inplace=True)
        y_category_train = df_train[['Company', 'dps_change_next_year']].set_index('Company').copy()
        y_numerical_train = df_train[['Company', 'dps_growth_absolute']].set_index('Company').copy()

        # Validation Data:
        #X_valid = df_valid.drop(columns=['dps_change_next_year', 'dps_growth_absolute', 'dps_growth_rate', 'next_year_dividend']).copy()
        X_valid = df_valid.drop(columns=['dps_change_next_year', 'dps_growth_rate', 'dps_growth_absolute', 'next_year_dividend']).copy()
        X_valid.drop(labels='Year', axis=1, inplace=True)
        X_valid.set_index('Company', inplace=True)
        y_category_valid = df_valid[['Company', 'dps_change_next_year']].set_index('Company').copy()
        y_numerical_valid = df_valid[['Company', 'dps_growth_absolute']].set_index('Company').copy()

        # Testing Data:
        X_test = df_test.drop(columns=['dps_change_next_year', 'dps_growth_rate', 'dps_growth_absolute', 'next_year_dividend']).copy()
        X_test.drop(labels='Year', axis=1, inplace=True)
        X_test.set_index('Company', inplace=True)
        y_category_test = df_test[['Company', 'dps_change_next_year']].set_index('Company').copy()
        y_numerical_test = df_test[['Company', 'dps_growth_absolute']].set_index('Company').copy()

        # Oversampling training data:
        # y_category_train.reset_index().groupby('dps_change_next_year').count()
        #sampling_strategy_dict = {'increased': 2998, "constant": 1668, "decreased":1100}
        sampling_strategy_dict = {'increased': 3998, "constant": 2168, "decreased": 1200}
        smote = SMOTE(random_state=12, sampling_strategy=sampling_strategy_dict)
        X_train_oversample, y_category_train_oversample = smote.fit_resample(X_train, y_category_train)
        # The newly synthesized data is appened to the last of the dataframe.
        # First with constant adding to the same number as increased
        # Then with decreased adding to the same number of increased

        # Standardize
        # For categorical data:
        scaler1 = self.scaler1
        X_train_std = scaler1.fit_transform(X_train_oversample, y_category_train_oversample)
        X_valid_std = scaler1.transform(X_valid)

        # For numerical data:
        scaler2 = self.scaler2
        X_train_std2 = scaler2.fit_transform(X_train, y_numerical_train)
        X_valid_std2 = scaler2.transform(X_valid)

        #Keep testing data original until client calls for it
        # Shuffle the data
        X_shuffled, y_shuffled = shuffle(X_train_std, y_category_train_oversample, random_state=42)

        # Return train, valid, test data in bundles
        Train_category = Data_bund(X_shuffled, y_shuffled, scaler1)
        Train_numerical = Data_bund(X_train_std2, y_numerical_train, scaler2)

        Valid_catetory = Data_bund(X_valid_std, y_category_valid, scaler1)
        Valid_numerical = Data_bund(X_valid_std2, y_numerical_valid, scaler2)

        # Note that Testing data hasen't been standardized yet. Keeping it as a privilege for clients
        Test_category = Data_bund(X_test, y_category_test, scaler1)
        Test_numerical = Data_bund(X_test, y_numerical_test, scaler2)

        return Train_category, Train_numerical, Valid_catetory, Valid_numerical, Test_category, Test_numerical

class Feature_importance():

    def __init__(self, X, y, column_index, model_dummy):
        self.X = X
        self.y = y.values.ravel()
        self.columns = column_index
        self.model = model_dummy

    def feature_importance(self):
        Model = self.model  # We want all features to be considered for each tree
        Model.fit(self.X, self.y) #Takes one minute
        df_feature_importance = pd.DataFrame(Model.feature_importances_.T, index=self.columns)

        return df_feature_importance


class Confusion_Matrix_Bundle():
    def __init__(self, conf_increase, conf_constant, conf_decrease):
        self.increase = conf_increase
        self.constant = conf_constant
        self.decrease = conf_decrease

class Classification_Model():

    def __init__(self, model_object, X_train, X_valid, y_train, y_valid):
        #self.model = LogisticRegression(multi_class='ovr', solver='liblinear')
        self.model = model_object
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train.values.ravel()
        self.y_valid = y_valid.values.ravel()

    def individual_confusion_matrix(self, class_specify, predict, actual):
        # Convert specified class to 1 and all other classes to 0
        predicted_binary = predict == class_specify
        actual_binary = actual == class_specify
        #actual_binary = actual
        #predicted_binary = predict
        df_ref = pd.DataFrame(data=[actual, actual_binary, predict, predicted_binary]).T
        df_ref.columns = ['Actual', 'Actual_Binary', 'Predict', 'Predict_Binary']
        df_ref2 = df_ref[(df_ref['Actual_Binary'] == True) & (df_ref['Predict_Binary'] == True)]

        df_FN = df_ref[(df_ref['Actual_Binary'] == True) & (df_ref['Predict_Binary'] == False)]
        df_TN = df_ref[(df_ref['Actual_Binary'] == False) & (df_ref['Predict_Binary'] == False)]
        df_FP = df_ref[(df_ref['Actual_Binary'] == False) & (df_ref['Predict_Binary'] == True)]
        df_TP = df_ref[(df_ref['Actual_Binary'] == True) & (df_ref['Predict_Binary'] == True)]
        unique_values, counts = np.unique(self.y_valid, return_counts=True)

        # Calculate accuracy score for the specified class
        conf_matrix = confusion_matrix(actual_binary, predicted_binary, labels=[True, False])
        confusion_df = pd.DataFrame(conf_matrix.T, index=['Positive_Predict', 'Negative_Predict'], columns=['Positive_Actual', 'Negative_Actual'])

        return confusion_df

    def Classification(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred_valid = self.model.predict(self.X_valid)
        y_pred_valid_prob = self.model.predict_proba(self.X_valid)
        Overall_accuracy = accuracy_score(self.y_valid, y_pred_valid)

        confusion_df_increase = self.individual_confusion_matrix('increased', y_pred_valid, self.y_valid)
        confusion_df_constant = self.individual_confusion_matrix('constant', y_pred_valid, self.y_valid)
        confusion_df_decrease = self.individual_confusion_matrix('decreased', y_pred_valid, self.y_valid)

        Result_Bund = Confusion_Matrix_Bundle(confusion_df_increase, confusion_df_constant, confusion_df_decrease)

        return Overall_accuracy, Result_Bund, self.y_valid, y_pred_valid, y_pred_valid_prob

class Regression_models():
    def __init__(self, model_object, X_train, X_valid, y_train, y_valid, temp_replace):
        #self.model = LogisticRegression(multi_class='ovr', solver='liblinear')
        self.model = model_object
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train.values.ravel() * temp_replace
        self.y_valid = y_valid.values.ravel() * temp_replace
        self.temp_replace = temp_replace

    def regression_prediction(self):
        # Replace all occurrences of 999 with 3.5
        self.y_train[self.y_train == 999] = self.temp_replace
        self.y_valid[self.y_valid == 999] = self.temp_replace

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred_valid = self.model.predict(self.X_valid)
        MSE = mean_squared_error(self.y_valid, y_pred_valid)
        print("MSE ",mean_squared_error(self.y_valid, y_pred_valid))
        #print("AUC ",auc(self.y_valid, y_pred_valid))
        df_ref = pd.DataFrame(data=[y_pred_valid, self.y_valid]).T

        return df_ref, MSE



class Hyperparameter_tuning():
    def __init__(self, model_object, search_grid):
        self.model = model_object
        self.search_grid = search_grid

    def grid_search(self):
        grid_search_result = GridSearchCV(self.model, self.search_grid, cv=5)
        return grid_search_result









