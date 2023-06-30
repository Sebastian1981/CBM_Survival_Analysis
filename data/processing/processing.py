import numpy as np
import pandas as pd

class ChurnDataset:
    def __init__(self):
        self.file_name = 'data/telco_customer_churn.csv'
        self.churn_df = None
        self.duration = ['tenure']
        self.event = ['Churn']
        self.cat_features = [
            'gender',
            'Partner',
            'Dependents',
            'PhoneService',
            'MultipleLines', 
            'InternetService', 
            'OnlineSecurity', 
            'OnlineBackup', 
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod'
            ]
        self.numeric_features = [
            'TotalCharges', 
            'MonthlyCharges',
            'SeniorCitizen'
            ]

    def load_data(self):
        self.churn_df = pd.read_csv(
            self.file_name, 
            usecols = self.numeric_features + self.cat_features + self.duration + self.event)
    
    def prep_total_charges(self):
        self.churn_df['TotalCharges'] = self.churn_df['TotalCharges'].replace(" ", np.nan).astype('float64')
        mean_val = self.churn_df['TotalCharges'].mean()
        self.churn_df['TotalCharges'] = self.churn_df['TotalCharges'].fillna(mean_val)

    def add_tenure_epsilon(self, epsilon=0.01):
        # add epsilon to tenure b/c some models dont like negative values!
        self.churn_df[self.duration] = self.churn_df[self.duration] + epsilon

    def encode_numeric_features(self):
        self.churn_df[self.numeric_features] = self.churn_df[self.numeric_features].astype('int')

    def encode_categorical_features(self):
        self.churn_df = pd.concat([
            self.churn_df, 
            pd.get_dummies(self.churn_df[self.cat_features + self.event], 
                           drop_first=True)], 
                   axis=1).drop(self.cat_features + self.event, axis=1)
    
    def get_data(self):
        return self.churn_df