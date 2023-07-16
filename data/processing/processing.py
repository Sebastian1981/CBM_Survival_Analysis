import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ChurnData:
    def __init__(self, filename: str, testsize: float, seed: int):
        self.file_name = filename
        self.seed = seed
        self.test_size = testsize
        self.churn_df = None
        self.churn_df_uncensored = None
        self.churn_df_train = None
        self.churn_df_test = None
        self.churn_df_uncensored_train = None
        self.churn_df_uncensored_test = None
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
            usecols=self.numeric_features + self.cat_features + self.duration + self.event)

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
                           drop_first=True,
                           dtype='int')],
            axis=1).drop(self.cat_features + self.event, axis=1)

    def get_uncensored_data(self):
        self.churn_df_uncensored = self.churn_df[self.churn_df['Churn_Yes'] == 1].drop(labels=['Churn_Yes'], axis=1)

    def process_data(self):
        # do the standard processing
        self.prep_total_charges()
        self.encode_numeric_features()
        self.encode_categorical_features()
        self.add_tenure_epsilon(epsilon=0.001)
        # train test split censored data
        self.churn_df_train, self.churn_df_test = train_test_split(self.churn_df,
                                                                   test_size=self.test_size,
                                                                   random_state=self.seed)
        # get uncensored data
        self.get_uncensored_data()
        # train test split uncensored data
        self.churn_df_uncensored_train, self.churn_df_uncensored_test = train_test_split(self.churn_df,
                                                                                         test_size=self.test_size,
                                                                                         random_state=self.seed)
