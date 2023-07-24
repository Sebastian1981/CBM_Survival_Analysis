import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from lifelines.utils import concordance_index
import shap


class LifetimeRegressionModel:
    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, seed: int):
        self.seed = seed
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.target = 'tenure'
        self.features = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def get_feature_names(self):
        features = self.train_dataset.columns.to_list()
        features.remove('TotalCharges')
        features.remove('tenure')
        self.features = features

    def separate_features_and_target(self):
        self.X_train = self.train_dataset[self.features]
        self.y_train = self.train_dataset['tenure'].values
        self.X_test = self.test_dataset[self.features]
        self.y_test = self.test_dataset['tenure'].values

    def build_model(self):
        self.get_feature_names()
        self.separate_features_and_target()
        self.model = GradientBoostingRegressor(random_state=self.seed).fit(self.X_train, self.y_train)

    def evaluate_model(self):
        print('r2-score on training set: {:.2f}'.format(r2_score(self.y_train, self.model.predict(self.X_train))))
        print('r2-score on test set: {:.2f}'.format(r2_score(self.y_test, self.model.predict(self.X_test))))
        print('rmse on training set: {:.2f}'.format(np.sqrt(mean_squared_error(self.y_train, self.model.predict(self.X_train)))))
        print('rmse on test set: {:.2f}'.format(np.sqrt(mean_squared_error(self.y_test, self.model.predict(self.X_test)))))
        print('c-index on train set: {:.2f}'.format(concordance_index(event_times=self.y_train, predicted_scores=self.model.predict(self.X_train))))
        print('c-index on test set: {:.2f}'.format(concordance_index(event_times=self.y_test, predicted_scores=self.model.predict(self.X_test))))

    def plot_feature_importance(self):
        importances = self.model.feature_importances_
        forest_importances = pd.Series(importances, index=self.features)
        forest_importances = pd.DataFrame(data=forest_importances, columns=['importance']).sort_values(by='importance', ascending=True)
        fig, ax = plt.subplots()
        forest_importances.plot.barh(ax=ax)
        ax.set_title("Feature Importance using Log-Loss")
        ax.set_ylabel("Mean log-loss")
        fig.tight_layout()

    def plot_shap_feature_importance(self, plot_type: str):
        """plot_type can be 'bar' or 'violin'"""
        explainer = shap.TreeExplainer(
            model = self.model,
            data = self.X_test,
            feature_perturbation = "interventional",
            model_output='raw')
        shap_values = explainer.shap_values(self.X_test, check_additivity=True)
        shap.summary_plot(
            shap_values=shap_values,
            features=self.X_test,
            plot_type=plot_type
        )

    def plot_model_prediction(self):
        plt.scatter(x=self.y_test, y=self.model.predict(self.X_test))
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red', linestyle='--')
        plt.title('Tenure: Predicted vs Actual')
        plt.xlabel('actual tenure [month]')
        plt.ylabel('predicted tenure [month]')
        plt.grid()
        plt.show()