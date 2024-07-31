import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml
from src.components.data_ingestion import DataIngestion
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os,sys
from sklearn.model_selection import train_test_split
import pickle


class ModelTrainer:
    
    
    
    def __init__(self):
        
        try:
        
            self.model = XGBRegressor(
                gamma=0.18789978831283782,
                learning_rate=0.18896547008552977,
                max_depth=8,
                min_child_weight=8,
                n_estimators=89
            )
        except Exception as e:
            raise CustomException(e,sys)
        
    def train(self, X_train, y_train):
        
        try:

            self.model.fit(X_train, y_train)
            pickle_filename = "artifacts/xgboost_model.pkl" #os.path.join("../artifacts/","xgboost_model.pkl")
            #os.path.join(os.path.dirname(__file__), "../artifacts/xgboost_model.pkl")
            #pickle_filename = 'xgboost_model.pkl'
            with open(pickle_filename, 'wb') as file:
                pickle.dump(self.model, file)
        except Exception as e:
            raise CustomException(e,sys)
            
        
    def predict(self, X_test):
        try:   
            return self.model.predict(X_test)
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def evaluate_model(self, y_true, y_pred, n_features):
        
        try:
            r2 = r2_score(y_true, y_pred)
            n = len(y_true)
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
            metrics = {
                'Mean Absolute Error': float(mean_absolute_error(y_true, y_pred)),
                'Mean Squared Error': float(mean_squared_error(y_true, y_pred)),
                'Root Mean Squared Error': float(mean_squared_error(y_true, y_pred, squared=False)),
                'R^2 Score': float(r2),
                'Adjusted R^2 Score': float(adjusted_r2)
            }
        except Exception as e:
            raise CustomException(e,sys)
        return metrics
    
    def evaluate_and_save(self, X_train, X_test, y_test, output_file):
        
        try:
            n_features = X_train.shape[1]
            y_pred = self.predict(X_test)
            metrics = self.evaluate_model(y_test, y_pred, n_features)
            with open(output_file, 'w') as file:
                yaml.dump(metrics, file,default_flow_style=False)
        except Exception as e:
            raise CustomException(e,sys)


    def inititate_training(self):

        try:
        
            di = DataIngestion()
            X_train,X_test, y_train, y_test = di.inititiate_data_ingestion()
            self.train(X_train, y_train)
            self.evaluate_and_save(X_train, X_test, y_test, 'model_metrics.yaml')
        except Exception as e:
            raise CustomException(e,sys)    
   


# Example usage:
# Initialize the trainer
'''
trainer = ModelTrainer(
    gamma=0.18789978831283782,
    learning_rate=0.18896547008552977,
    max_depth=8,
    min_child_weight=8,
    n_estimators=89
)
'''

# Assuming X_train, X_test, y_train, and y_test are already defined

    