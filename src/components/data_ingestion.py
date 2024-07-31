import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os,sys
from sklearn.model_selection import train_test_split




class DataIngestion:

    
    
    def __init__(self):
        
        try:
            self.DATASET_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../data/Folds5x2_pp.xlsx')
        except Exception as e:
            raise CustomException(e,sys)

    def inititiate_data_ingestion(self):
      
        try:


            df_initial = pd.read_excel(self.DATASET_FILE_PATH)
            independent_vars = ['AT', 'V', 'AP', 'RH']
            dependent_var = 'PE'
            X = df_initial.drop(columns=[dependent_var])
            y = df_initial[dependent_var]
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return self.X_train , self.X_test, self.y_train, self.y_test
        except Exception as e:
            raise CustomException(e,sys)
        

