from src.exception import CustomException
from src.logger import logging
import os,sys
from src.components.model_trainer1 import ModelTrainer
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    
    try:
        mt = ModelTrainer()
        mt.inititate_training()
    except Exception as e:
        raise CustomException(e,sys)    


