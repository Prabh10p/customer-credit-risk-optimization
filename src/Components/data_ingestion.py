import pandas as pd
import numpy as np
import os
import sys
from src.Exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from  dataclasses import dataclass
from src.Components.data_transformation import DataTransformation,DataTrasConfig
from src.Components.model_trainer import ModelTrainer,ModelTrainerConfig
@dataclass
class DataConfig:
    raw_data_path = os.path.join("Artifacts",'data.csv')
    train_data_path=os.path.join("Artifacts","train.csv")
    test_data_path = os.path.join("Artifacts","test.csv")



class DataIngestion:
      def __init__(self):
           self.data_config = DataConfig()

      def initiate_ingestion(self):
        try:
            logging.info("Ingestion of Data Started")
            logging.info("Dataset Loading started")
            df = pd.read_csv('src/notebook/data/credit_risk_dataset.csv')
            logging.info("Dataset is Loaded succesfully")


            logging.info("Train test split started")
            train,test = train_test_split(df,test_size=0.3,random_state=42)
            logging.info("Train Test split completed")


            os.makedirs("Artifacts",exist_ok=True)


            logging.info("Saving dataset now")
            df.to_csv(self.data_config.raw_data_path,index=False,header=True)
            train.to_csv(self.data_config.train_data_path,index=False,header=True)
            test.to_csv(self.data_config.test_data_path,index=False,header=True)
            logging.info("Dataset saved in Artifacts foldder")

            return (self.data_config.train_data_path,self.data_config.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)
        



if __name__ == "__main__":
  obj = DataIngestion()
  train_data,test_data = obj.initiate_ingestion()
          

class1 = DataTransformation()
train_array,test_array,label__encoder_path,preprocessor_path = class1.initiate_transformation()


class2 = ModelTrainer()
best_model, evaluation_report, model_path = class2.initiate_training(train_array, test_array)