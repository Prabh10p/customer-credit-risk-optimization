import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys


# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category= FutureWarning)



from dataclasses import dataclass
from src.Exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model





@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("Artifacts","model.pkl")

class ModelTrainer:
  try:
    def __init__(self):
        self.model_path = ModelTrainerConfig()

    def initiate_training(self,train_array,test_array):
        logging.info("Splitting of Data Started")
        X_train = train_array[:,:-1]
        X_test =  test_array[:,:-1]
        y_train = train_array[:,-1]
        y_test = test_array[:,-1]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        logging.info("Splitting of Data Completed")


        logging.info("Model training started")

        models = {
    "LogisticRegression": LogisticRegression(solver="saga", penalty="elasticnet", max_iter=10),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "CatBoostClassifier": CatBoostClassifier(verbose=0),
    "XGBClassifier": XGBClassifier(eval_metric="logloss", use_label_encoder=False)
}

        params = {
    "LogisticRegression": {
        "l1_ratio": [0.0, 0.5, 1.0],
        "C": [0.1, 1, 10]
    },
    "AdaBoostClassifier": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1]
    },
    "RandomForestClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [10, 30, 50],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "CatBoostClassifier": {
        "iterations": [100, 200],
        "depth": [4, 6, 10],
        "learning_rate": [0.01, 0.1]
    },
    "XGBClassifier": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1]
    }

}
        
        logging.info("Model training Done!")



        evaluation_report = evaluate_model(X_train, X_test, y_train, y_test, models, params)
        best_model_name = max(evaluation_report, key=lambda x: evaluation_report[x]['accuracy_score'])
        best_model = evaluation_report[best_model_name]['model']

# Save the best model
        save_object(self.model_path.trained_model_path, best_model)
        logging.info(f"Best model: {best_model_name} with accuracy: {evaluation_report[best_model_name]['accuracy_score']}")

        return best_model,evaluation_report,self.model_path.trained_model_path


        
  except Exception as e:
      raise CustomException(e,sys)

        










