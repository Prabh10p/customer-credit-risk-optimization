import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


from src.utils import load_object
from src.Exception import CustomException
from src.logger import logging


class Pipeline:
    def __init__(self):
        self.model_path = os.path.join("Artifacts","model.pkl")
        self.preprocessor_path = os.path.join("Artifacts","preprocessor.pkl")
        self.mapping_path = os.path.join("Artifacts","label_mapping.pkl")

    def MakePipeline(self,input_data:str):
        try:
            logging.info("loading preprocessor")
            preprocessor = load_object(self.preprocessor_path)
            logging.info("loading preprocessor Done")

            logging.info("loading Model")
            model = load_object(self.model_path)
            logging.info("loading Model Done")

            logging.info("loading label_mapping")
            label_mapping = load_object(self.mapping_path)
            logging.info("loading label_mapping Done")


            logging.info("Predicting User Input")
            prep_data  = preprocessor.transform(input_data)
            y_pred = model.predict(prep_data)
            logging.info("Predicting User Input Done")


            logging.info("Mapping the predicted label")
            y_pred = label_mapping.inverse_transform(y_pred)
            logging.info("Mapping the predicted label Done")

            return y_pred

        except Exception as e:
            raise


class ModelFeatures:
    def __init__(
        self,
        person_age,
        person_income,
        home_ownership_type,
        employement_duration,
        loan_intent,
        loan_grade,
        loan_amount,
        loan_int_rate,
        loan_status,
        loan_percent_income,
        credit_history_duration
    ):
        self.data = {
            "person_age": person_age,
            "person_income": person_income,
            "home_ownership_type": home_ownership_type,
            "employement_duration": employement_duration,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amount": loan_amount,
            "loan_int_rate": loan_int_rate,
            "loan_status": loan_status,
            "loan_percent_income": loan_percent_income,
            "credit_history_duration": credit_history_duration
        }

    def to_dataframe(self):
        return pd.DataFrame([self.data])




             






