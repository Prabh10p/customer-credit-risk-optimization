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
            # Assuming the label_mapping is like: {0: "NO DEFAULT", 1: "DEFAULT"}
            y_pred = [label_mapping.get(int(pred), pred) for pred in y_pred]

            logging.info("Mapping the predicted label Done")

            return y_pred

        except Exception as e:
            raise


class ModelFeatures:
    def __init__(
        self,
        person_age,
        person_income,
        person_home_ownership,       # ✅ Corrected
        person_emp_length,           # ✅ Corrected
        loan_intent,
        loan_grade,
        loan_amnt,                   # ✅ Corrected
        loan_int_rate,
        loan_status,
        loan_percent_income,
        cb_person_cred_hist_length   # ✅ Corrected
    ):
        self.person_age = person_age
        self.person_income = person_income
        self.person_home_ownership = person_home_ownership
        self.person_emp_length = person_emp_length
        self.loan_intent = loan_intent
        self.loan_grade = loan_grade
        self.loan_amnt = loan_amnt
        self.loan_int_rate = loan_int_rate
        self.loan_status = loan_status
        self.loan_percent_income = loan_percent_income
        self.cb_person_cred_hist_length = cb_person_cred_hist_length

    def to_dataframe(self):
        data = {
            'person_age': [self.person_age],
            'person_income': [self.person_income],
            'person_home_ownership': [self.person_home_ownership],
            'person_emp_length': [self.person_emp_length],
            'loan_intent': [self.loan_intent],
            'loan_grade': [self.loan_grade],
            'loan_amnt': [self.loan_amnt],
            'loan_int_rate': [self.loan_int_rate],
            'loan_status': [self.loan_status],
            'loan_percent_income': [self.loan_percent_income],
            'cb_person_cred_hist_length': [self.cb_person_cred_hist_length]
        }
        return pd.DataFrame(data)
