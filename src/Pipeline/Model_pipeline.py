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
        self.model_path = os.path.join("Artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")
        self.mapping_path = os.path.join("Artifacts", "label_mapping.pkl")

    def MakePipeline(self, input_data: pd.DataFrame):
        try:
            logging.info("Loading preprocessor")
            preprocessor = load_object(self.preprocessor_path)

            logging.info("Loading model")
            model = load_object(self.model_path)

            logging.info("Loading label mapping")
            label_mapping = load_object(self.mapping_path)

            logging.info("Transforming input data")
            prep_data = preprocessor.transform(input_data)

            logging.info("Making prediction")
            y_pred = model.predict(prep_data)
            probabilities = model.predict_proba(prep_data)[0]  # [prob_not_default, prob_default]

            logging.info("Mapping the predicted label")
            y_pred_label = [label_mapping.get(int(pred), pred) for pred in y_pred]

            return y_pred_label, probabilities

        except Exception as e:
            raise CustomException(e, sys)



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
