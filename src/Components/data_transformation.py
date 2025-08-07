from dataclasses import dataclass
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_object
from src.Exception import CustomException
from src.logger import logging

@dataclass
class DataTrasConfig:
    preprocesor_path = os.path.join("Artifacts", "preprocessor.pkl")
    label_encoder_path = os.path.join("Artifacts", "label_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.trans_path = DataTrasConfig()

    def initiate_transformation(self):
        try:
            logging.info("Transformation of data started")
            train = pd.read_csv("Artifacts/train.csv")
            test = pd.read_csv("Artifacts/test.csv")

            X_train = train.iloc[:, :-1]
            y_train = train.iloc[:, -1]

            X_test = test.iloc[:, :-1]
            y_test = test.iloc[:, -1]

            cat_features = X_train.select_dtypes(include=["object"]).columns.to_list()
            ordinal_features = ["loan_grade"]
            num_features = X_train.select_dtypes(exclude=["object"]).columns.to_list()

            cat_features = [col for col in cat_features if col not in ordinal_features]

            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            save_object(self.trans_path.label_encoder_path, le)

            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), cat_features),
                    ("ordinal", OrdinalEncoder(categories=[['A', 'B', 'C', 'D', 'E', 'F', 'G']]), ordinal_features),
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler())
                    ]), num_features)
                ]
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor)
            ])

            save_object(self.trans_path.preprocesor_path, pipeline)
            logging.info("Saved preprocessing pipeline.")

            X_train_processed = pipeline.fit_transform(X_train)
            X_test_processed = pipeline.transform(X_test)

            train_array = np.array(X_train_processed)
            test_array = np.array(X_test_processed)

            logging.info("Data preprocessing completed successfully.")

            return train_array, test_array, self.trans_path.label_encoder_path,self.trans_path.preprocesor_path

        except Exception as e:
            raise CustomException(e, sys)
