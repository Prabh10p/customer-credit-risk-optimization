from dataclasses import dataclass
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_object
from src.Exception import CustomException
from src.logger import logging

@dataclass
class DataTrasConfig:
    preprocesor_path = os.path.join("Artifacts", "preprocessor.pkl")
    label_mapping_path = os.path.join("Artifacts", "label_mapping.pkl")

class DataTransformation:
    def __init__(self):
        self.trans_path = DataTrasConfig()

    def balance_classes(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        try:
            logging.info("Balancing the dataset by upsampling the minority class.")
            majority = df[df[target_col] == 'N']
            minority = df[df[target_col] == 'Y']
            new_minority = minority.sample(len(majority), replace=True, random_state=42)
            balanced_df = pd.concat([majority, new_minority], axis=0)
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
            logging.info(f"Balanced dataset shape: {balanced_df.shape}")
            return balanced_df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformation(self):
        try:
            logging.info("Transformation of data started")

            # Load data
            raw_train = pd.read_csv("Artifacts/train.csv")
            test = pd.read_csv("Artifacts/test.csv")

            # Balance training data
            train = self.balance_classes(raw_train, target_col="cb_person_default_on_file")

            # Separate features and target
            X_train = train.drop("cb_person_default_on_file", axis=1)
            y_train = train["cb_person_default_on_file"]

            X_test = test.drop("cb_person_default_on_file", axis=1)
            y_test = test["cb_person_default_on_file"]

            # Feature categorization
            ordinal_features = ["loan_grade"]
            cat_features = X_train.select_dtypes(include=["object"]).columns.tolist()
            num_features = X_train.select_dtypes(exclude=["object"]).columns.tolist()
            cat_features = [col for col in cat_features if col not in ordinal_features]

            # Manual label mapping
            label_mapping = {'N': 0, 'Y': 1}
            y_train = y_train.map(label_mapping)
            y_test = y_test.map(label_mapping)

            # Save the label mapping
            save_object(self.trans_path.label_mapping_path, label_mapping)
            logging.info("Label mapping dictionary saved.")

            # Preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), cat_features),
                    ("ordinal", OrdinalEncoder(categories=[["A", "B", "C", "D", "E", "F", "G"]]), ordinal_features),
                    ("num", Pipeline([
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler())
                    ]), num_features)
                ]
            )

            pipeline = Pipeline([
                ("preprocessor", preprocessor)
            ])

            # Fit and transform
            pipeline.fit(X_train)
            X_train_processed = pipeline.transform(X_train)
            X_test_processed = pipeline.transform(X_test)

            # Save preprocessing pipeline
            save_object(self.trans_path.preprocesor_path, pipeline)
            logging.info("Preprocessing pipeline saved.")

            # Combine features and labels
            train_array = np.c_[X_train_processed, y_train.values]
            test_array = np.c_[X_test_processed, y_test.values]

            logging.info("Data transformation completed successfully.")

            return train_array, test_array, self.trans_path.label_mapping_path, self.trans_path.preprocesor_path

        except Exception as e:
            raise CustomException(e, sys)
