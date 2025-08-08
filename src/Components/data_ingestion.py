import pandas as pd
import numpy as np
import os
import sys
from src.Exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.Components.data_transformation import DataTransformation, DataTrasConfig
from src.Components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataConfig:
    raw_data_path = os.path.join("Artifacts", 'data.csv')
    train_data_path = os.path.join("Artifacts", "train.csv")
    test_data_path = os.path.join("Artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.data_config = DataConfig()

    def initiate_ingestion(self):
        try:
            logging.info("📥 Ingestion of Data Started")
            logging.info("📄 Loading dataset from file")

            df = pd.read_csv('src/notebook/data/credit_risk_dataset.csv')
            logging.info(f"✅ Dataset loaded successfully with shape: {df.shape}")

            # ✅ Use cb_person_default_on_file as target column
            target_col = "cb_person_default_on_file"

            # ✅ Remove rows with missing target or invalid classes
            logging.info("🔍 Cleaning and filtering data for binary classification based on 'cb_person_default_on_file'")
            df = df[df[target_col].isin(["Y", "N"])]
            df = df.dropna(subset=[target_col])
            logging.info(f"✅ Cleaned dataset shape: {df.shape}")
            logging.info(f"🔢 Class distribution:\n{df[target_col].value_counts()}")

            # ✅ Train-test split
            logging.info("✂️ Performing stratified train-test split")
            train_df, test_df = train_test_split(
                df,
                test_size=0.3,
                random_state=42,
                stratify=df[target_col]
            )
            logging.info(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # ✅ Save files
            os.makedirs("Artifacts", exist_ok=True)
            df.to_csv(self.data_config.raw_data_path, index=False)
            train_df.to_csv(self.data_config.train_data_path, index=False)
            test_df.to_csv(self.data_config.test_data_path, index=False)
            logging.info("💾 Datasets saved to 'Artifacts/' folder")

            return (self.data_config.train_data_path, self.data_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

# 🔁 Pipeline Runner
if __name__ == "__main__":
    # Step 1: Ingest
    obj = DataIngestion()
    train_data, test_data = obj.initiate_ingestion()

    # Step 2: Transform
    transformer = DataTransformation()
    train_array, test_array, label_encoder_path, preprocessor_path = transformer.initiate_transformation()

    # Step 3: Train
    trainer = ModelTrainer()
    best_model, evaluation_report, model_path = trainer.initiate_training(train_array, test_array)
