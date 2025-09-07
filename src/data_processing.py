import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import os
import sys

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path
        
        os.makedirs(self.processed_path, exist_ok=True)
        logger.info("Data Processing Initialized")

    def load_data(self):
        try:
            df = pd.read_csv(self.raw_path)
            logger.info("Data Loaded..")
            return df
        except Exception as e:
            logger.error(f"Error While reading Data: {e}")
            raise CustomException("Failed to read_data.", sys)

    def handle_outliers(self, df:pd.DataFrame, column:str):
        try:
            logger.info("Handling Outliers...")
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3-Q1

            Lower_value = Q1 - 1.5*IQR
            Upper_value = Q3 + 1.5*IQR

            sepal_median = np.median(df[column])

            df[column] = df[column].apply(lambda x: sepal_median if x>Upper_value or x<Lower_value else x)

            logger.info(f"Handle Outliers in {column} with Upper Bound: {Upper_value} and Lower Bound: {Lower_value}")
            
            return df
        except Exception as e:
            logger.error(f"Error While Handling Outliers: {e}")
            raise CustomException("Failed to Handle Outliers.", sys)
        
    def split_data(self, df):
        try:
            X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
            y = df["Species"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info(f"Data Splitted Successfully.")

            joblib.dump(X_train, os.path.join(self.processed_path, "X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.processed_path, "X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.processed_path, "y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.processed_path, "y_test.pkl"))

            logger.info(f"Split Train/Test data saved in {self.processed_path}")

        except Exception as e:
            logger.error(f"Error While Splitting and Saving data: {e}")
            raise CustomException("Failed to Split and Save data.", sys)
        
    def process_data(self):
        df = self.load_data()
        df = self.handle_outliers(df, "SepalWidthCm")
        self.split_data(df)

if __name__ == "__main__":
    raw_path = "artifacts/raw/data.csv"
    processed_path = "artifacts/processed"
    processor = DataProcessing(raw_path=raw_path, processed_path=processed_path)

    processor.process_data()