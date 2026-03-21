import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformer
@dataclass

class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info('entered the data ingesiton method')
        try:
            # Using a raw string (r'') is safer for Windows paths
            df = pd.read_csv(r'D:\PRO AIML\KAGGLE PROJECTS\HEART DISEASE\heart.csv')
            logging.info('read the dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # --- ADD THIS LINE TO CREATE DATA.CSV ---
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved to artifacts')

            logging.info('train test split initiate')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # These correctly save the split files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    try:
        print("Step 1: Starting Data Ingestion...")
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        print(f"Step 1 Complete. Files saved at: {train_data}")

        print("Step 2: Starting Data Transformation...")
        data_transformation = DataTransformer()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
        print(f"Step 2 Complete. Preprocessor saved at: {preprocessor_path}")
        
    except Exception as e:
        print(f"PROCESS FAILED: {e}")