import os
import sys
import pandas as pd

from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.proj1_data import CustomerData


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: DataIngestionConfig
            Configuration for data ingestion process.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)
        
    def export_data_from_mongo(self) -> DataFrame:
        """
        Method Name: export_data_from_mongo
        Description: Export the entire collection as a pandas DataFrame.
        If MongoDB is not available, fallback to local CSV data.

        Outputs: data is returned as artifact of data ingestion component
        On failure: Raise Exception
        """
        try:
            logging.info("Attempting to export data from MongoDB to pandas DataFrame")
            
            # Try MongoDB first
            try:
                customer_data = CustomerData()
                df = customer_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
                logging.info("Successfully exported data from MongoDB")
                
                # Remove MongoDB's _id column if it exists
                if '_id' in df.columns:
                    logging.info("Removing MongoDB '_id' column from DataFrame")
                    df = df.drop(columns=['_id'])
                    
            except Exception as mongo_error:
                logging.warning(f"MongoDB connection failed: {mongo_error}")
                logging.info("Falling back to local CSV data")
                
                # Fallback to local CSV data
                local_data_path = "notebook/customer_segmentation.csv"
                if os.path.exists(local_data_path):
                    logging.info(f"Loading data from local file: {local_data_path}")
                    df = pd.read_csv(local_data_path)
                    logging.info(f"Successfully loaded data from local file with shape: {df.shape}")
                else:
                    raise Exception(f"Neither MongoDB nor local data file ({local_data_path}) is available")
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Saving data to feature store at: {feature_store_file_path}")
            df.to_csv(feature_store_file_path, index=False, header=True)
            return df
        except Exception as e:
            raise MyException(e, sys)
        
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: Initiates the data ingestion component of training pipeline.

        Outputs: returns as a artifact of data ingestion component
        On failure: Raise Exception
        """
        try:
            logging.info("Starting data ingestion")
            df = self.export_data_from_mongo()
            logging.info("Data ingestion completed successfully")
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            )
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)
        
            

            

        