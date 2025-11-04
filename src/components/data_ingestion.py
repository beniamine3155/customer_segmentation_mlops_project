import os
import sys

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

        Outputs: data is returned as artifact of data ingestion component
        On failure: Raise Exception
        """
        try:
            logging.info("Exporting data from MongoDB to pandas DataFrame")
            customer_data = CustomerData()
            df = customer_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
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
        
            

            

        