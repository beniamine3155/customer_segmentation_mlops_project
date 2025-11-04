import json
import os
import sys

import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self,
                 data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        """
        :param data_validation_config: DataValidationConfig
            Configuration for data validation.
        :param data_ingestion_artifact: DataIngestionArtifact
            Artifact from data ingestion containing file paths.
        """
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_info = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)
    
    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validate if the dataframe has the expected number of columns.
        
        :param dataframe: DataFrame
            The dataframe to validate.
        :return: bool
            True if the number of columns matches the schema, else False.
        """
        try:
            # Count total columns from the list of dictionaries
            expected_num_columns = 0
            for column_dict in self.schema_info['columns']:
                expected_num_columns += len(column_dict.keys())
            
            actual_num_columns = len(dataframe.columns)
            
            # Log detailed information for debugging
            logging.info(f"Expected columns: {expected_num_columns}, Actual columns: {actual_num_columns}")
            logging.info(f"DataFrame columns: {list(dataframe.columns)}")
            
            # Check if DataFrame has at least the expected number of columns
            # This allows for extra columns (like MongoDB _id) while ensuring all required columns are present
            if actual_num_columns >= expected_num_columns:
                # Verify that all required columns exist
                required_columns = []
                for column_dict in self.schema_info['columns']:
                    required_columns.extend(column_dict.keys())
                
                missing_columns = [col for col in required_columns if col not in dataframe.columns]
                if len(missing_columns) == 0:
                    if actual_num_columns > expected_num_columns:
                        logging.info(f"DataFrame has {actual_num_columns - expected_num_columns} extra columns, but all required columns are present")
                    return True
                else:
                    logging.info(f"Missing required columns: {missing_columns}")
                    return False
            
            return False
        except Exception as e:
            raise MyException(e, sys)
    
    def is_columns_exist(self, dataframe: DataFrame) -> bool:
        """
        Check if all required columns exist in the dataframe.
        
        :param dataframe: DataFrame
            The dataframe to validate.
        :return: bool
            True if all required columns exist, else False.
        """
        try:
            required_columns = self.schema_info['columns']
            # Extract column names from the list of dictionaries
            column_names = []
            for column_dict in required_columns:
                column_names.extend(column_dict.keys())
            
            logging.info(f"Required columns from schema: {column_names}")
            logging.info(f"DataFrame columns: {list(dataframe.columns)}")
            
            missing_columns = []
            extra_columns = []
            
            # Check for missing columns
            for column_name in column_names:
                if column_name not in dataframe.columns:
                    missing_columns.append(column_name)
            
            # Check for extra columns (not in schema)
            for column_name in dataframe.columns:
                if column_name not in column_names:
                    extra_columns.append(column_name)
            
            if missing_columns:
                logging.info(f"Missing columns: {missing_columns}")
            if extra_columns:
                logging.info(f"Extra columns not in schema: {extra_columns}")
                
            return len(missing_columns) == 0
        except Exception as e:
            raise MyException(e, sys)
        
    
    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process.
        
        :return: DataValidationArtifact
            The artifact containing the results of data validation.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation process")
            file_path = self.data_ingestion_artifact.feature_store_file_path
            dataframe = self.read_data(file_path)
            
            num_columns_valid = self.validate_number_of_columns(dataframe)
            columns_exist = self.is_columns_exist(dataframe)
            
            if not num_columns_valid:
                logging.info("Number of columns validation failed")
                validation_error_msg += "Number of columns validation failed. "
            if not columns_exist:
                logging.info("Column existence validation failed")
                validation_error_msg += "Column existence validation failed. "

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )
            # Ensure the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, 'w') as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation process completed")
            return data_validation_artifact
        
        except Exception as e:
            raise MyException(e, sys)
