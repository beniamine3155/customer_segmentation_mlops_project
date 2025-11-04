import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig)

from src.entity.artifact_entity import (DataIngestionArtifact,
                                         DataValidationArtifact,
                                         DataTransformationArtifact,
                                         ModelTrainerArtifact,
                                         ModelEvaluationArtifact)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method starts the data ingestion process and returns the artifact.
        """
        try:
            logging.info("Starting data ingestion process")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion process completed")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)


    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method starts the data validation process and returns the artifact.
        """
        try:
            logging.info("Starting data validation process")
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation process completed")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)
        

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method starts the data transformation process and returns the artifact.
        """
        try:
            logging.info("Starting data transformation process")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation process completed")
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method starts the model training process and returns the artifact.
        """
        try:
            logging.info("Starting model training process")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training process completed")
            return model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_model_evaluation(self, data_transformation_artifact: DataTransformationArtifact, 
                              model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method starts the model evaluation process and returns the artifact.
        """
        try:
            logging.info("Starting model evaluation process")
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Model evaluation process completed")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)
        
        

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
        except Exception as e:
            raise MyException(e, sys)