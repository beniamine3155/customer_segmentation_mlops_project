import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, load_object, write_yaml_file
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (DataTransformationArtifact, 
                                         ModelTrainerArtifact, 
                                         ModelEvaluationArtifact)


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        :param model_evaluation_config: ModelEvaluationConfig
            Configuration for model evaluation.
        :param data_transformation_artifact: DataTransformationArtifact
            Artifact from data transformation containing transformed data paths.
        :param model_trainer_artifact: ModelTrainerArtifact
            Artifact from model training containing trained model path.
        """
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def load_data(self) -> pd.DataFrame:
        """
        Load the transformed data for evaluation.
        
        :return: DataFrame
            The transformed data.
        """
        try:
            logging.info("Loading transformed data for evaluation")
            transformed_data_path = self.data_transformation_artifact.transformed_data_path
            df = pd.read_csv(transformed_data_path)
            logging.info(f"Data loaded with shape: {df.shape}")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for evaluation.
        
        :param df: DataFrame
            The transformed data.
        :return: np.ndarray
            The features ready for evaluation.
        """
        try:
            logging.info("Preparing features for evaluation")
            feature_columns = self.data_transformation_artifact.feature_columns
            X = df[feature_columns].values
            logging.info(f"Features prepared with shape: {X.shape}")
            return X
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self, model, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering model using multiple metrics.
        
        :param model: Clustering model
            The fitted clustering model.
        :param X: np.ndarray
            The input features.
        :return: Dict[str, float]
            Dictionary containing evaluation metrics.
        """
        try:
            if hasattr(model, 'labels_'):
                labels = model.labels_
            else:
                labels = model.fit_predict(X)
            
            # Check if we have valid clusters
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                logging.warning("Not enough clusters formed")
                return {
                    'silhouette_score': -1.0,
                    'calinski_harabasz_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'n_clusters': len(unique_labels)
                }
            
            metrics = {}
            
            # Calculate metrics
            try:
                metrics['silhouette_score'] = silhouette_score(X, labels)
            except Exception:
                metrics['silhouette_score'] = -1.0
            
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            except Exception:
                metrics['calinski_harabasz_score'] = 0.0
            
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            except Exception:
                metrics['davies_bouldin_score'] = float('inf')
            
            metrics['n_clusters'] = len(unique_labels)
            
            return metrics
            
        except Exception as e:
            raise MyException(e, sys)

    def get_best_model_path(self) -> str:
        """
        Get the path to the best model (currently returns None as no previous model exists).
        In production, this would check for previously saved models.
        
        :return: str or None
            Path to the best model or None if no previous model exists.
        """
        try:
            # In a real scenario, you would check for previously saved models
            # For now, we'll return None as this is the first model
            return None
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates the model evaluation process.
        
        :return: ModelEvaluationArtifact
            The artifact containing the results of model evaluation.
        """
        try:
            logging.info("Starting model evaluation process")
            
            # Load data and prepare features
            df = self.load_data()
            X = self.prepare_features(df)
            
            # Load the trained model
            logging.info("Loading trained model")
            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            trained_model = load_object(trained_model_path)
            
            # Evaluate the trained model
            logging.info("Evaluating trained model")
            train_model_metrics = self.evaluate_model(trained_model, X)
            
            logging.info(f"Trained model metrics: {train_model_metrics}")
            
            # Get best model path (previous model)
            best_model_path = self.get_best_model_path()
            
            is_model_accepted = False
            improved_accuracy = 0.0
            best_model_metrics = {}
            
            if best_model_path is None:
                # No previous model exists, accept the current model
                logging.info("No previous model found. Accepting current model.")
                is_model_accepted = True
                best_model_path = trained_model_path
                best_model_metrics = train_model_metrics
                improved_accuracy = train_model_metrics['silhouette_score']
            else:
                # Load and evaluate the previous best model
                logging.info("Loading and evaluating previous best model")
                best_model = load_object(best_model_path)
                best_model_metrics = self.evaluate_model(best_model, X)
                
                # Compare models
                current_score = train_model_metrics['silhouette_score']
                best_score = best_model_metrics['silhouette_score']
                improved_accuracy = current_score - best_score
                
                logging.info(f"Current model score: {current_score}")
                logging.info(f"Best model score: {best_score}")
                logging.info(f"Improvement: {improved_accuracy}")
                
                # Check if current model is better
                if improved_accuracy > self.model_evaluation_config.changed_threshold_score:
                    logging.info("Current model is better than previous model")
                    is_model_accepted = True
                    best_model_path = trained_model_path
                    best_model_metrics = train_model_metrics
                else:
                    logging.info("Previous model is better than current model")
                    is_model_accepted = False
            
            # Create directories
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            
            # Save evaluation report
            evaluation_report = {
                'is_model_accepted': is_model_accepted,
                'improved_accuracy': float(improved_accuracy),
                'train_model_metrics': {k: float(v) for k, v in train_model_metrics.items()},
                'best_model_metrics': {k: float(v) for k, v in best_model_metrics.items()},
                'trained_model_path': trained_model_path,
                'best_model_path': best_model_path
            }
            
            write_yaml_file(self.model_evaluation_config.model_evaluation_report_file_path, 
                          evaluation_report)
            
            logging.info(f"Evaluation report saved to: {self.model_evaluation_config.model_evaluation_report_file_path}")
            
            # Create model evaluation artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=best_model_path,
                trained_model_path=trained_model_path,
                train_model_metric_artifact=train_model_metrics,
                best_model_metric_artifact=best_model_metrics
            )
            
            logging.info("Model evaluation process completed")
            logging.info(f"Model accepted: {is_model_accepted}")
            
            return model_evaluation_artifact
            
        except Exception as e:
            raise MyException(e, sys)
