import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, load_object, read_yaml_file
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        """
        :param model_trainer_config: ModelTrainerConfig
            Configuration for model training.
        :param data_transformation_artifact: DataTransformationArtifact
            Artifact from data transformation containing transformed data paths.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_config = read_yaml_file(file_path=model_trainer_config.model_config_file_path)
        except Exception as e:
            raise MyException(e, sys)

    def load_transformed_data(self) -> pd.DataFrame:
        """
        Load the transformed data from the data transformation artifact.
        
        :return: DataFrame
            The transformed data.
        """
        try:
            logging.info("Loading transformed data")
            transformed_data_path = self.data_transformation_artifact.transformed_data_path
            df = pd.read_csv(transformed_data_path)
            logging.info(f"Transformed data loaded with shape: {df.shape}")
            return df
        except Exception as e:
            raise MyException(e, sys)

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for clustering.
        
        :param df: DataFrame
            The transformed data.
        :return: np.ndarray
            The features ready for clustering.
        """
        try:
            logging.info("Preparing features for clustering")
            feature_columns = self.data_transformation_artifact.feature_columns
            
            # Select only the scaled features for clustering
            X = df[feature_columns].values
            
            logging.info(f"Features prepared with shape: {X.shape}")
            return X
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_clustering_model(self, model, X: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Evaluate clustering model using multiple metrics.
        
        :param model: Clustering model
            The fitted clustering model.
        :param X: np.ndarray
            The input features.
        :param model_name: str
            Name of the model for logging.
        :return: Dict[str, float]
            Dictionary containing evaluation metrics.
        """
        try:
            if hasattr(model, 'labels_'):
                labels = model.labels_
            else:
                labels = model.fit_predict(X)
            
            # Check if we have valid clusters (more than 1 unique label, excluding noise points)
            unique_labels = np.unique(labels)
            if model_name == 'DBSCAN':
                # For DBSCAN, -1 indicates noise points
                valid_labels = unique_labels[unique_labels != -1]
                if len(valid_labels) < 2:
                    logging.warning(f"{model_name}: Not enough clusters formed (only {len(valid_labels)} clusters)")
                    return {
                        'silhouette_score': -1.0,
                        'calinski_harabasz_score': 0.0,
                        'davies_bouldin_score': float('inf'),
                        'n_clusters': len(valid_labels),
                        'n_noise': np.sum(labels == -1)
                    }
            else:
                if len(unique_labels) < 2:
                    logging.warning(f"{model_name}: Not enough clusters formed")
                    return {
                        'silhouette_score': -1.0,
                        'calinski_harabasz_score': 0.0,
                        'davies_bouldin_score': float('inf'),
                        'n_clusters': len(unique_labels)
                    }
            
            metrics = {}
            
            # Silhouette Score (higher is better, range: -1 to 1)
            try:
                metrics['silhouette_score'] = silhouette_score(X, labels)
            except Exception as e:
                logging.warning(f"Could not calculate silhouette score for {model_name}: {e}")
                metrics['silhouette_score'] = -1.0
            
            # Calinski-Harabasz Score (higher is better)
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
            except Exception as e:
                logging.warning(f"Could not calculate Calinski-Harabasz score for {model_name}: {e}")
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin Score (lower is better)
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            except Exception as e:
                logging.warning(f"Could not calculate Davies-Bouldin score for {model_name}: {e}")
                metrics['davies_bouldin_score'] = float('inf')
            
            metrics['n_clusters'] = len(unique_labels)
            
            if model_name == 'DBSCAN':
                metrics['n_noise'] = np.sum(labels == -1)
            
            return metrics
            
        except Exception as e:
            raise MyException(e, sys)

    def train_kmeans(self, X: np.ndarray) -> Tuple[Any, Dict[str, float], Dict]:
        """
        Train KMeans clustering model with hyperparameter tuning.
        
        :param X: np.ndarray
            The input features.
        :return: Tuple[Any, Dict[str, float], Dict]
            Best model, best metrics, and best parameters.
        """
        try:
            logging.info("Training KMeans model")
            
            kmeans_params = self.model_config['grid_search']['KMeans']
            param_grid = list(ParameterGrid(kmeans_params))
            
            best_model = None
            best_score = -1
            best_metrics = None
            best_params = None
            
            for params in param_grid:
                model = KMeans(**params)
                model.fit(X)
                
                metrics = self.evaluate_clustering_model(model, X, 'KMeans')
                
                # Use silhouette score as the primary metric for model selection
                current_score = metrics['silhouette_score']
                
                if current_score > best_score:
                    best_score = current_score
                    best_model = model
                    best_metrics = metrics
                    best_params = params
            
            logging.info(f"Best KMeans parameters: {best_params}")
            logging.info(f"Best KMeans metrics: {best_metrics}")
            
            return best_model, best_metrics, best_params
            
        except Exception as e:
            raise MyException(e, sys)

    def train_agglomerative(self, X: np.ndarray) -> Tuple[Any, Dict[str, float], Dict]:
        """
        Train Agglomerative clustering model with hyperparameter tuning.
        
        :param X: np.ndarray
            The input features.
        :return: Tuple[Any, Dict[str, float], Dict]
            Best model, best metrics, and best parameters.
        """
        try:
            logging.info("Training Agglomerative Clustering model")
            
            agg_params = self.model_config['grid_search']['AgglomerativeClustering']
            param_grid = list(ParameterGrid(agg_params))
            
            best_model = None
            best_score = -1
            best_metrics = None
            best_params = None
            
            for params in param_grid:
                model = AgglomerativeClustering(**params)
                model.fit(X)
                
                metrics = self.evaluate_clustering_model(model, X, 'AgglomerativeClustering')
                
                # Use silhouette score as the primary metric for model selection
                current_score = metrics['silhouette_score']
                
                if current_score > best_score:
                    best_score = current_score
                    best_model = model
                    best_metrics = metrics
                    best_params = params
            
            logging.info(f"Best Agglomerative parameters: {best_params}")
            logging.info(f"Best Agglomerative metrics: {best_metrics}")
            
            return best_model, best_metrics, best_params
            
        except Exception as e:
            raise MyException(e, sys)

    def train_dbscan(self, X: np.ndarray) -> Tuple[Any, Dict[str, float], Dict]:
        """
        Train DBSCAN clustering model with hyperparameter tuning.
        
        :param X: np.ndarray
            The input features.
        :return: Tuple[Any, Dict[str, float], Dict]
            Best model, best metrics, and best parameters.
        """
        try:
            logging.info("Training DBSCAN model")
            
            dbscan_params = self.model_config['grid_search']['DBSCAN']
            param_grid = list(ParameterGrid(dbscan_params))
            
            best_model = None
            best_score = -1
            best_metrics = None
            best_params = None
            
            for params in param_grid:
                model = DBSCAN(**params)
                model.fit(X)
                
                metrics = self.evaluate_clustering_model(model, X, 'DBSCAN')
                
                # Use silhouette score as the primary metric for model selection
                current_score = metrics['silhouette_score']
                
                if current_score > best_score:
                    best_score = current_score
                    best_model = model
                    best_metrics = metrics
                    best_params = params
            
            logging.info(f"Best DBSCAN parameters: {best_params}")
            logging.info(f"Best DBSCAN metrics: {best_metrics}")
            
            return best_model, best_metrics, best_params
            
        except Exception as e:
            raise MyException(e, sys)

    def compare_models(self, X: np.ndarray) -> Tuple[Any, str, Dict[str, float], Dict]:
        """
        Compare different clustering models and select the best one.
        
        :param X: np.ndarray
            The input features.
        :return: Tuple[Any, str, Dict[str, float], Dict]
            Best model, model name, best metrics, and best parameters.
        """
        try:
            logging.info("Comparing different clustering models")
            
            models_results = {}
            
            # Train KMeans
            kmeans_model, kmeans_metrics, kmeans_params = self.train_kmeans(X)
            models_results['KMeans'] = {
                'model': kmeans_model,
                'metrics': kmeans_metrics,
                'params': kmeans_params,
                'score': kmeans_metrics['silhouette_score']
            }
            
            # Train Agglomerative Clustering
            agg_model, agg_metrics, agg_params = self.train_agglomerative(X)
            models_results['AgglomerativeClustering'] = {
                'model': agg_model,
                'metrics': agg_metrics,
                'params': agg_params,
                'score': agg_metrics['silhouette_score']
            }
            
            # Train DBSCAN
            dbscan_model, dbscan_metrics, dbscan_params = self.train_dbscan(X)
            models_results['DBSCAN'] = {
                'model': dbscan_model,
                'metrics': dbscan_metrics,
                'params': dbscan_params,
                'score': dbscan_metrics['silhouette_score']
            }
            
            # Select the best model based on silhouette score
            best_model_name = max(models_results.keys(), key=lambda k: models_results[k]['score'])
            best_result = models_results[best_model_name]
            
            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model score: {best_result['score']}")
            
            return (best_result['model'], best_model_name, 
                   best_result['metrics'], best_result['params'])
            
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.
        
        :return: ModelTrainerArtifact
            The artifact containing the results of model training.
        """
        try:
            logging.info("Starting model training process")
            
            # Load transformed data
            df = self.load_transformed_data()
            
            # Prepare features for clustering
            X = self.prepare_features(df)
            
            # Compare models and select the best one
            best_model, best_model_name, best_metrics, best_params = self.compare_models(X)
            
            # Check if the model meets the expected accuracy
            model_score = best_metrics['silhouette_score']
            if model_score < self.model_trainer_config.expected_accuracy:
                logging.warning(f"Best model score {model_score} is below expected accuracy {self.model_trainer_config.expected_accuracy}")
            else:
                logging.info(f"Model meets expected accuracy. Score: {model_score}")
            
            # Create directories
            os.makedirs(self.model_trainer_config.trained_model_dir, exist_ok=True)
            
            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model ({best_model_name}) saved to {self.model_trainer_config.trained_model_file_path}")
            
            # Prepare train and test metrics (for clustering, we use the same data)
            train_metric_artifact = {
                'model_name': best_model_name,
                'model_parameters': best_params,
                **best_metrics
            }
            
            test_metric_artifact = train_metric_artifact.copy()  # Same metrics for clustering
            
            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric_artifact,
                test_metric_artifact=test_metric_artifact
            )
            
            logging.info("Model training process completed")
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
            
        except Exception as e:
            raise MyException(e, sys)
