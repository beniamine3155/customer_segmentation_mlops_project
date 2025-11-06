import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from src.exception import MyException
from src.logger import logging

class CustomerSegmentMapping:
    """
    Mapping class for customer segmentation clusters
    """
    def __init__(self, n_clusters: int = 6):
        # Create meaningful cluster mapping for customer segmentation
        if n_clusters == 6:
            self.cluster_mapping = {
                0: "Budget_Conscious",
                1: "High_Value",
                2: "Regular_Customers", 
                3: "Premium_Shoppers",
                4: "Occasional_Buyers",
                5: "Loyal_Customers"
            }
        elif n_clusters == 2:
            self.cluster_mapping = {
                0: "Low_Value_Customers",
                1: "High_Value_Customers"
            }
        elif n_clusters == 3:
            self.cluster_mapping = {
                0: "Low_Value_Customers",
                1: "Medium_Value_Customers", 
                2: "High_Value_Customers"
            }
        elif n_clusters == 4:
            self.cluster_mapping = {
                0: "Budget_Conscious",
                1: "Regular_Customers",
                2: "Premium_Shoppers", 
                3: "Loyal_Customers"
            }
        elif n_clusters == 5:
            self.cluster_mapping = {
                0: "Budget_Conscious",
                1: "Regular_Customers",
                2: "Premium_Shoppers",
                3: "High_Value",
                4: "Loyal_Customers"
            }
        else:
            # Default to generic naming for other cluster counts
            self.cluster_mapping = {i: f"Segment_{i}" for i in range(n_clusters)}
        
        self.n_clusters = n_clusters
    
    def _asdict(self):
        return self.cluster_mapping
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    
    def get_segment_name(self, cluster_id: int) -> str:
        """Get segment name for a cluster ID"""
        return self.cluster_mapping.get(cluster_id, f"Segment_{cluster_id}")

class CustomerSegmentationModel:
    def __init__(self, preprocessing_object: StandardScaler, trained_model_object: object, 
                 feature_columns: list):
        """
        Customer Segmentation Model for clustering predictions
        
        :param preprocessing_object: StandardScaler object for feature scaling
        :param trained_model_object: Trained clustering model (KMeans, AgglomerativeClustering, or DBSCAN)
        :param feature_columns: List of feature columns used for clustering
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.feature_columns = feature_columns
        
        # Get number of clusters from the model
        if hasattr(trained_model_object, 'n_clusters'):
            n_clusters = trained_model_object.n_clusters
        elif hasattr(trained_model_object, 'labels_'):
            # For DBSCAN or models with labels_ attribute
            unique_labels = np.unique(trained_model_object.labels_)
            # Filter out noise points (-1) if present
            n_clusters = len(unique_labels[unique_labels >= 0])
        else:
            n_clusters = 6  # Default for 6 clusters based on your data
            
        logging.info(f"Detected {n_clusters} clusters for segmentation mapping")
        self.segment_mapping = CustomerSegmentMapping(n_clusters)

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Predict customer segments for the input dataframe
        
        :param dataframe: Input DataFrame containing customer data
        :return: Array of cluster predictions
        """
        try:
            logging.info("Starting customer segmentation prediction process.")
            
            # Step 1: Select only the feature columns used during training
            if not all(col in dataframe.columns for col in self.feature_columns):
                missing_cols = [col for col in self.feature_columns if col not in dataframe.columns]
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            feature_data = dataframe[self.feature_columns]
            logging.info(f"Selected feature columns: {self.feature_columns}")
            
            # Step 2: Apply scaling transformations using the pre-trained preprocessing object
            transformed_features = self.preprocessing_object.transform(feature_data)
            logging.info(f"Applied scaling transformation. Shape: {transformed_features.shape}")

            # Step 3: Perform clustering prediction using the trained model
            logging.info("Using the trained clustering model to get segment predictions")
            
            if hasattr(self.trained_model_object, 'predict'):
                # For KMeans and AgglomerativeClustering
                cluster_predictions = self.trained_model_object.predict(transformed_features)
            else:
                # For DBSCAN or models without predict method
                cluster_predictions = self.trained_model_object.fit_predict(transformed_features)
            
            logging.info(f"Prediction completed. Found {len(np.unique(cluster_predictions))} unique segments")
            return cluster_predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e

    def predict_with_segments(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Predict customer segments and return DataFrame with segment names
        
        :param dataframe: Input DataFrame containing customer data
        :return: DataFrame with cluster predictions and segment names
        """
        try:
            # Get cluster predictions
            cluster_predictions = self.predict(dataframe)
            
            # Create result DataFrame
            result_df = dataframe.copy()
            result_df['Cluster_ID'] = cluster_predictions
            result_df['Segment_Name'] = [self.segment_mapping.get_segment_name(cluster_id) 
                                       for cluster_id in cluster_predictions]
            
            # Add cluster statistics
            unique_clusters, counts = np.unique(cluster_predictions, return_counts=True)
            logging.info("Cluster distribution:")
            for cluster_id, count in zip(unique_clusters, counts):
                segment_name = self.segment_mapping.get_segment_name(cluster_id)
                percentage = (count / len(cluster_predictions)) * 100
                logging.info(f"  {segment_name} (ID: {cluster_id}): {count} customers ({percentage:.1f}%)")
            
            return result_df
            
        except Exception as e:
            logging.error("Error occurred in predict_with_segments method", exc_info=True)
            raise MyException(e, sys) from e

    def get_cluster_centers(self):
        """
        Get cluster centers if available (for KMeans)
        """
        try:
            if hasattr(self.trained_model_object, 'cluster_centers_'):
                return self.trained_model_object.cluster_centers_
            else:
                logging.warning("Cluster centers not available for this model type")
                return None
        except Exception as e:
            logging.error("Error occurred in get_cluster_centers method", exc_info=True)
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"CustomerSegmentationModel(model={type(self.trained_model_object).__name__}, features={len(self.feature_columns)})"

    def __str__(self):
        return f"CustomerSegmentationModel with {type(self.trained_model_object).__name__} clustering"