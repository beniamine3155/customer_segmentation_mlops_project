import sys
import pandas as pd
from src.entity.config_entity import CustomerSegmentationPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class CustomerSegmentationData:
    """
    Class to handle input data for customer segmentation prediction
    Based on the exact features used in training: 
    ['Age', 'Income', 'Total_Spend', 'Recency', 'NumWebPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
    """
    def __init__(self, 
                 age: float,
                 income: float,
                 total_spend: float,
                 recency: int,
                 num_web_purchases: int,
                 num_store_purchases: int,
                 num_web_visits_month: int):
        """
        Initialize customer data for prediction with exact training features
        
        Args:
            age: Customer age (derived from Year_Birth: 2025 - Year_Birth)
            income: Annual income of customer
            total_spend: Total spending across all product categories
            recency: Number of days since last purchase
            num_web_purchases: Number of purchases made through web
            num_store_purchases: Number of purchases made in stores
            num_web_visits_month: Number of visits to company's website in last month
        """
        try:
            self.Age = age
            self.Income = income
            self.Total_Spend = total_spend
            self.Recency = recency
            self.NumWebPurchases = num_web_purchases
            self.NumStorePurchases = num_store_purchases
            self.NumWebVisitsMonth = num_web_visits_month
                
        except Exception as e:
            raise MyException(e, sys) from e
    
    def get_customer_input_data_frame(self) -> DataFrame:
        """
        Convert customer data to DataFrame for prediction with exact feature order
        
        Returns:
            DataFrame: Customer data in DataFrame format with correct column order
        """
        try:
            # Create dictionary with exact feature names and order used in training
            customer_input_dict = {
                'Age': self.Age,
                'Income': self.Income,
                'Total_Spend': self.Total_Spend,
                'Recency': self.Recency,
                'NumWebPurchases': self.NumWebPurchases,
                'NumStorePurchases': self.NumStorePurchases,
                'NumWebVisitsMonth': self.NumWebVisitsMonth
            }
            
            # Convert to DataFrame with single row
            return pd.DataFrame([customer_input_dict])
            
        except Exception as e:
            raise MyException(e, sys) from e


class CustomerSegmentationClassifier:
    """
    Customer Segmentation Prediction Pipeline
    """
    def __init__(self):
        """
        Initialize the prediction pipeline
        """
        try:
            self.prediction_pipeline_config = CustomerSegmentationPredictorConfig()
            
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: DataFrame) -> dict:
        """
        Predict customer segments for input data
        
        Args:
            dataframe: Input DataFrame containing customer data
            
        Returns:
            dict: Prediction results with cluster and segment information
        """
        try:
            logging.info("Starting customer segmentation prediction")
            
            # Initialize the model estimator
            model_estimator = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path
            )
            
            # Check if model exists
            if not model_estimator.is_model_present(self.prediction_pipeline_config.model_file_path):
                raise Exception("Model not found in S3 bucket. Please train the model first.")
            
            # Load the model and make predictions
            logging.info("Loading model from S3 and making predictions")
            predictions = model_estimator.predict(dataframe)
            
            # Get detailed predictions with segment names
            loaded_model = model_estimator.load_model()
            detailed_predictions = loaded_model.predict_with_segments(dataframe)
            
            # Prepare results
            results = {
                "predictions": predictions.tolist(),
                "detailed_results": detailed_predictions.to_dict('records'),
                "total_customers": len(predictions),
                "unique_segments": len(set(predictions))
            }
            
            # Add cluster distribution
            cluster_counts = pd.Series(predictions).value_counts().sort_index()
            results["cluster_distribution"] = cluster_counts.to_dict()
            
            logging.info(f"Prediction completed for {len(predictions)} customers")
            logging.info(f"Found {len(set(predictions))} unique segments")
            
            return results
            
        except Exception as e:
            logging.error("Error in prediction pipeline", exc_info=True)
            raise MyException(e, sys) from e
    
    def predict_single_customer(self, customer_data: CustomerSegmentationData) -> dict:
        """
        Predict segment for a single customer
        
        Args:
            customer_data: CustomerSegmentationData object
            
        Returns:
            dict: Prediction result for single customer
        """
        try:
            logging.info("Starting single customer prediction")
            
            # Convert customer data to DataFrame
            customer_df = customer_data.get_customer_input_data_frame()
            
            # Get prediction
            results = self.predict(customer_df)
            
            # Extract single customer result
            single_result = {
                "customer_data": customer_df.to_dict('records')[0],
                "predicted_cluster": results["predictions"][0],
                "segment_details": results["detailed_results"][0],
                "prediction_confidence": "High" if len(results["predictions"]) == 1 else "Medium"
            }
            
            logging.info(f"Single customer prediction completed: Cluster {single_result['predicted_cluster']}")
            
            return single_result
            
        except Exception as e:
            logging.error("Error in single customer prediction", exc_info=True)
            raise MyException(e, sys) from e
    
    def predict_batch(self, csv_file_path: str) -> dict:
        """
        Predict customer segments for batch data from CSV file
        
        Args:
            csv_file_path: Path to CSV file containing customer data
            
        Returns:
            dict: Batch prediction results
        """
        try:
            logging.info(f"Starting batch prediction from file: {csv_file_path}")
            
            # Read CSV file
            customer_df = pd.read_csv(csv_file_path)
            logging.info(f"Loaded {len(customer_df)} customers from CSV")
            
            # Get predictions
            results = self.predict(customer_df)
            
            # Add file information
            results["source_file"] = csv_file_path
            results["processed_timestamp"] = pd.Timestamp.now().isoformat()
            
            logging.info("Batch prediction completed successfully")
            
            return results
            
        except Exception as e:
            logging.error("Error in batch prediction", exc_info=True)
            raise MyException(e, sys) from e


# Utility function for quick predictions
def predict_customer_segment(age: float, income: float, total_spend: float, 
                           recency: int, num_web_purchases: int, 
                           num_store_purchases: int, num_web_visits_month: int) -> dict:
    """
    Quick utility function for single customer prediction
    
    Args:
        age: Customer age (2025 - Year_Birth)
        income: Annual income
        total_spend: Total spending across all categories
        recency: Days since last purchase
        num_web_purchases: Number of web purchases
        num_store_purchases: Number of store purchases  
        num_web_visits_month: Web visits per month
        
    Returns:
        dict: Prediction result
    """
    try:
        # Create customer data object
        customer_data = CustomerSegmentationData(
            age=age,
            income=income,
            total_spend=total_spend,
            recency=recency,
            num_web_purchases=num_web_purchases,
            num_store_purchases=num_store_purchases,
            num_web_visits_month=num_web_visits_month
        )
        
        # Create classifier and predict
        classifier = CustomerSegmentationClassifier()
        result = classifier.predict_single_customer(customer_data)
        
        return result
        
    except Exception as e:
        logging.error("Error in quick prediction function", exc_info=True)
        raise MyException(e, sys) from e


# Additional utility function for predictions from raw customer data
def predict_from_raw_data(year_birth: int, income: float, 
                         mnt_wines: float, mnt_fruits: float, mnt_meat_products: float,
                         mnt_fish_products: float, mnt_sweet_products: float, mnt_gold_prods: float,
                         recency: int, num_web_purchases: int, 
                         num_store_purchases: int, num_web_visits_month: int) -> dict:
    """
    Utility function for prediction from raw customer data (as in original dataset)
    
    Args:
        year_birth: Year of birth (Age will be calculated as 2025 - year_birth)
        income: Annual income
        mnt_wines: Amount spent on wine
        mnt_fruits: Amount spent on fruits
        mnt_meat_products: Amount spent on meat
        mnt_fish_products: Amount spent on fish
        mnt_sweet_products: Amount spent on sweets
        mnt_gold_prods: Amount spent on gold products
        recency: Days since last purchase
        num_web_purchases: Number of web purchases
        num_store_purchases: Number of store purchases
        num_web_visits_month: Web visits per month
        
    Returns:
        dict: Prediction result
    """
    try:
        # Calculate derived features as done in training
        age = 2025 - year_birth
        total_spend = mnt_wines + mnt_fruits + mnt_meat_products + mnt_fish_products + mnt_sweet_products + mnt_gold_prods
        
        # Use the main prediction function
        result = predict_customer_segment(
            age=age,
            income=income,
            total_spend=total_spend,
            recency=recency,
            num_web_purchases=num_web_purchases,
            num_store_purchases=num_store_purchases,
            num_web_visits_month=num_web_visits_month
        )
        
        return result
        
    except Exception as e:
        logging.error("Error in raw data prediction function", exc_info=True)
        raise MyException(e, sys) from e
