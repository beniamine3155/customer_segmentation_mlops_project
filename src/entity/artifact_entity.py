from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_data_path: str
    scaler_object_path: str
    feature_columns: list

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: dict
    test_metric_artifact: dict

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: dict
    best_model_metric_artifact: dict

