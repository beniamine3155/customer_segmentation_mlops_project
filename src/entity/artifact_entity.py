from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str