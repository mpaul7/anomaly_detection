from dataclasses import dataclass
from typing import List

@dataclass
class DataIngestionConfig:
    root_dir: str
    data_source_dir: str
    train_data_file_name: str
    test_data_file_name: str

@dataclass
class DataTransformationConfig:
    root_dir: str
    data_source_dir: str
    train_data_file_name: str
    test_data_file_name: str
    params: dict

@dataclass
class ModelTrainConfig:
    root_dir: str
    data_source_dir: str
    train_data_file_name: str
    model_config: str
    params: dict
    
@dataclass
class ModelTestConfig:
    root_dir: str
    data_source_dir: str
    test_bucket_data_file_name: str
    test_flat_bucket_data_file_name: str
    model_pipeline_source_dir: str
    model_pipeline_file_name: str
    predicted_label_file_name: str
    confusion_matrix_file_name: str
    params: dict