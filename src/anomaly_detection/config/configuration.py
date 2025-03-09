

from pathlib import Path

from src.anomaly_detection import logger
from src.anomaly_detection.constants import *
from src.anomaly_detection.utils.common import *
from src.anomaly_detection.entity.config_entity import *

class ConfigurationManager:
    
    def __init__(self, config_filepath=Path(CONFIG_FILE_PATH),
                params_filepath=Path(PARAMS_FILE_PATH)):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config_data_ingestion = self.config.data_ingestion
        create_directories([config_data_ingestion.root_dir])
        logger.info(f"Creating directory: {config_data_ingestion.root_dir}")
        data_ingestion_config = DataIngestionConfig(
            root_dir=config_data_ingestion  .root_dir,
            data_source_dir=config_data_ingestion.data_source_dir,
            train_data_file_name=config_data_ingestion.train_data_file_name,
            test_data_file_name=config_data_ingestion.test_data_file_name,
        )
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config_data_transformation = self.config.data_transformation
        create_directories([config_data_transformation.root_dir])
        logger.info(f"Creating directory: {config_data_transformation.root_dir}")
        data_transformation_config = DataTransformationConfig(
            root_dir=config_data_transformation.root_dir,
            data_source_dir=config_data_transformation.data_source_dir,
            train_data_file_name=config_data_transformation.train_data_file_name,
            test_data_file_name=config_data_transformation.test_data_file_name,
            params=self.params
        )   
        return data_transformation_config
    
    def get_model_train_config(self) -> ModelTrainConfig:
        config_model_train = self.config.model_train
        create_directories([config_model_train.root_dir])
        logger.info(f"Creating directory: {config_model_train.root_dir}")
        model_train_config = ModelTrainConfig(
            root_dir=config_model_train.root_dir,
            data_source_dir=config_model_train.data_source_dir,
            train_data_file_name=config_model_train.train_data_file_name,
            model_config=config_model_train.model_config,
            params=self.params
        )
        return model_train_config
    
    def get_model_test_config(self) -> ModelTestConfig:
        config_model_test = self.config.model_test
        create_directories([config_model_test.root_dir])
        logger.info(f"Creating directory: {config_model_test.root_dir}")
        model_test_config = ModelTestConfig(
            root_dir=config_model_test.root_dir,
            data_source_dir=config_model_test.data_source_dir,
            test_bucket_data_file_name=config_model_test.test_bucket_data_file_name,
            test_flat_bucket_data_file_name=config_model_test.test_flat_bucket_data_file_name,
            model_pipeline_source_dir=config_model_test.model_pipeline_source_dir,
            model_pipeline_file_name=config_model_test.model_pipeline_file_name,
            predicted_label_file_name=config_model_test.predicted_label_file_name,
            confusion_matrix_file_name=config_model_test.confusion_matrix_file_name,
            params=self.params
        )
        return model_test_config
