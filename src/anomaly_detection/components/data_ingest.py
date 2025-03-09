import os

import zipfile as zip
from pathlib import Path
import urllib.request as request

from anomaly_detection import logger
from anomaly_detection.utils.common import get_size 
from anomaly_detection.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):

        train_file_path = Path(self.config.data_source_dir, self.config.train_data_file_name)
        test_file_path = Path(self.config.data_source_dir, self.config.test_data_file_name)
        destination_path = Path(self.config.root_dir)

        os.system(f"cp {train_file_path} {destination_path}")
        os.system(f"cp {test_file_path} {destination_path}")
       
        logger.info(f"file - {train_file_path} downloaded to {destination_path}")
        logger.info(f"file - {test_file_path} downloaded to {destination_path}")
