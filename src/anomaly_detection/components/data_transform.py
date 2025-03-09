
import time
from pathlib import Path

from anomaly_detection.entity.config_entity import DataTransformationConfig
from anomaly_detection.commons.data_bucketization import bucketize_data
from anomaly_detection.utils.common import *

class DataTransformation:
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.params = config.params
        
        
    def transform_data(self):
        start_time = time.time()
        
        train_file_path = Path(self.config.data_source_dir, self.config.train_data_file_name)
        test_file_path = Path(self.config.data_source_dir, self.config.test_data_file_name)
        train_df = read_file(train_file_path)   
        test_df = read_file(test_file_path)
        train_df = train_df[(train_df['sport'] == 53) | (train_df['dport'] == 53)]
        test_df = test_df[(test_df['sport'] == 53) | (test_df['dport'] == 53)]
        train_bucket, train_flat = bucketize_data(self.params, train_df)
        test_bucket, test_flat = bucketize_data(self.params, test_df)
        
        train_flat.to_csv(Path(self.config.root_dir, 'train_flat.csv'), index=False)
        test_flat.to_csv(Path(self.config.root_dir, 'test_flat_bucket.csv'), index=False)
        
        logger.info(f'\n\nData description flow based:\n{"=" * 40}')
        print(f'[{train_df.shape[0]}] : Training data DNS flows \n[{test_df.shape[0]}] : Test data DNS flows\n')
        
        logger.info(f'\n\nData description bucket based:\n{"=" * 40}')
        print(f'[{train_bucket.shape[0]}] : Training bucket data DNS flows \n[{test_bucket.shape[0]}] : Test bucket data DNS flows\n')
        train_bucket.to_csv(Path(self.config.root_dir, 'train_bucket.csv'), index=False)
        test_bucket.to_csv(Path(self.config.root_dir, 'test_bucket.csv'), index=False)
        
        end_time = time.time()
        logger.info(f"Data transformation completed in {end_time - start_time:.2f} seconds")