from joblib import dump
from pathlib import Path

from anomaly_detection import logger
from anomaly_detection.utils.common import read_file
from anomaly_detection.config.configuration import ConfigurationManager
from anomaly_detection.components.model_train import ModelTrain

class ModelTrainPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        model_train_config = config_manager.get_model_train_config()
        train_data_file_path = Path(model_train_config.data_source_dir, model_train_config.train_data_file_name)
        train_data = read_file(train_data_file_path)
        model_train = ModelTrain(config=model_train_config, train_data=train_data)
        pipe = model_train.train_model()
        dump(pipe, Path(model_train_config.root_dir, 'model.joblib'))
        

if __name__ == "__main__":
    
    try:
        logger.info(f"{'>>'*20} Stage 03 Model Training {'<<'*20}")
        obj = ModelTrainPipeline()
        obj.main()
        logger.info(f">>>>>> stage Model Training completed <<<<<<\n\nX==========X")
    except Exception as e:
        logger.exception(e)
        raise e 