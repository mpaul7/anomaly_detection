from anomaly_detection import logger
from anomaly_detection.config.configuration import ConfigurationManager
from anomaly_detection.components.data_transform import DataTransformation

class DataTransformationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform_data()


if __name__ == "__main__":
    try:
        logger.info(f"{'>>'*20} Stage 02 Data Transformation {'<<'*20}")
        obj = DataTransformationPipeline()
        obj.main()
        logger.info(f">>>>>> stage Data Transformation completed <<<<<<\n\nX==========X")
    except Exception as e:
        logger.exception(e)
        raise e