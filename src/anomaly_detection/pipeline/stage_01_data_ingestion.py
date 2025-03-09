from anomaly_detection import logger
from anomaly_detection.config.configuration import ConfigurationManager
from anomaly_detection.components.data_ingest import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()


if __name__ == "__main__":
    try:
        logger.info(f"{'>>'*20} Stage 01 Data Ingestion {'<<'*20}")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> stage Data Ingestion completed <<<<<<\n\nX==========X")
    except Exception as e:
        logger.exception(e)
        raise e