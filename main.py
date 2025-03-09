from src.anomaly_detection import logger
from src.anomaly_detection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.anomaly_detection.pipeline.stage_02_data_transformation import DataTransformationPipeline
from src.anomaly_detection.pipeline.stage_03_model_train import ModelTrainPipeline
from src.anomaly_detection.pipeline.stage_04_model_test import ModelTestPipeline
# from src.anomaly_detection.pipeline.stage_05_anomaly_report import AnomalyReportPipeline

try:
    logger.info(f">>>>>> stage Data Ingestion started <<<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>>> stage Data Ingestion completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage Data Transformation started <<<<<<")
    obj = DataTransformationPipeline()
    obj.main()
    logger.info(f">>>>>> stage Data Transformation completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage Model Train started <<<<<<")
    obj = ModelTrainPipeline()
    obj.main()
    logger.info(f">>>>>> stage Model Train completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

try:
    logger.info(f">>>>>> stage Model Test started <<<<<<")
    obj = ModelTestPipeline()
    obj.main()
    logger.info(f">>>>>> stage Model Test completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# try:
#     logger.info(f">>>>>> stage Anomaly Report started <<<<<<")
#     obj = AnomalyReportPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage Anomaly Report completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e