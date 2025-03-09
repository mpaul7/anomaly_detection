from joblib import load
from pathlib import Path

from anomaly_detection import logger
from anomaly_detection.utils.common import read_file
from anomaly_detection.components.model_test import ModelTest
from anomaly_detection.components.anomaly_report import DNSAttackReport 
from anomaly_detection.config.configuration import ConfigurationManager


class ModelTestPipeline:
    def __init__(self):
        pass
    
    def main(self):
        """Load the model pipeline and test the model"""
        config_manager = ConfigurationManager()
        model_test_config = config_manager.get_model_test_config()
        
        test_data_file_path = Path(model_test_config.data_source_dir, model_test_config.test_bucket_data_file_name)
        test_data = read_file(test_data_file_path)
        
        test_flat_data_file_path = Path(model_test_config.data_source_dir, model_test_config.test_flat_bucket_data_file_name)
        test_flat_data = read_file(test_flat_data_file_path)
        
        model_pipeline_file_path = Path(model_test_config.model_pipeline_source_dir, model_test_config.model_pipeline_file_name)
        model_pipeline = load(model_pipeline_file_path)
        
        model_test = ModelTest(config=model_test_config, test_data=test_data, model_pipeline=model_pipeline)
        predicted_df, confusion_matrix_report = model_test.test_model()
        
        predicted_label_file_path = Path(model_test_config.root_dir, model_test_config.predicted_label_file_name)
        predicted_df.to_csv(predicted_label_file_path, index=False)
        
        confusion_matrix_file_path = Path(model_test_config.root_dir, model_test_config.confusion_matrix_file_name)
        confusion_matrix_report.to_csv(confusion_matrix_file_path, index=False)
        logger.info(f"\n\nConfusion Matrix Report: \n{confusion_matrix_report}\n\n")
        
        """model_test.generate_report (model_pipeline, confusion_matrix_report) """
        dns_attack_report = DNSAttackReport(predicted_df, 
                                            data=test_flat_data,
                                            type=model_test_config.params.settings.type, 
                                            time_window=model_test_config.params.settings.attack_interval )
    
        """Generate the attack report"""
        attack_flows = dns_attack_report.generate_attack_report()
        attack_flow_file_path = Path(model_test_config.root_dir, "attack_flows.csv")
        attack_flows.to_csv(attack_flow_file_path, index=False)

        

if __name__ == "__main__":
    
    try:
        logger.info(f"{'>>'*20} Stage 04 Model Testing {'<<'*20}")
        obj = ModelTestPipeline()
        obj.main()
        logger.info(f">>>>>> stage Model Testing completed <<<<<<\n\nX==========X")
    except Exception as e:
        logger.exception(e)
        raise e 