



# from joblib import dump, load
# from pathlib import Path

# from anomaly_detection import logger
# from anomaly_detection.utils.common import read_file
# from anomaly_detection.config.configuration import ConfigurationManager
# from anomaly_detection.components.anomaly_report import AnomalyReport, DNSAttackReport
# # from anomaly_detection.components.dns_attack_report import DNSAttackReport
# class AnomalyReportPipeline:
#     def __init__(self):
#         pass
    
#     def main(self):
#         config_manager = ConfigurationManager()
#         anomaly_report_config = config_manager.get_anomaly_report_config()
#         test_data_file_path = Path(anomaly_report_config.data_source_dir, anomaly_report_config.test_data_file_name)
#         test_data = read_file(test_data_file_path)

#         train_data_file_path = Path(anomaly_report_config.data_source_dir, anomaly_report_config.train_data_file_name)
#         train_data = read_file(train_data_file_path)

#         model_pipeline_file_path = Path(anomaly_report_config.model_pipeline_source_dir, anomaly_report_config.model_pipeline_file_name)
#         model_pipeline = load(model_pipeline_file_path)
#         confusion_matrix_file_path = Path(anomaly_report_config.confusion_matrix_source_dir, anomaly_report_config.confusion_matrix_file_name)
#         confusion_matrix_report = read_file(confusion_matrix_file_path)
#         predicted_df_file_path = Path(anomaly_report_config.confusion_matrix_source_dir, anomaly_report_config.predicted_df_file_name)
#         predicted_df = read_file(predicted_df_file_path)
#         anomaly_report = AnomalyReport(config=anomaly_report_config, 
#                                         train_data=train_data, 
#                                         test_data=test_data, 
#                                         model_pipeline=model_pipeline, 
#                                         confusion_matrix_report=confusion_matrix_report)
#         anomaly_report.generate_anomaly_report()
#         dns_attack_report = DNSAttackReport(predicted_df=predicted_df, 
#                                             data={"train_data":train_data, "test_data":test_data}, 
#                                             type="bucket", 
#                                             time_window=60)
#         dns_attack_report.generate_attack_report()
    
       
#         logger.info(f"\n\nAnomaly Report: \n{anomaly_report}\n\n")

# if __name__ == "__main__":
    
#     try:
#         logger.info(f"{'>>'*20} Stage 05 Anomaly Report {'<<'*20}")
#         obj = AnomalyReportPipeline()
#         obj.main()
#         logger.info(f">>>>>> stage Anomaly Report completed <<<<<<\n\nX==========X")
#     except Exception as e:
#         logger.exception(e)
#         raise e 