import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from pathlib import Path
from anomaly_detection import logger
from anomaly_detection.entity.config_entity import ModelTestConfig

from anomaly_detection.utils.common import *
from anomaly_detection.utils.classification_report import *
from anomaly_detection.components.anomaly_report import DNSAttackModelReport
class ModelTest:
    
    def __init__(self, config: ModelTestConfig, test_data, model_pipeline):
        self.config = config
        self.params = config.params
        self.test_data = test_data
        self.model_pipeline = model_pipeline
        
    def test_model(self):
        
        # Separate features and labels
        X_test = self.test_data[self.params.features.bucket_target_features]
        y_test = self.test_data['label']

        # Predict anomalies in the test data using the best model
        y_pred = self.model_pipeline.predict(X_test)

        # Convert predictions to match the label format
        # 1 for 'dns' (normal) and 0 for 'dns_attack' (outlier)
        y_pred_labels = np.where(y_pred == -1, self.params.features.target_labels[1], self.params.features.target_labels[0])

        outliers = self.test_data[y_pred_labels == self.params.features.target_labels[1]]
        self.test_data['predicted_label'] = y_pred_labels
        # outliers.to_csv(output_outlier_file)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_labels, labels=self.params.features.target_labels)

        # Print the confusion matrix and classification report
        confusion_matrix_report = getClassificationReport(_confusion_matrix=conf_matrix,
                                                        traffic_classes=np.unique(y_test)
                                                        )
        
        return self.test_data, confusion_matrix_report
    
    def generate_report(self, pipe, confusion_matrix_report):
        # Get test features
        X_test = self.test_data[self.params.features.bucket_target_features]
        
        # Add these two lines to generate predictions
        y_pred = pipe.predict(X_test)
        y_pred_labels = np.where(y_pred == -1, self.params.features.target_labels[1], self.params.features.target_labels[0])
        y_test = self.test_data['label']
        
        # Continue with existing code...
        decision_scores = pipe.decision_function(X_test)
        anomaly_scores = -pipe.score_samples(X_test)

        # Create a DataFrame with features and scores
        results_df = pd.DataFrame(X_test, columns=self.params.features.bucket_target_features)
        results_df['predicted_label'] = y_pred_labels
        results_df['true_label'] = y_test
        results_df['anomaly_score'] = anomaly_scores
        results_df['decision_score'] = decision_scores

        # Normalize scores to [0,1] range to make them more interpretable
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        results_df['normalized_anomaly_score'] = scaler.fit_transform(anomaly_scores.reshape(-1, 1))

        # Sort by anomaly score to see the most anomalous samples
        results_df_sorted = results_df.sort_values('normalized_anomaly_score', ascending=False)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_results_file = os.path.join(self.config.root_dir, f'detailed_results_{timestamp}.csv')
        results_df_sorted.to_csv(detailed_results_file, index=False)

        # Get test results first
        # _, confusion_matrix_report = self.test_model(pipe)
        
       
        # Assuming you have your model and data ready
        report_generator = DNSAttackModelReport(
            train_data=self.config.train_data,
            test_data=self.config.test_data,
            model=pipe,
            results_df=results_df_sorted,
            y_test=y_test,
            y_pred_labels=y_pred_labels,
            anomaly_scores=anomaly_scores,
            feature_importances=self.feature_importances,
            confusion_matrix_report=confusion_matrix_report,
            target_features=self.target_features,
            # target_labels=self.target_labels
        )

        # Generate the report
        output_path = os.path.join(self.result_path, f'dns_attack_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        report_path = report_generator.generate_html_report(output_path)
    