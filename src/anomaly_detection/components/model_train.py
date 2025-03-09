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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import OneClassSVM
from datetime import datetime
from io import BytesIO
import base64

import yaml
from pathlib import Path
from anomaly_detection import logger
from anomaly_detection.entity.config_entity import ModelTrainConfig

from anomaly_detection.utils.common import *

class ModelTrain:
    
    def __init__(self, config: ModelTrainConfig, train_data):
        self.config = config
        self.params = config.params
        self.train_data = train_data
    
    def _create_pipeline(self):
        steps = []
        model_config = read_yaml(Path(self.config.model_config))
        for step in model_config.pipeline.steps:
            if step['class'] == 'StandardScaler':
                steps.append((step['name'], StandardScaler()))
            elif step['class'] == 'IsolationForest':
                steps.append((
                    step['name'], 
                    IsolationForest(**step.get('params', {}))
                ))
        return Pipeline(steps)
    
    def _get_param_grid(self):
        model_config = read_yaml(Path(self.config.model_config))
        return model_config.param_grid
    
    def train_model(self):
        
        X_train = self.train_data[self.params.features.bucket_target_features]

        y_train = self.train_data['label']
        label_mapping = {label: i for i, label in enumerate(self.params.features.target_labels)}
        
        y_train = np.array([label_mapping[label] for label in y_train])

        # Create a pipeline with StandardScaler and IsolationForest
        pipeline = self._create_pipeline()

        # Define the parameter grid for hyperparameter tuning
        param_grid = self._get_param_grid()

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        pipe = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=kf, 
            scoring='f1_weighted', 
            n_jobs=-1
            )
        
        pipe.fit(X_train, y_train)

        logger.info(f"Best parameters: {pipe.best_params_}")   
        logger.info(f"Best score: {pipe.best_score_}")
        

        def _get_feature_importance(model, X):
            """Calculate feature importance scores for IsolationForest"""
            scores = np.zeros(X.shape[1])
            for estimator in model.estimators_:
                scores += estimator.feature_importances_
            return scores / len(model.estimators_)

        # Get feature importances from the trained IsolationForest
        feature_importances = _get_feature_importance(pipe.best_estimator_.named_steps['iso_forest'], X_train)
        self.feature_importances = feature_importances  # Store as instance variable
        logger.info(f"Feature importances: {feature_importances}")
        
        return pipe
    
        # def test_model(self, pipe):
            
        #     # Separate features and labels
        #     X_test = self.test_data[self.target_features]
        #     y_test = self.test_data['label']

        #     # Predict anomalies in the test data using the best model
        #     y_pred = pipe.predict(X_test)

        #     # Convert predictions to match the label format
        #     # 1 for 'dns' (normal) and 0 for 'dns_attack' (outlier)
        #     y_pred_labels = np.where(y_pred == -1, self.target_labels[1], self.target_labels[0])

        #     outliers = self.test_data[y_pred_labels == self.target_labels[1]]
        #     self.test_data['predicted_label'] = y_pred_labels
        #     # outliers.to_csv(output_outlier_file)

        #     # Generate confusion matrix
        #     conf_matrix = confusion_matrix(y_test, y_pred_labels, labels=self.target_labels)

        #     # Print the confusion matrix and classification report
        #     confusion_matrix_report = getClassificationReport(_confusion_matrix=conf_matrix,
        #                                                       traffic_classes=np.unique(y_test)
        #                                                       )
            
        #     return self.test_data, confusion_matrix_report
        
    # def generate_report(self, pipe):
    #     # Get test features
    #     X_test = self.test_data[self.target_features]
        
    #     # Add these two lines to generate predictions
    #     y_pred = pipe.predict(X_test)
    #     y_pred_labels = np.where(y_pred == -1, self.target_labels[1], self.target_labels[0])
    #     y_test = self.test_data['label']
        
    #     # Continue with existing code...
    #     decision_scores = pipe.decision_function(X_test)
    #     anomaly_scores = -pipe.score_samples(X_test)

    #     # Create a DataFrame with features and scores
    #     results_df = pd.DataFrame(X_test, columns=self.target_features)
    #     results_df['predicted_label'] = y_pred_labels
    #     results_df['true_label'] = y_test
    #     results_df['anomaly_score'] = anomaly_scores
    #     results_df['decision_score'] = decision_scores

    #     # Normalize scores to [0,1] range to make them more interpretable
    #     from sklearn.preprocessing import MinMaxScaler
    #     scaler = MinMaxScaler()
    #     results_df['normalized_anomaly_score'] = scaler.fit_transform(anomaly_scores.reshape(-1, 1))

    #     # Sort by anomaly score to see the most anomalous samples
    #     results_df_sorted = results_df.sort_values('normalized_anomaly_score', ascending=False)

    #     # Save detailed results
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     detailed_results_file = os.path.join(self.result_path, f'detailed_results_{timestamp}.csv')
    #     results_df_sorted.to_csv(detailed_results_file, index=False)

    #     # Get test results first
    #     _, confusion_matrix_report = self.test_model(pipe)
        
    #     from dns_attack_report import DNSAttackModelReport
    #     # Assuming you have your model and data ready
    #     report_generator = DNSAttackModelReport(
    #         train_data=self.train_data,
    #         test_data=self.test_data,
    #         model=pipe,
    #         results_df=results_df_sorted,
    #         y_test=y_test,
    #         y_pred_labels=y_pred_labels,
    #         anomaly_scores=anomaly_scores,
    #         feature_importances=self.feature_importances,
    #         confusion_matrix_report=confusion_matrix_report,
    #         target_features=self.target_features,
    #         # target_labels=self.target_labels
    #     )

    #     # Generate the report
    #     output_path = os.path.join(self.result_path, f'dns_attack_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    #     report_path = report_generator.generate_html_report(output_path)
    #     # print(f"Report generated at: {report_path}")
    


