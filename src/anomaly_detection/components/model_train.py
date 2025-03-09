import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold

from anomaly_detection.utils.common import *
from anomaly_detection import logger
from anomaly_detection.entity.config_entity import ModelTrainConfig

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