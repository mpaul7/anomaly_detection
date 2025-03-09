
import os
import jinja2
import numpy as np
import pandas as pd
from datetime import datetime

import base64
from io import BytesIO

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (auc,
                            roc_curve, 
                            accuracy_score,
                            classification_report,
                            precision_recall_curve, 
                            average_precision_score)

class DNSAttackModelReport:
    """Class for generating DNS attack detection model reports with feature distribution visualizations"""
    
    def __init__(self, train_data, test_data, model, results_df, target_features, 
                 y_test, y_pred_labels, anomaly_scores, feature_importances, 
                 confusion_matrix_report):
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.results_df = results_df
        self.target_features = target_features
        self.y_test = y_test
        self.y_pred_labels = y_pred_labels
        self.anomaly_scores = anomaly_scores
        self.feature_importances = feature_importances
        self.confusion_matrix_report = confusion_matrix_report
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_feature_boxplots(self):
        """Generate box plots for each target feature comparing DNS vs DNS Attack traffic"""
        feature_boxplots = {}
        
        for feature in self.target_features:
            plt.figure(figsize=(10, 6))
            
            # Create DataFrame with feature values and labels for plotting
            plot_data = pd.DataFrame({
                'Traffic Type': self.test_data['label'],
                'Value': self.test_data[feature]
            })
            
            # Create box plot using seaborn
            sns.boxplot(x='Traffic Type', y='Value', data=plot_data)
            
            plt.title(f'Box Plot of {feature} by Traffic Type')
            plt.xlabel('Traffic Type')
            plt.ylabel(f'{feature} Value')
            plt.grid(True, alpha=0.3)
            
            # Convert plot to base64 string
            feature_boxplots[feature] = self._fig_to_base64(plt.gcf())
            plt.close()
            
        return feature_boxplots
    def generate_feature_distributions(self):
        """Generate distribution plots for each target feature in training data"""
        feature_plots = {}
        
        for feature in self.target_features:
            plt.figure(figsize=(10, 6))
            
            # Get raw values for normal and attack samples
            normal_data = self.test_data[self.test_data['label'] == 'dns'][feature]
            attack_data = self.test_data[self.test_data['label'] == 'dns_attack'][feature]
            
            # Create overlapping histograms
            plt.hist(normal_data, bins=50, alpha=0.5, label='Normal DNS', weights=np.log(np.ones(len(normal_data)) / len(normal_data) * 100))
            plt.hist(attack_data, bins=50, alpha=0.5, label='DNS Attack', weights=np.log(np.ones(len(attack_data)) / len(attack_data) * 100))
            
            plt.title(f'Distribution of {feature} by Traffic Type')
            plt.xlabel(f'{feature} Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Convert plot to base64 string
            feature_plots[feature] = self._fig_to_base64(plt.gcf())
            plt.close()
            
            
        return feature_plots
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for HTML embedding"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def generate_executive_summary(self):
        """Generate executive summary with key metrics"""
        accuracy = np.mean(self.y_test == self.y_pred_labels)
        attack_detection_rate = np.mean(
            (self.y_pred_labels == 'dns_attack')[self.y_test == 'dns_attack']
        )
        
        return {
            'accuracy': f"{accuracy:.2%}",
            'attack_detection_rate': f"{attack_detection_rate:.2%}",
            'total_samples': len(self.test_data),
            'attack_samples': sum(self.y_test == 'dns_attack'),
            'normal_samples': sum(self.y_test == 'dns'),
            'timestamp': self.timestamp
        }
    
    def generate_dataset_summary(self):
        """Generate detailed dataset statistics"""
        return {
            'training': {
                'total_samples': len(self.train_data),
                'distribution': self.train_data['label'].value_counts().to_dict(),
                'features': len(self.target_features),
                'feature_names': self.target_features
            },
            'testing': {
                'total_samples': len(self.test_data),
                'distribution': self.test_data['label'].value_counts().to_dict()
            }
        }
    
    def generate_model_configuration(self):
        """Extract and format model configuration"""
        iso_forest = self.model.best_estimator_.named_steps['iso_forest']
        return {
            'type': 'Isolation Forest',
            'parameters': {
                'n_estimators': iso_forest.n_estimators,
                'contamination': iso_forest.contamination,
                'max_samples': iso_forest.max_samples,
                'max_features': iso_forest.max_features
            },
            'best_params': self.model.best_params_
        }
    
    def generate_performance_metrics(self):
        """Generate comprehensive performance metrics"""
        fpr, tpr, _ = roc_curve(self.y_test == 'dns_attack', self.anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(
            self.y_test == 'dns_attack', 
            self.anomaly_scores
        )
        avg_precision = average_precision_score(
            self.y_test == 'dns_attack', 
            self.anomaly_scores
        )
        
        # Generate ROC curve plot
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_plot = self._fig_to_base64(plt.gcf())
        plt.close()
        
        return {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'roc_plot': roc_plot
        }
    
    def generate_feature_analysis(self):
        """Generate feature importance analysis and visualizations"""
        # Feature importance plot
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'Feature': self.target_features,
            'Importance': self.feature_importances
        }).sort_values('Importance', ascending=True)
        
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.title('Feature Importance Scores')
        plt.xlabel('Importance Score')
        feature_plot = self._fig_to_base64(plt.gcf())
        plt.close()
        
        return {
            'importance_plot': feature_plot,
            'importance_data': importance_df.to_dict('records')
        }
    
    def generate_anomaly_analysis(self):
        """Generate anomaly score analysis and visualizations"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='true_label', y='normalized_anomaly_score', data=self.results_df)
        plt.title('Anomaly Score Distribution by Class')
        score_dist_plot = self._fig_to_base64(plt.gcf())
        plt.close()
        
        return {
            'distribution_plot': score_dist_plot,
            'statistics': self.results_df.groupby('true_label')['normalized_anomaly_score'].describe().to_dict()
        }
    
    def generate_anomaly_distribution(self):
        """Generate distribution analysis of anomaly scores"""
        plt.figure(figsize=(10, 5))
        plt.hist(self.anomaly_scores[self.y_test == 'dns'], bins=50, alpha=0.5, label='Normal', density=True)
        plt.hist(self.anomaly_scores[self.y_test == 'dns_attack'], bins=50, alpha=0.5, label='Attack', density=True)
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores')
        plt.legend()
        dist_plot = self._fig_to_base64(plt.gcf())
        plt.close()

        return {
            'distribution_plot': dist_plot,
            'statistics': {
                'normal': {
                    'mean': np.mean(self.anomaly_scores[self.y_test == 'dns']),
                    'std': np.std(self.anomaly_scores[self.y_test == 'dns'])
                },
                'attack': {
                    'mean': np.mean(self.anomaly_scores[self.y_test == 'dns_attack']),
                    'std': np.std(self.anomaly_scores[self.y_test == 'dns_attack'])
                }
            }
        }
    
    def generate_model_performance_section(self):
        # Add confusion matrix report to the performance metrics
        performance_metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred_labels),
            'classification_report': classification_report(self.y_test, self.y_pred_labels),
            'confusion_matrix_report': self.confusion_matrix_report.to_html(),  # Convert DataFrame to HTML
        }
        return performance_metrics
    
    def generate_html_report(self, output_path):
        """Generate complete HTML report"""
        # Add the correct template directory path
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')  # or wherever your templates are stored
        template_env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir))
        template = template_env.get_template("report_template.html")
        
        # Gather all report components
        report_data = {
            'summary': self.generate_executive_summary(),
            'dataset_summary': self.generate_dataset_summary(),
            'model_configuration': self.generate_model_configuration(),
            'performance_metrics': self.generate_performance_metrics(),
            'feature_analysis': self.generate_feature_analysis(),
            'anomaly_analysis': self.generate_anomaly_analysis(),
            'anomaly_distribution': self.generate_anomaly_distribution(),
            'model_performance': self.generate_model_performance_section(),
            'confusion_matrix_report': self.confusion_matrix_report.to_html(),
            'feature_distributions': self.generate_feature_distributions(),
            'feature_boxplots': self.generate_feature_boxplots()
        }
        
        # Render HTML
        html_output = template.render(report_data)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(html_output)
        
        return output_path

class DNSAttackReport:
    
    def __init__(self, predicted_df, data, type, time_window=60):
        self.predicted_df = predicted_df
        self.data = data
        self.type = type    
        self.time_window = time_window
        
    def generate_attack_report(self):
        
        """Generate attack report"""
        if self.type == 'bucket':
            attack_buckets = self.predicted_df[self.predicted_df['predicted_label'] == 'dns_attack']['bucket'].unique()
            
            attack_flows = []
            for bucket in attack_buckets:
                bucket_flows = self.data[self.data['bucket'] == bucket]
                bucket_suspicious_flows = bucket_flows[
                    # Check for high number of responses compared to requests 
                    ((bucket_flows['pl_bwd_count'] / bucket_flows['pl_fwd_count']) > 1) 
                ]
                attack_flows.append(bucket_suspicious_flows)
                
            attack_flows = pd.concat(attack_flows)
            total_flows = len(self.data)
            attack_flows_count = len(attack_flows)
            good_flows = total_flows - attack_flows_count
            
            self.print_report(total_flows, attack_flows, good_flows)
            self.analyze_attack_time_windows(attack_flows)
            return attack_flows
            

        elif self.type == 'flat':
            # For flat file analysis, identify suspicious flows directly
            attack_flows = self.predicted_df[self.predicted_df['predicted_label'] == 'dns_attack']
            good_flows = self.predicted_df[self.predicted_df['predicted_label'] == 'dns']
            total_flows = len(self.predicted_df)
            attack_flows_count = len(attack_flows) 
            self.print_report(total_flows, attack_flows, good_flows)
            # Move this code to a helper method and call it from both if/else branches
            self.analyze_attack_time_windows(attack_flows)
            return attack_flows
        
    def print_report(self, total_flows, attack_flows, good_flows):
        attack_flows_count = len(attack_flows) 

        attack_percentage = (attack_flows_count / total_flows) * 100
        good_percentage = (good_flows / total_flows) * 100

        unique_attack_dips = attack_flows['dip'].unique()
        print("\nDNS Attack Detection Summary:")
        print("=" * 60)
        print("{:<25} {:>15} {:>15}".format("Flow Type", "Count", "Percentage"))
        print("=" * 60)
        print("{:<25} {:>15,}".format("Total Flows", total_flows))
        print("{:<25} {:>15,} {:>14.2f}%".format("Normal DNS Flows", good_flows, good_percentage))
        print("{:<25} {:>15,} {:>14.2f}%".format("Suspicious DNS Flows", attack_flows_count, attack_percentage))
        print("=" * 60)
        print("\nSuspected IP Addresses:")
        print("=" * 45)
        print("{:<5} {:<20} {:<15}".format("#.", "IP Address", "Suspicious Flows"))
        print("=" * 45)
        for i, ip in enumerate(unique_attack_dips, 1):
            count = len(attack_flows[attack_flows['dip'] == ip])
            print("{:<5} {:<20} {:<15,}".format(i, ip, count))
            
    def analyze_attack_time_windows(self, attack_flows):
            attack_flows['first_timestamp'] = (attack_flows['first_timestamp'] // 1).astype(int)
            attack_flows = attack_flows.sort_values(by='first_timestamp', ascending=True)
            # Calculate time ranges in 2 second intervals
            min_time = attack_flows['first_timestamp'].min()
            max_time = attack_flows['first_timestamp'].max()
            time_ranges = range(min_time, max_time + self.time_window, self.time_window)
            
            # Count attacks in each 2 second window
            print(f"\n\nDNS Attack Frequency ({self.time_window}-seconds intervals):")
            print("=" * 120)
            print("{:<5}{:<40}{:5} {:<15} {:<15} {:<50}".format("#","Time Window", "", "Attack Count", "Percentage", "Target IPs"))
            print("=" * 120)
            for start_time in time_ranges:
                end_time = start_time + self.time_window
                count = len(attack_flows[(attack_flows['first_timestamp'] >= start_time) & 
                                    (attack_flows['first_timestamp'] < end_time)])
                if count > 0:
                    start_datetime = pd.to_datetime(start_time, unit='s').tz_localize('UTC').tz_convert('US/Eastern')
                    end_datetime = pd.to_datetime(end_time, unit='s').tz_localize('UTC').tz_convert('US/Eastern')
                    window_flows = attack_flows[(attack_flows['first_timestamp'] >= start_time) & (attack_flows['first_timestamp'] < end_time)]
                    unique_ips = window_flows['dip'].unique()
                    
                    print("{:<5} {:<40}{:5} {:<15} {:<15} {:<50}     ".format(
                        str(list(time_ranges).index(start_time) + 1),
                        f"{start_datetime.strftime('%Y-%m-%d %H:%M:%S')} - {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}", "",
                        str(count),
                        f"{count / len(attack_flows):.2%}",
                        ", ".join(unique_ips)
                    ))