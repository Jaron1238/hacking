#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation für das WLAN-Analyse-Tool.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
import joblib

from ..constants import Constants, ErrorCodes, get_error_message
from ..exceptions import ValidationError, ResourceError
from ..validation import validate_dataframe
from ..metrics import record_timing, record_counter

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Umfassender Model Evaluator für verschiedene ML-Aufgaben."""
    
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def evaluate_classification_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                    model_name: str = "classification_model") -> Dict[str, Any]:
        """Evaluiere Klassifikationsmodell."""
        logger.info(f"Evaluating classification model: {model_name}")
        start_time = time.time()
        
        try:
            # Vorhersagen
            y_pred = model.predict(X_test)
            
            # Wahrscheinlichkeiten (falls verfügbar)
            y_proba = None
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            
            # Basis-Metriken
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Detaillierter Report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC AUC (falls binär oder multi-class)
            roc_auc = None
            if y_proba is not None:
                try:
                    if len(np.unique(y_test)) == 2:  # Binär
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:  # Multi-class
                        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                except Exception as e:
                    logger.warning(f"ROC AUC calculation failed: {e}")
            
            # Cross-Validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            
            # Ergebnisse
            evaluation = {
                'model_name': model_name,
                'model_type': 'classification',
                'timestamp': time.time(),
                'evaluation_duration': time.time() - start_time,
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'roc_auc': float(roc_auc) if roc_auc is not None else None,
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std())
                },
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_proba.tolist() if y_proba is not None else None,
                'n_samples': len(X_test),
                'n_features': X_test.shape[1],
                'n_classes': len(np.unique(y_test))
            }
            
            # Metriken aufzeichnen
            record_timing("ml_classification_evaluation_duration", evaluation['evaluation_duration'])
            record_counter("ml_classification_evaluations", 1)
            
            # History aktualisieren
            self.evaluation_history.append(evaluation)
            
            logger.info(f"Classification evaluation completed: accuracy={accuracy:.4f}, f1={f1:.4f}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in classification evaluation: {e}")
            return {
                'model_name': model_name,
                'model_type': 'classification',
                'error': str(e),
                'evaluation_duration': time.time() - start_time
            }
    
    def evaluate_regression_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                                model_name: str = "regression_model") -> Dict[str, Any]:
        """Evaluiere Regressionsmodell."""
        logger.info(f"Evaluating regression model: {model_name}")
        start_time = time.time()
        
        try:
            # Vorhersagen
            y_pred = model.predict(X_test)
            
            # Metriken
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Relative Metriken
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
            
            # Cross-Validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='r2')
            
            # Residuals
            residuals = y_test - y_pred
            
            # Ergebnisse
            evaluation = {
                'model_name': model_name,
                'model_type': 'regression',
                'timestamp': time.time(),
                'evaluation_duration': time.time() - start_time,
                'metrics': {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2_score': float(r2),
                    'mape': float(mape),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std())
                },
                'predictions': y_pred.tolist(),
                'residuals': residuals.tolist(),
                'n_samples': len(X_test),
                'n_features': X_test.shape[1],
                'target_range': [float(np.min(y_test)), float(np.max(y_test))]
            }
            
            # Metriken aufzeichnen
            record_timing("ml_regression_evaluation_duration", evaluation['evaluation_duration'])
            record_counter("ml_regression_evaluations", 1)
            
            # History aktualisieren
            self.evaluation_history.append(evaluation)
            
            logger.info(f"Regression evaluation completed: r2={r2:.4f}, rmse={rmse:.4f}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in regression evaluation: {e}")
            return {
                'model_name': model_name,
                'model_type': 'regression',
                'error': str(e),
                'evaluation_duration': time.time() - start_time
            }
    
    def evaluate_clustering_model(self, model: Any, X: np.ndarray, labels: np.ndarray,
                                model_name: str = "clustering_model") -> Dict[str, Any]:
        """Evaluiere Clustering-Modell."""
        logger.info(f"Evaluating clustering model: {model_name}")
        start_time = time.time()
        
        try:
            # Clustering-Metriken
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            
            # Cluster-Statistiken
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            n_noise = np.sum(labels == -1) if -1 in labels else 0
            
            cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
            
            # Inertia (falls verfügbar)
            inertia = None
            if hasattr(model, 'inertia_'):
                inertia = model.inertia_
            
            # Ergebnisse
            evaluation = {
                'model_name': model_name,
                'model_type': 'clustering',
                'timestamp': time.time(),
                'evaluation_duration': time.time() - start_time,
                'metrics': {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'davies_bouldin_score': float(davies_bouldin),
                    'inertia': float(inertia) if inertia is not None else None
                },
                'cluster_info': {
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'cluster_sizes': cluster_sizes,
                    'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                    'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0
                },
                'labels': labels.tolist(),
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            # Metriken aufzeichnen
            record_timing("ml_clustering_evaluation_duration", evaluation['evaluation_duration'])
            record_counter("ml_clustering_evaluations", 1)
            
            # History aktualisieren
            self.evaluation_history.append(evaluation)
            
            logger.info(f"Clustering evaluation completed: silhouette={silhouette:.4f}, clusters={n_clusters}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error in clustering evaluation: {e}")
            return {
                'model_name': model_name,
                'model_type': 'clustering',
                'error': str(e),
                'evaluation_duration': time.time() - start_time
            }
    
    def compare_models(self, evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Vergleiche verschiedene Modelle."""
        logger.info(f"Comparing {len(evaluations)} models")
        
        comparison_data = []
        
        for eval_result in evaluations:
            if 'error' in eval_result:
                continue
            
            model_info = {
                'model_name': eval_result['model_name'],
                'model_type': eval_result['model_type'],
                'evaluation_duration': eval_result['evaluation_duration']
            }
            
            # Metriken hinzufügen
            if 'metrics' in eval_result:
                model_info.update(eval_result['metrics'])
            
            # Cluster-Info hinzufügen
            if 'cluster_info' in eval_result:
                model_info.update(eval_result['cluster_info'])
            
            comparison_data.append(model_info)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Bestes Modell für jeden Typ bestimmen
            if 'accuracy' in comparison_df.columns:
                best_classification = comparison_df.loc[comparison_df['accuracy'].idxmax()]
                logger.info(f"Best classification model: {best_classification['model_name']} "
                          f"(accuracy: {best_classification['accuracy']:.4f})")
            
            if 'r2_score' in comparison_df.columns:
                best_regression = comparison_df.loc[comparison_df['r2_score'].idxmax()]
                logger.info(f"Best regression model: {best_regression['model_name']} "
                          f"(r2: {best_regression['r2_score']:.4f})")
            
            if 'silhouette_score' in comparison_df.columns:
                best_clustering = comparison_df.loc[comparison_df['silhouette_score'].idxmax()]
                logger.info(f"Best clustering model: {best_clustering['model_name']} "
                          f"(silhouette: {best_clustering['silhouette_score']:.4f})")
        
        return comparison_df
    
    def generate_evaluation_report(self, evaluation: Dict[str, Any], 
                                 save_plots: bool = True) -> Dict[str, Any]:
        """Generiere detaillierten Evaluationsbericht."""
        logger.info(f"Generating evaluation report for {evaluation['model_name']}")
        
        report = {
            'summary': {
                'model_name': evaluation['model_name'],
                'model_type': evaluation['model_type'],
                'timestamp': evaluation['timestamp'],
                'evaluation_duration': evaluation['evaluation_duration']
            },
            'metrics': evaluation.get('metrics', {}),
            'plots': []
        }
        
        if save_plots and 'error' not in evaluation:
            try:
                # Plots generieren
                plot_paths = self._generate_evaluation_plots(evaluation)
                report['plots'] = plot_paths
            except Exception as e:
                logger.warning(f"Error generating plots: {e}")
                report['plot_error'] = str(e)
        
        # Bericht speichern
        report_path = self.results_dir / f"{evaluation['model_name']}_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved: {report_path}")
        return report
    
    def _generate_evaluation_plots(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generiere Evaluations-Plots."""
        plots = []
        model_name = evaluation['model_name']
        model_type = evaluation['model_type']
        
        try:
            if model_type == 'classification':
                # Confusion Matrix
                if 'confusion_matrix' in evaluation:
                    plt.figure(figsize=(8, 6))
                    cm = np.array(evaluation['confusion_matrix'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {model_name}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    
                    plot_path = self.results_dir / f"{model_name}_confusion_matrix.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots.append(str(plot_path))
                
                # ROC Curve (falls verfügbar)
                if 'probabilities' in evaluation and evaluation['probabilities'] is not None:
                    plt.figure(figsize=(8, 6))
                    y_proba = np.array(evaluation['probabilities'])
                    y_test = evaluation.get('y_test', [])
                    
                    if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:  # Binär
                        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                        auc = roc_auc_score(y_test, y_proba[:, 1])
                        
                        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
                        plt.plot([0, 1], [0, 1], 'k--', label='Random')
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title(f'ROC Curve - {model_name}')
                        plt.legend()
                        
                        plot_path = self.results_dir / f"{model_name}_roc_curve.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plots.append(str(plot_path))
            
            elif model_type == 'regression':
                # Actual vs Predicted
                if 'predictions' in evaluation:
                    plt.figure(figsize=(8, 6))
                    y_pred = evaluation['predictions']
                    y_test = evaluation.get('y_test', [])
                    
                    plt.scatter(y_test, y_pred, alpha=0.6)
                    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
                    plt.xlabel('Actual Values')
                    plt.ylabel('Predicted Values')
                    plt.title(f'Actual vs Predicted - {model_name}')
                    
                    plot_path = self.results_dir / f"{model_name}_actual_vs_predicted.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots.append(str(plot_path))
                
                # Residuals Plot
                if 'residuals' in evaluation:
                    plt.figure(figsize=(8, 6))
                    residuals = evaluation['residuals']
                    y_pred = evaluation['predictions']
                    
                    plt.scatter(y_pred, residuals, alpha=0.6)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.xlabel('Predicted Values')
                    plt.ylabel('Residuals')
                    plt.title(f'Residuals Plot - {model_name}')
                    
                    plot_path = self.results_dir / f"{model_name}_residuals.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots.append(str(plot_path))
            
            elif model_type == 'clustering':
                # Cluster Distribution
                if 'cluster_info' in evaluation:
                    plt.figure(figsize=(10, 6))
                    cluster_sizes = evaluation['cluster_info']['cluster_sizes']
                    
                    plt.bar(range(len(cluster_sizes)), cluster_sizes)
                    plt.xlabel('Cluster ID')
                    plt.ylabel('Cluster Size')
                    plt.title(f'Cluster Size Distribution - {model_name}')
                    
                    plot_path = self.results_dir / f"{model_name}_cluster_distribution.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots.append(str(plot_path))
        
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        return plots
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Hole Evaluations-Zusammenfassung."""
        if not self.evaluation_history:
            return {'total_evaluations': 0}
        
        # Statistiken berechnen
        total_evaluations = len(self.evaluation_history)
        
        # Nach Typ gruppieren
        type_counts = {}
        for eval_result in self.evaluation_history:
            model_type = eval_result.get('model_type', 'unknown')
            type_counts[model_type] = type_counts.get(model_type, 0) + 1
        
        # Durchschnittliche Metriken
        avg_metrics = {}
        for eval_result in self.evaluation_history:
            if 'metrics' in eval_result:
                for metric, value in eval_result['metrics'].items():
                    if value is not None:
                        if metric not in avg_metrics:
                            avg_metrics[metric] = []
                        avg_metrics[metric].append(value)
        
        # Durchschnittswerte berechnen
        for metric in avg_metrics:
            avg_metrics[metric] = np.mean(avg_metrics[metric])
        
        return {
            'total_evaluations': total_evaluations,
            'type_distribution': type_counts,
            'average_metrics': avg_metrics,
            'evaluation_duration_avg': np.mean([e.get('evaluation_duration', 0) for e in self.evaluation_history])
        }
    
    def export_evaluation_history(self, filepath: str) -> None:
        """Exportiere Evaluations-Historie."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2, default=str)
        
        logger.info(f"Evaluation history exported to {filepath}")


# Convenience-Funktionen
def create_model_evaluator(results_dir: str = "evaluation_results") -> ModelEvaluator:
    """Erstelle Model Evaluator."""
    return ModelEvaluator(results_dir)


def quick_model_evaluation(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                         model_type: str = "classification") -> Dict[str, Any]:
    """Schnelle Modell-Evaluation."""
    evaluator = create_model_evaluator()
    
    if model_type == "classification":
        return evaluator.evaluate_classification_model(model, X_test, y_test)
    elif model_type == "regression":
        return evaluator.evaluate_regression_model(model, X_test, y_test)
    elif model_type == "clustering":
        return evaluator.evaluate_clustering_model(model, X_test, y_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")