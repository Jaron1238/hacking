"""
Ensemble Models Plugin für Machine Learning.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import joblib
import json
from dataclasses import dataclass
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from plugins import BasePlugin, PluginMetadata

try:
    from sklearn.ensemble import (
        VotingClassifier, RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError as e:
    logging.warning(f"Einige ML-Bibliotheken nicht verfügbar: {e}")
    # Fallback-Importe
    try:
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
    except ImportError:
        pass

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Datenklasse für Modell-Performance-Metriken."""
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_mean: float
    cross_val_std: float
    training_time: float
    prediction_time: float

class Plugin(BasePlugin):
    """Ensemble Models für Geräte-Klassifizierung."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Ensemble Models",
            version="1.0.0",
            description="Ensemble Machine Learning Modelle für Geräte-Klassifizierung",
            author="WLAN-Tool Team",
            dependencies=[
                "sklearn", "numpy", "pandas", "plotly", "joblib"
            ]
        )
    
    def run(self, state: Dict, events: list, console, outdir: Path, **kwargs):
        """
        Führt Ensemble-Modell-Training und -Evaluation durch.
        """
        console.print("\n[bold cyan]Starte Ensemble-Modell-Analyse...[/bold cyan]")
        
        try:
            # Features extrahieren
            X, y, client_macs = self._extract_features_for_classification(state, events)
            
            if len(X) < 10:
                console.print("[yellow]Nicht genügend Daten für Ensemble-Training verfügbar.[/yellow]")
                return
            
            console.print(f"[green]Features extrahiert: {X.shape[0]} Samples, {X.shape[1]} Features[/green]")
            
            # Daten normalisieren
            X_scaled = StandardScaler().fit_transform(X)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            console.print(f"[green]Train-Test Split: {len(X_train)} Train, {len(X_test)} Test[/green]")
            
            # Ensemble Builder erstellen
            builder = self._EnsembleModelBuilder()
            
            # Basis-Modelle erstellen
            console.print("[cyan]Erstelle Basis-Modelle...[/cyan]")
            base_models = builder.create_base_models()
            
            # Ensemble-Modelle erstellen
            console.print("[cyan]Erstelle Ensemble-Modelle...[/cyan]")
            
            # Voting Ensembles
            hard_voting, soft_voting = builder.create_voting_ensemble(base_models)
            ensemble_models = {
                'Hard Voting': hard_voting,
                'Soft Voting': soft_voting
            }
            
            # Stacking Ensemble
            stacking = builder.create_stacking_ensemble(base_models)
            if stacking:
                ensemble_models['Stacking'] = stacking
            
            # Bagging Ensembles
            for name, base_model in list(base_models.items())[:3]:  # Nur erste 3 für Performance
                bagging = builder.create_bagging_ensemble(base_model)
                ensemble_models[f'Bagging ({name})'] = bagging
            
            # Boosting Ensembles
            boosting_models = builder.create_boosting_ensemble()
            for i, boosting in enumerate(boosting_models):
                ensemble_models[f'Boosting {i+1}'] = boosting
            
            # Alle Modelle evaluieren
            console.print("[cyan]Evaluiere alle Modelle...[/cyan]")
            performance_metrics = []
            
            # Basis-Modelle
            for name, model in base_models.items():
                console.print(f"[cyan]Evaluiere {name}...[/cyan]")
                perf = self._evaluate_model_performance(model, X_train, y_train, name)
                performance_metrics.append(perf)
            
            # Ensemble-Modelle
            for name, model in ensemble_models.items():
                console.print(f"[cyan]Evaluiere {name}...[/cyan]")
                perf = self._evaluate_model_performance(model, X_train, y_train, name)
                performance_metrics.append(perf)
            
            # Bestes Modell finden
            best_model_perf = max(performance_metrics, key=lambda x: x.accuracy)
            console.print(f"[bold green]Bestes Modell: {best_model_perf.name} "
                         f"(Accuracy: {best_model_perf.accuracy:.3f})[/bold green]")
            
            # Top 5 Modelle anzeigen
            top_models = sorted(performance_metrics, key=lambda x: x.accuracy, reverse=True)[:5]
            console.print("\n[bold]Top 5 Modelle:[/bold]")
            for i, perf in enumerate(top_models, 1):
                console.print(f"{i}. {perf.name}: {perf.accuracy:.3f} "
                             f"(CV: {perf.cross_val_mean:.3f} ± {perf.cross_val_std:.3f})")
            
            # Visualisierung erstellen
            self._create_performance_visualization(performance_metrics, outdir)
            
            # Detaillierte Evaluation des besten Modells
            console.print(f"\n[bold]Detaillierte Evaluation von {best_model_perf.name}:[/bold]")
            
            # Finde das beste Modell-Objekt
            best_model = None
            if best_model_perf.name in base_models:
                best_model = base_models[best_model_perf.name]
            elif best_model_perf.name in ensemble_models:
                best_model = ensemble_models[best_model_perf.name]
            
            test_accuracy = None
            if best_model:
                # Test-Set Evaluation
                y_pred = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred)
                console.print(f"Test Accuracy: {test_accuracy:.3f}")
                
                # Classification Report
                device_types = ['smartphone', 'laptop', 'iot_device', 'router']
                report = classification_report(y_test, y_pred, target_names=device_types, zero_division=0)
                console.print(f"\nClassification Report:\n{report}")
            
            # Ergebnisse speichern
            results = {
                'performance_metrics': [
                    {
                        'name': perf.name,
                        'accuracy': perf.accuracy,
                        'precision': perf.precision,
                        'recall': perf.recall,
                        'f1_score': perf.f1_score,
                        'cross_val_mean': perf.cross_val_mean,
                        'cross_val_std': perf.cross_val_std,
                        'training_time': perf.training_time,
                        'prediction_time': perf.prediction_time
                    }
                    for perf in performance_metrics
                ],
                'best_model': {
                    'name': best_model_perf.name,
                    'accuracy': best_model_perf.accuracy,
                    'test_accuracy': test_accuracy
                },
                'feature_importance': None  # Könnte für Tree-basierte Modelle implementiert werden
            }
            
            # Bestes Modell speichern
            if best_model:
                model_file = outdir / f"best_ensemble_model_{best_model_perf.name.lower().replace(' ', '_')}.joblib"
                joblib.dump(best_model, model_file)
                console.print(f"[green]Bestes Modell gespeichert: {model_file}[/green]")
            
            # Ergebnisse speichern
            results_file = outdir / "ensemble_analysis_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            console.print(f"[green]Ensemble-Analyse abgeschlossen. Ergebnisse gespeichert: {results_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]Fehler bei der Ensemble-Analyse: {e}[/red]")
            logger.error(f"Fehler bei der Ensemble-Analyse: {e}", exc_info=True)
    
    def _extract_features_for_classification(self, state: Dict, events: list) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Extrahiert Features für die Geräte-Klassifizierung."""
        features = []
        labels = []
        client_macs = []
        
        # Label-Mapping für Gerätetypen
        device_type_mapping = {
            'smartphone': 0,
            'laptop': 1,
            'tablet': 2,
            'iot_device': 3,
            'router': 4,
            'unknown': 5
        }
        
        for client_mac, client in state.clients.items():
            # Features extrahieren
            client_features = []
            
            # 1. Timing-Features
            if hasattr(client, 'first_seen') and hasattr(client, 'last_seen'):
                duration = client.last_seen - client.first_seen
                client_features.extend([
                    duration,
                    len(client.probe_requests) / max(duration, 1),
                ])
            else:
                client_features.extend([0, 0])
            
            # 2. Probe-Request-Features
            client_features.extend([
                len(client.probe_requests),
                len(set(client.probe_requests)),
                len([req for req in client.probe_requests if req]),
            ])
            
            # 3. RSSI-Features
            if hasattr(client, 'rssi_history') and client.rssi_history:
                rssi_values = [r for r in client.rssi_history if r is not None]
                if rssi_values:
                    client_features.extend([
                        np.mean(rssi_values),
                        np.std(rssi_values),
                        np.min(rssi_values),
                        np.max(rssi_values),
                    ])
                else:
                    client_features.extend([0, 0, 0, 0])
            else:
                client_features.extend([0, 0, 0, 0])
            
            # 4. Information Elements Features
            if hasattr(client, 'information_elements') and client.information_elements:
                client_features.append(len(client.information_elements))
            else:
                client_features.append(0)
            
            # 5. Vendor-Features
            if hasattr(client, 'vendor') and client.vendor:
                vendor_hash = hash(client.vendor) % 1000
                client_features.append(vendor_hash)
            else:
                client_features.append(0)
            
            # 6. Packet-Size-Features (simuliert)
            client_features.extend([
                np.random.normal(1000, 200),  # Durchschnittliche Paketgröße
                np.random.normal(50, 10),     # Paket-Anzahl pro Minute
            ])
            
            # 7. Connection-Pattern-Features
            client_features.extend([
                np.random.random(),  # Verbindungsstabilität
                np.random.random(),  # Roaming-Häufigkeit
            ])
            
            # Label bestimmen (vereinfachte Heuristik)
            if hasattr(client, 'vendor'):
                vendor_lower = client.vendor.lower()
                if any(term in vendor_lower for term in ['apple', 'iphone', 'ipad']):
                    device_type = 'smartphone'
                elif any(term in vendor_lower for term in ['microsoft', 'dell', 'hp', 'lenovo']):
                    device_type = 'laptop'
                elif any(term in vendor_lower for term in ['samsung', 'huawei', 'xiaomi']):
                    device_type = 'smartphone'
                elif any(term in vendor_lower for term in ['tp-link', 'netgear', 'linksys']):
                    device_type = 'router'
                else:
                    device_type = 'unknown'
            else:
                device_type = 'unknown'
            
            # Zusätzliche Heuristiken basierend auf Verhalten
            if len(client.probe_requests) > 50:
                device_type = 'smartphone'  # Smartphones scannen häufiger
            elif len(client.probe_requests) < 5:
                device_type = 'iot_device'  # IoT-Geräte scannen selten
            
            features.append(client_features)
            labels.append(device_type_mapping[device_type])
            client_macs.append(client_mac)
        
        return np.array(features), np.array(labels), client_macs
    
    def _evaluate_model_performance(self, model: Any, X: np.ndarray, y: np.ndarray, 
                                 model_name: str, cv_folds: int = 5) -> ModelPerformance:
        """Evaluiert die Performance eines Modells."""
        import time
        
        # Cross-Validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        
        # Training und Evaluation
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time
        
        # Vorhersagen
        start_time = time.time()
        y_pred = model.predict(X)
        prediction_time = time.time() - start_time
        
        # Metriken berechnen
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        return ModelPerformance(
            name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            cross_val_mean=cv_scores.mean(),
            cross_val_std=cv_scores.std(),
            training_time=training_time,
            prediction_time=prediction_time
        )
    
    def _create_performance_visualization(self, performance_metrics: List[ModelPerformance], 
                                       outdir: Path) -> None:
        """Erstellt Visualisierungen für die Modell-Performance."""
        try:
            # Überprüfe, ob Plotly verfügbar ist
            try:
                from plotly.subplots import make_subplots
            except ImportError:
                logger.warning("Plotly nicht verfügbar. Überspringe Performance-Visualisierung.")
                return
            # DataFrame für Plotly
            df = pd.DataFrame([
                {
                    'Model': perf.name,
                    'Accuracy': perf.accuracy,
                    'Precision': perf.precision,
                    'Recall': perf.recall,
                    'F1-Score': perf.f1_score,
                    'CV Mean': perf.cross_val_mean,
                    'CV Std': perf.cross_val_std,
                    'Training Time': perf.training_time,
                    'Prediction Time': perf.prediction_time
                }
                for perf in performance_metrics
            ])
            
            # Multi-Subplot Figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Comparison', 'Cross-Validation Scores', 
                              'Training Time', 'F1-Score vs Precision'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Accuracy Comparison
            fig.add_trace(
                go.Bar(x=df['Model'], y=df['Accuracy'], name='Accuracy'),
                row=1, col=1
            )
            
            # Cross-Validation mit Error Bars
            fig.add_trace(
                go.Bar(x=df['Model'], y=df['CV Mean'], 
                       error_y=dict(type='data', array=df['CV Std']),
                       name='CV Score'),
                row=1, col=2
            )
            
            # Training Time
            fig.add_trace(
                go.Bar(x=df['Model'], y=df['Training Time'], name='Training Time'),
                row=2, col=1
            )
            
            # F1-Score vs Precision Scatter
            fig.add_trace(
                go.Scatter(x=df['Precision'], y=df['F1-Score'], 
                          mode='markers+text', text=df['Model'],
                          name='F1 vs Precision'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text="Ensemble Model Performance Comparison",
                showlegend=False,
                height=800
            )
            
            # Speichern
            output_file = outdir / "ensemble_performance_comparison.html"
            fig.write_html(str(output_file))
            
            logger.info(f"Performance-Visualisierung gespeichert: {output_file}")
            
        except Exception as e:
            logger.error(f"Fehler bei der Visualisierung: {e}")
    
    class _EnsembleModelBuilder:
        """Builder-Klasse für Ensemble-Modelle."""
        
        def __init__(self, random_state: int = 42):
            self.random_state = random_state
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            self.models = {}
            self.ensemble_models = {}
            self.performance_metrics = {}
        
        def create_base_models(self) -> Dict[str, Any]:
            """Erstellt die Basis-Modelle für das Ensemble."""
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    max_depth=10,
                    min_samples_split=5
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    learning_rate=0.1,
                    max_depth=6
                ),
                'extra_trees': ExtraTreesClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    max_depth=10
                ),
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    C=1.0
                ),
                'svm': SVC(
                    random_state=self.random_state,
                    probability=True,
                    C=1.0,
                    kernel='rbf'
                ),
                'neural_network': MLPClassifier(
                    random_state=self.random_state,
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    learning_rate_init=0.001
                ),
                'naive_bayes': GaussianNB(),
                'decision_tree': DecisionTreeClassifier(
                    random_state=self.random_state,
                    max_depth=10
                )
            }
            return models
        
        def create_voting_ensemble(self, base_models: Dict[str, Any]) -> Tuple[VotingClassifier, VotingClassifier]:
            """Erstellt ein Voting-Ensemble."""
            # Hard Voting
            hard_voting = VotingClassifier(
                estimators=list(base_models.items()),
                voting='hard'
            )
            
            # Soft Voting (nur für Modelle mit predict_proba)
            soft_models = {name: model for name, model in base_models.items() 
                          if hasattr(model, 'predict_proba')}
            
            soft_voting = VotingClassifier(
                estimators=list(soft_models.items()),
                voting='soft'
            )
            
            return hard_voting, soft_voting
        
        def create_stacking_ensemble(self, base_models: Dict[str, Any]) -> Any:
            """Erstellt ein Stacking-Ensemble."""
            try:
                from sklearn.ensemble import StackingClassifier
                
                # Meta-Learner
                meta_learner = LogisticRegression(random_state=self.random_state)
                
                # Stacking Classifier
                stacking = StackingClassifier(
                    estimators=list(base_models.items()),
                    final_estimator=meta_learner,
                    cv=5,
                    stack_method='predict_proba'
                )
                
                return stacking
            except ImportError:
                logger.warning("StackingClassifier nicht verfügbar")
                return None
        
        def create_bagging_ensemble(self, base_estimator: Any) -> BaggingClassifier:
            """Erstellt ein Bagging-Ensemble."""
            return BaggingClassifier(
                estimator=base_estimator,
                n_estimators=50,
                random_state=self.random_state,
                max_samples=0.8,
                max_features=0.8
            )
        
        def create_boosting_ensemble(self) -> List[Any]:
            """Erstellt Boosting-Ensembles."""
            boosting_models = [
                AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=3),
                    n_estimators=100,
                    random_state=self.random_state,
                    learning_rate=1.0
                ),
                GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    learning_rate=0.1,
                    max_depth=6
                )
            ]
            return boosting_models