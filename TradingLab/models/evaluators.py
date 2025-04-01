# Valutazione modelli

"""
Sistema di valutazione per i modelli nel progetto TradingLab.
Questo modulo fornisce classi e funzioni per valutare le prestazioni dei modelli.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
)

# Importazioni dal modulo base
from .base import BaseModel

# Importazioni dal modulo utils
from ..utils import (
    app_logger, model_logger, ModelError, EvaluationError, 
    time_it, ensure_directory, save_json, load_json
)

# Importazioni dal modulo config
from ..config import (
    REPORTS_DIR, get_report_path
)


class ModelEvaluator:
    """
    Classe base per la valutazione dei modelli.
    """
    
    def __init__(self, model: BaseModel):
        """
        Inizializza il valutatore.
        
        Args:
            model: Modello da valutare
        """
        self.model = model
        self.evaluation_results = {}
        model_logger.info(f"Valutatore inizializzato per il modello {model.name}")
    
    @time_it
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Valuta il modello sui dati forniti.
        
        Args:
            X: Features di test
            y: Target di test
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con metriche di valutazione
        """
        try:
            # Valuta il modello
            metrics = self.model.evaluate(X, y)
            self.evaluation_results = metrics
            
            # Log delle metriche
            metrics_str = ", ".join([f"{name}={value:.4f}" for name, value in metrics.items()])
            model_logger.info(f"Valutazione completata per {self.model.name}. Metriche: {metrics_str}")
            
            return metrics
        
        except Exception as e:
            error_msg = f"Errore durante la valutazione del modello {self.model.name}: {str(e)}"
            model_logger.error(error_msg)
            raise EvaluationError(error_msg)
    
    def save_evaluation_report(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: Optional[List[str]] = None,
                              target_names: Optional[List[str]] = None,
                              filepath: Optional[str] = None,
                              include_predictions: bool = True) -> str:
        """
        Salva un report di valutazione completo.
        
        Args:
            X: Features di test
            y: Target di test
            feature_names: Nomi delle feature (opzionale)
            target_names: Nomi dei target (opzionale)
            filepath: Percorso del file (opzionale)
            include_predictions: Se includere le previsioni
            
        Returns:
            Percorso del file salvato
        """
        # Genera percorso se non fornito
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = "evaluation"
            symbol = self.model.metadata.get("symbol", "unknown")
            timeframe = self.model.metadata.get("timeframe", "unknown")
            filepath = get_report_path(report_type, symbol, timeframe, extension="json")
        
        # Assicurati che la directory esista
        ensure_directory(os.path.dirname(filepath))
        
        # Valuta il modello se non è stato fatto
        if not self.evaluation_results:
            self.evaluate(X, y)
        
        # Memorizza i nomi delle feature e target nel modello
        if feature_names is not None:
            self.model.feature_names = feature_names
        
        if target_names is not None:
            self.model.target_names = target_names
        
        # Prepara previsioni se richiesto
        predictions = None
        if include_predictions:
            y_pred = self.model.predict(X)
            
            # Usiamo al massimo 1000 sample per le previsioni nel report
            max_samples = min(1000, len(y))
            if len(y) > max_samples:
                indices = np.random.choice(len(y), max_samples, replace=False)
                y_pred_sample = y_pred[indices]
                y_true_sample = y[indices]
            else:
                y_pred_sample = y_pred
                y_true_sample = y
            
            # Converti in lista per JSON
            predictions = {
                "y_pred": y_pred_sample.tolist(),
                "y_true": y_true_sample.tolist()
            }
        
        # Prepara il report
        report = {
            "model_name": self.model.name,
            "model_version": self.model.version,
            "evaluation_date": datetime.now().isoformat(),
            "metrics": self.evaluation_results,
            "feature_names": feature_names,
            "target_names": target_names,
            "samples_count": len(y),
            "predictions": predictions
        }
        
        # Calcola metriche aggiuntive in base al tipo di problema
        if 'accuracy' in self.evaluation_results:  # Classificazione
            # Aggiungi metriche di classificazione
            report.update(self._calculate_classification_metrics(X, y))
        else:  # Regressione
            # Aggiungi metriche di regressione
            report.update(self._calculate_regression_metrics(X, y))
        
        # Salva il report
        save_json(filepath, report)
        model_logger.info(f"Report di valutazione salvato in {filepath}")
        
        return filepath
    
    def _calculate_classification_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Calcola metriche aggiuntive per problemi di classificazione.
        
        Args:
            X: Features di test
            y: Target di test
            
        Returns:
            Dizionario con metriche aggiuntive
        """
        try:
            # Effettua previsioni
            y_pred = self.model.predict(X)
            
            # Converti previsioni e target in formato adatto
            if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoding
                y_true_classes = np.argmax(y, axis=1)
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                y_true_classes = y
                y_pred_classes = (y_pred > 0.5).astype(int)
            
            # Calcola matrice di confusione
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            
            # Calcola AUC e curva ROC per classificazione binaria
            roc_data = {}
            if len(np.unique(y_true_classes)) == 2:
                fpr, tpr, thresholds = roc_curve(y_true_classes, y_pred_classes)
                roc_auc = auc(fpr, tpr)
                
                roc_data = {
                    "roc_auc": float(roc_auc),
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist()
                }
            
            # Prepara risultati
            return {
                "confusion_matrix": cm.tolist(),
                "roc_data": roc_data
            }
        
        except Exception as e:
            model_logger.warning(f"Errore nel calcolo delle metriche di classificazione: {str(e)}")
            return {}
    
    def _calculate_regression_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Calcola metriche aggiuntive per problemi di regressione.
        
        Args:
            X: Features di test
            y: Target di test
            
        Returns:
            Dizionario con metriche aggiuntive
        """
        try:
            # Effettua previsioni
            y_pred = self.model.predict(X)
            
            # Calcola errori
            errors = y_pred - y
            
            # Calcola statistiche di errore
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs(errors / y)) * 100 if np.all(y != 0) else np.nan
            
            # Prepara risultati
            return {
                "detailed_metrics": {
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mape": float(mape) if not np.isnan(mape) else None
                },
                "error_stats": {
                    "mean": float(np.mean(errors)),
                    "std": float(np.std(errors)),
                    "min": float(np.min(errors)),
                    "max": float(np.max(errors)),
                    "median": float(np.median(errors)),
                    "quantiles": {
                        "25%": float(np.percentile(errors, 25)),
                        "75%": float(np.percentile(errors, 75))
                    }
                }
            }
        
        except Exception as e:
            model_logger.warning(f"Errore nel calcolo delle metriche di regressione: {str(e)}")
            return {}


class ClassificationEvaluator(ModelEvaluator):
    """
    Valutatore specializzato per modelli di classificazione.
    """
    
    @time_it
    def evaluate_thresholds(self, X: np.ndarray, y: np.ndarray, 
                          thresholds: Optional[List[float]] = None) -> Dict[str, Dict[float, Dict[str, float]]]:
        """
        Valuta il modello con diverse soglie di decisione.
        
        Args:
            X: Features di test
            y: Target di test
            thresholds: Lista di soglie da testare (default: da 0.1 a 0.9 con passo 0.1)
            
        Returns:
            Dizionario con metriche per ogni soglia
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        model_logger.info(f"Valutazione del modello {self.model.name} con diverse soglie")
        
        try:
            # Effettua previsioni (probabilità)
            y_pred_proba = self.model.predict(X)
            
            # Converti target in formato adatto
            if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoding
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y
            
            # Valuta con diverse soglie
            threshold_results = {}
            
            for threshold in thresholds:
                # Applica la soglia
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    # Multi-classe: usa la classe con probabilità maggiore
                    y_pred = np.argmax(y_pred_proba, axis=1)
                else:
                    # Binaria: applica soglia
                    y_pred = (y_pred_proba > threshold).astype(int)
                
                # Calcola metriche
                accuracy = accuracy_score(y_true, y_pred)
                
                # Per classificazione binaria, calcola anche precision, recall, F1
                if len(np.unique(y_true)) <= 2:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    
                    threshold_results[float(threshold)] = {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1)
                    }
                else:
                    threshold_results[float(threshold)] = {
                        "accuracy": float(accuracy)
                    }
            
            # Memorizza risultati
            self.evaluation_results["threshold_analysis"] = threshold_results
            
            metrics_str = ", ".join([f"{t:.1f}: acc={m['accuracy']:.4f}" 
                                   for t, m in threshold_results.items()])
            model_logger.info(f"Valutazione con soglie completata. Risultati: {metrics_str}")
            
            return {"threshold_analysis": threshold_results}
        
        except Exception as e:
            error_msg = f"Errore durante la valutazione con soglie del modello {self.model.name}: {str(e)}"
            model_logger.error(error_msg)
            raise EvaluationError(error_msg)
    
    @time_it
    def evaluate_per_class(self, X: np.ndarray, y: np.ndarray, 
                         class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        
        """
        Valuta le prestazioni del modello per classe.
        
        Args:
            X: Features di test
            y: Target di test
            class_names: Nomi delle classi (opzionale)
            
        Returns:
            Dizionario con metriche per ogni classe
        """
        model_logger.info(f"Valutazione del modello {self.model.name} per classe")
        
        try:
            # Effettua previsioni
            y_pred = self.model.predict(X)
            
            # Converti previsioni e target in formato adatto
            if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoding
                y_true = np.argmax(y, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
                
                # Determina il numero di classi
                n_classes = y.shape[1]
            else:
                y_true = y
                y_pred = (y_pred > 0.5).astype(int)
                
                # Determina il numero di classi
                n_classes = len(np.unique(y_true))
            
            # Usa nomi delle classi forniti o crea nomi predefiniti
            if class_names is None:
                class_names = [f"Classe {i}" for i in range(n_classes)]
            
            # Calcola metriche per ogni classe
            per_class_metrics = {}
            
            for i in range(n_classes):
                # Converti il problema in binario (classe i vs resto)
                y_true_binary = (y_true == i).astype(int)
                y_pred_binary = (y_pred == i).astype(int)
                
                # Calcola metriche
                precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                
                # Calcola matrice di confusione per questa classe
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                tn, fp, fn, tp = cm.ravel()
                
                # Calcola metriche aggiuntive
                support = np.sum(y_true_binary)
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                per_class_metrics[class_names[i]] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "specificity": float(specificity),
                    "support": int(support),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn)
                }
            
            # Memorizza risultati
            self.evaluation_results["per_class_metrics"] = per_class_metrics
            
            metrics_str = ", ".join([f"{c}: f1={m['f1']:.4f}" 
                                   for c, m in per_class_metrics.items()])
            model_logger.info(f"Valutazione per classe completata. Risultati: {metrics_str}")
            
            return {"per_class_metrics": per_class_metrics}
        
        except Exception as e:
            error_msg = f"Errore durante la valutazione per classe del modello {self.model.name}: {str(e)}"
            model_logger.error(error_msg)
            raise EvaluationError(error_msg)
    
    def generate_classification_report(self, X: np.ndarray, y: np.ndarray, 
                                     filepath: Optional[str] = None) -> str:
        """
        Genera un report completo per un modello di classificazione.
        
        Args:
            X: Features di test
            y: Target di test
            filepath: Percorso del file (opzionale)
            
        Returns:
            Percorso del file del report
        """
        # Genera percorso se non fornito
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = "classification"
            symbol = self.model.metadata.get("symbol", "unknown")
            timeframe = self.model.metadata.get("timeframe", "unknown")
            filepath = get_report_path(report_type, symbol, timeframe, extension="json")
        
        # Esegui tutte le valutazioni
        basic_metrics = self.evaluate(X, y)
        threshold_metrics = self.evaluate_thresholds(X, y)
        class_metrics = self.evaluate_per_class(X, y)
        
        # Prepara il report
        report = {
            "model_name": self.model.name,
            "model_version": self.model.version,
            "evaluation_date": datetime.now().isoformat(),
            "basic_metrics": basic_metrics,
            "threshold_analysis": threshold_metrics.get("threshold_analysis", {}),
            "per_class_metrics": class_metrics.get("per_class_metrics", {})
        }
        
        # Aggiungi metriche di classificazione avanzate
        report.update(self._calculate_classification_metrics(X, y))
        
        # Salva il report
        save_json(filepath, report)
        model_logger.info(f"Report di classificazione completo salvato in {filepath}")
        
        return filepath


class RegressionEvaluator(ModelEvaluator):
    """
    Valutatore specializzato per modelli di regressione.
    """
    
    @time_it
    def evaluate_residuals(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Analizza i residui del modello.
        
        Args:
            X: Features di test
            y: Target di test
            
        Returns:
            Dizionario con analisi dei residui
        """
        model_logger.info(f"Analisi dei residui per il modello {self.model.name}")
        
        try:
            # Effettua previsioni
            y_pred = self.model.predict(X)
            
            # Calcola residui
            residuals = y - y_pred
            residuals_flat = residuals.flatten()
            
            # Calcola statistiche sui residui
            mean_residual = np.mean(residuals_flat)
            std_residual = np.std(residuals_flat)
            
            # Test di normalità dei residui (Shapiro-Wilk)
            from scipy import stats
            
            # Limita il campione a 5000 punti per il test di Shapiro-Wilk
            max_samples = min(5000, len(residuals_flat))
            sample_indices = np.random.choice(len(residuals_flat), max_samples, replace=False)
            shapiro_test = stats.shapiro(residuals_flat[sample_indices])
            
            # Test di autocorrelazione (Durbin-Watson)
            from statsmodels.stats.stattools import durbin_watson
            dw_test = durbin_watson(residuals_flat)
            
            # Prepara risultati
            residual_analysis = {
                "mean": float(mean_residual),
                "std": float(std_residual),
                "min": float(np.min(residuals_flat)),
                "max": float(np.max(residuals_flat)),
                "median": float(np.median(residuals_flat)),
                "quantiles": {
                    "25%": float(np.percentile(residuals_flat, 25)),
                    "75%": float(np.percentile(residuals_flat, 75))
                },
                "normality_test": {
                    "shapiro_statistic": float(shapiro_test[0]),
                    "shapiro_pvalue": float(shapiro_test[1]),
                    "is_normal": shapiro_test[1] > 0.05
                },
                "autocorrelation_test": {
                    "durbin_watson": float(dw_test),
                    "has_autocorrelation": dw_test < 1.5 or dw_test > 2.5
                }
            }
            
            # Memorizza risultati
            self.evaluation_results["residual_analysis"] = residual_analysis
            
            model_logger.info(f"Analisi dei residui completata. "
                            f"Mean={mean_residual:.4f}, Std={std_residual:.4f}, DW={dw_test:.4f}")
            
            return {"residual_analysis": residual_analysis}
        
        except Exception as e:
            error_msg = f"Errore durante l'analisi dei residui del modello {self.model.name}: {str(e)}"
            model_logger.error(error_msg)
            raise EvaluationError(error_msg)
    
    @time_it
    def evaluate_by_range(self, X: np.ndarray, y: np.ndarray, 
                         n_bins: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Valuta le prestazioni del modello per diversi range di valori target.
        
        Args:
            X: Features di test
            y: Target di test
            n_bins: Numero di bin per suddividere i valori target
            
        Returns:
            Dizionario con metriche per ogni range
        """
        model_logger.info(f"Valutazione del modello {self.model.name} per range di valori")
        
        try:
            # Appiattisci y per sicurezza
            y_flat = y.flatten()
            
            # Calcola i bin basati sui quantili
            bins = np.percentile(y_flat, np.linspace(0, 100, n_bins + 1))
            
            # Elimina potenziali duplicati nei bin
            bins = np.unique(bins)
            
            # Assicurati che ci siano almeno 2 bin
            if len(bins) < 2:
                raise ValueError("Numero insufficiente di bin unici")
            
            # Crea etichette per i bin
            bin_labels = [f"Bin {i+1}: [{bins[i]:.4f}, {bins[i+1]:.4f}]" 
                         for i in range(len(bins) - 1)]
            
            # Effettua previsioni
            y_pred = self.model.predict(X).flatten()
            
            # Valuta per ogni bin
            range_metrics = {}
            
            for i in range(len(bin_labels)):
                # Seleziona i dati in questo bin
                mask = (y_flat >= bins[i]) & (y_flat <= bins[i+1])
                
                # Salta se non ci sono dati in questo bin
                if np.sum(mask) == 0:
                    continue
                
                y_bin = y_flat[mask]
                y_pred_bin = y_pred[mask]
                
                # Calcola metriche
                mae = mean_absolute_error(y_bin, y_pred_bin)
                mse = mean_squared_error(y_bin, y_pred_bin)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_bin, y_pred_bin)
                
                # Calcola errore percentuale medio assoluto
                mape = np.mean(np.abs((y_bin - y_pred_bin) / y_bin)) * 100 if np.all(y_bin != 0) else np.nan
                
                range_metrics[bin_labels[i]] = {
                    "count": int(np.sum(mask)),
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "mape": float(mape) if not np.isnan(mape) else None
                }
            
            # Memorizza risultati
            self.evaluation_results["range_metrics"] = range_metrics
            
            metrics_str = ", ".join([f"{r}: RMSE={m['rmse']:.4f}" 
                                   for r, m in range_metrics.items()])
            model_logger.info(f"Valutazione per range completata. Risultati: {metrics_str}")
            
            return {"range_metrics": range_metrics}
        
        except Exception as e:
            error_msg = f"Errore durante la valutazione per range del modello {self.model.name}: {str(e)}"
            model_logger.error(error_msg)
            raise EvaluationError(error_msg)
    
    def generate_regression_report(self, X: np.ndarray, y: np.ndarray, 
                                 filepath: Optional[str] = None) -> str:
        """
        Genera un report completo per un modello di regressione.
        
        Args:
            X: Features di test
            y: Target di test
            filepath: Percorso del file (opzionale)
            
        Returns:
            Percorso del file del report
        """
        # Genera percorso se non fornito
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_type = "regression"
            symbol = self.model.metadata.get("symbol", "unknown")
            timeframe = self.model.metadata.get("timeframe", "unknown")
            filepath = get_report_path(report_type, symbol, timeframe, extension="json")
        
        # Esegui tutte le valutazioni
        basic_metrics = self.evaluate(X, y)
        residual_analysis = self.evaluate_residuals(X, y)
        range_metrics = self.evaluate_by_range(X, y)
        
        # Prepara il report
        report = {
            "model_name": self.model.name,
            "model_version": self.model.version,
            "evaluation_date": datetime.now().isoformat(),
            "basic_metrics": basic_metrics,
            "residual_analysis": residual_analysis.get("residual_analysis", {}),
            "range_metrics": range_metrics.get("range_metrics", {})
        }
        
        # Aggiungi metriche di regressione avanzate
        report.update(self._calculate_regression_metrics(X, y))
        
        # Salva il report
        save_json(filepath, report)
        model_logger.info(f"Report di regressione completo salvato in {filepath}")
        
        return filepath


class VisualizationUtils:
    """
    Utilità per visualizzare i risultati della valutazione.
    """
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None, 
                            figsize: Tuple[int, int] = (10, 8),
                            cmap: str = 'Blues') -> plt.Figure:
        """
        Visualizza una matrice di confusione.
        
        Args:
            cm: Matrice di confusione
            class_names: Nomi delle classi
            figsize: Dimensioni della figura
            cmap: Mappa colori
            
        Returns:
            Oggetto figura matplotlib
        """
        # Crea figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Visualizza la matrice
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # Imposta etichette
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]
        
        ax.set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              xticklabels=class_names, yticklabels=class_names,
              ylabel='True label',
              xlabel='Predicted label')
        
        # Ruota le etichette sull'asse x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop su tutte le celle per annotare
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
                     figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Visualizza una curva ROC.
        
        Args:
            fpr: False Positive Rate
            tpr: True Positive Rate
            roc_auc: Area Under the Curve
            figsize: Dimensioni della figura
            
        Returns:
            Oggetto figura matplotlib
        """
        # Crea figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Visualizza curva ROC
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Imposta etichette
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc="lower right")
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray, 
                                  average_precision: float,
                                  figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Visualizza una curva Precision-Recall.
        
        Args:
            precision: Precision
            recall: Recall
            average_precision: Average Precision Score
            figsize: Dimensioni della figura
            
        Returns:
            Oggetto figura matplotlib
        """
        # Crea figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Visualizza curva Precision-Recall
        ax.plot(recall, precision, lw=2, 
               label=f'Precision-Recall curve (AP = {average_precision:.3f})')
        
        # Imposta etichette
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Visualizza grafici di analisi dei residui.
        
        Args:
            y_true: Valori reali
            y_pred: Valori predetti
            figsize: Dimensioni della figura
            
        Returns:
            Oggetto figura matplotlib
        """
        # Appiattisci gli array
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calcola residui
        residuals = y_true_flat - y_pred_flat
        
        # Crea figure con subplot
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Residui vs predetti
        axs[0, 0].scatter(y_pred_flat, residuals, alpha=0.5)
        axs[0, 0].axhline(y=0, color='r', linestyle='-')
        axs[0, 0].set_xlabel('Predicted values')
        axs[0, 0].set_ylabel('Residuals')
        axs[0, 0].set_title('Residuals vs Predicted')
        
        # 2. Istogramma dei residui
        axs[0, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axs[0, 1].axvline(x=0, color='r', linestyle='-')
        axs[0, 1].set_xlabel('Residuals')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Histogram of Residuals')
        
        # 3. QQ-plot dei residui
        from scipy import stats
        stats.probplot(residuals, plot=axs[1, 0])
        axs[1, 0].set_title('Q-Q Plot of Residuals')
        
        # 4. Actual vs Predicted
        axs[1, 1].scatter(y_true_flat, y_pred_flat, alpha=0.5)
        
        # Aggiungi linea di perfetta previsione
        min_val = min(y_true_flat.min(), y_pred_flat.min())
        max_val = max(y_true_flat.max(), y_pred_flat.max())
        axs[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axs[1, 1].set_xlabel('Actual values')
        axs[1, 1].set_ylabel('Predicted values')
        axs[1, 1].set_title('Actual vs Predicted')
        
        fig.tight_layout()
        return fig


# Factory functions

def create_evaluator(model: BaseModel, model_type: str = "auto") -> ModelEvaluator:
    """
    Crea un valutatore appropriato per il modello.
    
    Args:
        model: Modello da valutare
        model_type: Tipo di modello ('classification', 'regression', 'auto')
        
    Returns:
        Istanza del valutatore appropriato
        
    Raises:
        ValueError: Se il tipo di modello non è supportato
    """
    if model_type == "auto":
        # Cerca di determinare il tipo di modello dalle metriche o dal nome
        metrics = model.metadata.get("training_metrics", {})
        
        if "accuracy" in metrics or "precision" in metrics or "recall" in metrics:
            model_type = "classification"
        elif "mse" in metrics or "mae" in metrics or "rmse" in metrics:
            model_type = "regression"
        elif "accuracy" in model.name.lower() or "class" in model.name.lower():
            model_type = "classification"
        elif "regression" in model.name.lower() or "predict" in model.name.lower():
            model_type = "regression"
        else:
            # Default a classificazione
            model_type = "classification"
    
    if model_type == "classification":
        return ClassificationEvaluator(model)
    elif model_type == "regression":
        return RegressionEvaluator(model)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")