# Classe base per modelli

"""
Classe base e interfacce per i modelli nel progetto TradingLab.
Questo modulo definisce le classi base e le interfacce per tutti i modelli predittivi.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Importazioni dal modulo utils
from ..utils import (
    app_logger, model_logger, ModelError, ModelNotFoundError, TrainingError, InferenceError,
    time_it, ensure_directory, save_json, load_json, save_pickle, load_pickle
)

# Importazioni dal modulo config
from ..config import (
    MODELS_DIR, SCALER_DIR, MODEL_FILENAME_FORMAT, get_model_path,
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_TRAIN_TEST_SPLIT, DEFAULT_VALIDATION_SPLIT
)


class BaseModel(ABC):
    """
    Classe base astratta per tutti i modelli predittivi.
    Definisce l'interfaccia comune che tutti i modelli devono implementare.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Inizializza il modello base.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        self.name = name
        self.version = version
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_names = None
        self.scaler_X = None
        self.scaler_y = None
        self.training_history = None
        self.config = {}
        self.metadata = {
            "name": name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "last_trained": None,
            "training_metrics": {},
            "validation_metrics": {},
            "features": None,
            "targets": None
        }
        model_logger.info(f"Inizializzato modello {name} (versione {version})")
    
    @abstractmethod
    def build(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Costruisce l'architettura del modello.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi specifici del modello
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello sui dati forniti.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi per l'addestramento
            
        Returns:
            Dizionario con metriche e risultati dell'addestramento
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Effettua previsioni con il modello.
        
        Args:
            X: Dati di input
            
        Returns:
            Previsioni del modello
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Valuta le prestazioni del modello su un set di dati.
        
        Args:
            X: Dati di input
            y: Target reali
            
        Returns:
            Dizionario con metriche di valutazione
        """
        pass
    
    def preprocess_X(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocessa i dati di input.
        
        Args:
            X: Dati di input
            fit: Se addestrare lo scaler sui dati
            
        Returns:
            Dati preprocessati
        """
        if X is None:
            return None
        
        # Crea lo scaler se necessario
        if self.scaler_X is None and fit:
            self.scaler_X = StandardScaler()
        
        # Applica lo scaling
        if self.scaler_X is not None:
            X_shape = X.shape
            X_flat = X.reshape(X_shape[0], -1)
            
            if fit:
                X_scaled = self.scaler_X.fit_transform(X_flat)
            else:
                X_scaled = self.scaler_X.transform(X_flat)
            
            return X_scaled.reshape(X_shape)
        
        return X
    
    def preprocess_y(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocessa i target.
        
        Args:
            y: Dati target
            fit: Se addestrare lo scaler sui dati
            
        Returns:
            Target preprocessati
        """
        if y is None:
            return None
        
        # Crea lo scaler se necessario
        if self.scaler_y is None and fit:
            self.scaler_y = StandardScaler()
        
        # Applica lo scaling
        if self.scaler_y is not None:
            y_shape = y.shape
            y_flat = y.reshape(y_shape[0], -1)
            
            if fit:
                y_scaled = self.scaler_y.fit_transform(y_flat)
            else:
                y_scaled = self.scaler_y.transform(y_flat)
            
            return y_scaled.reshape(y_shape)
        
        return y
    
    def postprocess_y(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Riconverte le previsioni nella scala originale.
        
        Args:
            y_pred: Previsioni nella scala del modello
            
        Returns:
            Previsioni nella scala originale
        """
        if y_pred is None:
            return None
        
        # Riconverti le previsioni se è stato usato uno scaler
        if self.scaler_y is not None:
            y_shape = y_pred.shape
            y_flat = y_pred.reshape(y_shape[0], -1)
            y_orig = self.scaler_y.inverse_transform(y_flat)
            return y_orig.reshape(y_shape)
        
        return y_pred
    
    def save(self, filepath: Optional[str] = None, save_config: bool = True, 
            custom_objects: Optional[Dict[str, Any]] = None) -> str:
        """
        Salva il modello su disco.
        
        Args:
            filepath: Percorso del file (opzionale)
            save_config: Se salvare anche la configurazione
            custom_objects: Oggetti personalizzati per il caricamento
            
        Returns:
            Percorso del file salvato
            
        Raises:
            ModelError: Se il salvataggio fallisce
        """
        # Genera percorso se non fornito
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MODELS_DIR, f"{self.name}_{timestamp}.h5")
        
        # Assicurati che la directory esista
        ensure_directory(os.path.dirname(filepath))
        
        try:
            # Salva modello specifico (implementazione nelle sottoclassi)
            self._save_model_impl(filepath)
            
            if save_config:
                # Salva configurazione e metadata
                config_path = filepath.replace(".h5", ".json")
                
                # Aggiungi percorsi degli scaler al metadata
                self.metadata["scaler_X_path"] = filepath.replace(".h5", "_scaler_X.pkl") if self.scaler_X else None
                self.metadata["scaler_y_path"] = filepath.replace(".h5", "_scaler_y.pkl") if self.scaler_y else None
                self.metadata["features"] = self.feature_names
                self.metadata["targets"] = self.target_names
                
                save_json(config_path, {
                    "config": self.config,
                    "metadata": self.metadata
                })
                
                # Salva gli scaler
                if self.scaler_X is not None:
                    save_pickle(filepath.replace(".h5", "_scaler_X.pkl"), self.scaler_X)
                
                if self.scaler_y is not None:
                    save_pickle(filepath.replace(".h5", "_scaler_y.pkl"), self.scaler_y)
            
            model_logger.info(f"Modello {self.name} salvato in {filepath}")
            return filepath
        
        except Exception as e:
            error_msg = f"Errore durante il salvataggio del modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise ModelError(error_msg)
    
    @abstractmethod
    def _save_model_impl(self, filepath: str) -> None:
        """
        Implementazione specifica per il salvataggio del modello.
        Da sovrascrivere nelle sottoclassi.
        
        Args:
            filepath: Percorso del file
        """
        pass
    
    @classmethod
    def load(cls, filepath: str, custom_objects: Optional[Dict[str, Any]] = None) -> 'BaseModel':
        """
        Carica un modello da disco.
        
        Args:
            filepath: Percorso del file
            custom_objects: Oggetti personalizzati per il caricamento
            
        Returns:
            Istanza del modello caricato
            
        Raises:
            ModelNotFoundError: Se il file non esiste
            ModelError: Se il caricamento fallisce
        """
        if not os.path.exists(filepath):
            raise ModelNotFoundError(f"File del modello non trovato: {filepath}")
        
        try:
            # Carica configurazione e metadata
            config_path = filepath.replace(".h5", ".json")
            if os.path.exists(config_path):
                config_data = load_json(config_path)
                config = config_data.get("config", {})
                metadata = config_data.get("metadata", {})
            else:
                config = {}
                metadata = {}
            
            # Crea un'istanza del modello
            model_instance = cls(
                name=metadata.get("name", "unknown"),
                version=metadata.get("version", "1.0.0")
            )
            model_instance.config = config
            model_instance.metadata = metadata
            
            # Carica scaler
            scaler_X_path = metadata.get("scaler_X_path")
            if scaler_X_path and os.path.exists(scaler_X_path):
                model_instance.scaler_X = load_pickle(scaler_X_path)
            
            scaler_y_path = metadata.get("scaler_y_path")
            if scaler_y_path and os.path.exists(scaler_y_path):
                model_instance.scaler_y = load_pickle(scaler_y_path)
            
            # Carica nomi delle feature e target
            model_instance.feature_names = metadata.get("features")
            model_instance.target_names = metadata.get("targets")
            
            # Carica il modello specifico
            model_instance._load_model_impl(filepath, custom_objects)
            model_instance.is_trained = True
            
            model_logger.info(f"Modello {model_instance.name} caricato da {filepath}")
            return model_instance
        
        except Exception as e:
            error_msg = f"Errore durante il caricamento del modello da {filepath}: {str(e)}"
            model_logger.error(error_msg)
            raise ModelError(error_msg)
    
    @abstractmethod
    def _load_model_impl(self, filepath: str, custom_objects: Optional[Dict[str, Any]] = None) -> None:
        """
        Implementazione specifica per il caricamento del modello.
        Da sovrascrivere nelle sottoclassi.
        
        Args:
            filepath: Percorso del file
            custom_objects: Oggetti personalizzati per il caricamento
        """
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Ottiene un riepilogo del modello.
        
        Returns:
            Dizionario con informazioni sul modello
        """
        summary = {
            "name": self.name,
            "version": self.version,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names) if self.feature_names else None,
            "target_count": len(self.target_names) if self.target_names else None,
            "created_at": self.metadata.get("created_at"),
            "last_trained": self.metadata.get("last_trained"),
            "metrics": self.metadata.get("validation_metrics", {})
        }
        
        return summary
    
    def reset(self) -> None:
        """Resetta il modello, rimuovendo tutti i pesi addestrati."""
        self.is_trained = False
        self.training_history = None
        self.metadata["last_trained"] = None
        self.metadata["training_metrics"] = {}
        self.metadata["validation_metrics"] = {}
        self._reset_model_impl()
        model_logger.info(f"Modello {self.name} resettato")
    
    @abstractmethod
    def _reset_model_impl(self) -> None:
        """
        Implementazione specifica per il reset del modello.
        Da sovrascrivere nelle sottoclassi.
        """
        pass
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Aggiorna i metadata del modello.
        
        Args:
            key: Chiave da aggiornare
            value: Nuovo valore
        """
        self.metadata[key] = value


class SklearnModel(BaseModel):
    """
    Wrapper per modelli scikit-learn.
    """
    
    def __init__(self, sklearn_model: Any, name: str, version: str = "1.0.0"):
        """
        Inizializza un wrapper per modelli scikit-learn.
        
        Args:
            sklearn_model: Istanza del modello scikit-learn
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        self.model = sklearn_model
        self.config = {
            "model_type": "sklearn",
            "sklearn_class": sklearn_model.__class__.__name__,
            "parameters": getattr(sklearn_model, 'get_params', lambda: {})()
        }
    
    def build(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione vuota per compatibilità.
        I modelli scikit-learn non richiedono una build esplicita.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi
        """
        pass  # Nessuna azione necessaria
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello scikit-learn.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (ignorato in sklearn)
            y_val: Target di validazione (ignorato in sklearn)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con metriche e risultati dell'addestramento
            
        Raises:
            TrainingError: Se l'addestramento fallisce
        """
        try:
            # Preprocessa i dati
            X_train_proc = self.preprocess_X(X_train, fit=True)
            y_train_proc = self.preprocess_y(y_train, fit=True)
            
            # Memorizza i nomi delle feature e target se forniti
            feature_names = kwargs.get("feature_names")
            target_names = kwargs.get("target_names")
            
            if feature_names is not None:
                self.feature_names = feature_names
            
            if target_names is not None:
                self.target_names = target_names
            
            # Addestra il modello
            start_time = datetime.now()
            self.model.fit(X_train_proc, y_train_proc, **kwargs)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Aggiorna i metadati
            self.is_trained = True
            self.metadata["last_trained"] = datetime.now().isoformat()
            
            # Valuta il modello sul training set
            train_metrics = self.evaluate(X_train, y_train)
            self.metadata["training_metrics"] = train_metrics
            
            # Valuta su validation set se disponibile
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                self.metadata["validation_metrics"] = val_metrics
            
            # Preparazione risultati
            results = {
                "training_time": training_time,
                "training_metrics": train_metrics,
                "validation_metrics": self.metadata["validation_metrics"],
                "model_type": "sklearn",
                "model_class": self.model.__class__.__name__
            }
            
            model_logger.info(f"Modello {self.name} addestrato in {training_time:.2f} secondi")
            return results
        
        except Exception as e:
            error_msg = f"Errore durante l'addestramento del modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise TrainingError(error_msg)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Effettua previsioni con il modello scikit-learn.
        
        Args:
            X: Dati di input
            
        Returns:
            Previsioni del modello
            
        Raises:
            InferenceError: Se la previsione fallisce
        """
        if not self.is_trained:
            raise InferenceError(f"Modello {self.name} non addestrato")
        
        try:
            # Preprocessa l'input
            X_proc = self.preprocess_X(X)
            
            # Effettua la previsione
            if hasattr(self.model, 'predict_proba'):
                y_pred = self.model.predict_proba(X_proc)
            else:
                y_pred = self.model.predict(X_proc)
            
            # Riconverti nella scala originale
            y_pred = self.postprocess_y(y_pred)
            
            return y_pred
        
        except Exception as e:
            error_msg = f"Errore durante la previsione con il modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Valuta le prestazioni del modello scikit-learn.
        
        Args:
            X: Dati di input
            y: Target reali
            
        Returns:
            Dizionario con metriche di valutazione
            
        Raises:
            InferenceError: Se la valutazione fallisce
        """
        if not self.is_trained:
            raise InferenceError(f"Modello {self.name} non addestrato")
        
        try:
            # Effettua la previsione
            y_pred = self.predict(X)
            
            # Calcola le metriche
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
            
            metrics = {}
            
            # Determina se classificazione o regressione
            if hasattr(self.model, 'predict_proba'):
                # Classificazione
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # Multi-classe: prendi l'indice della classe con probabilità massima
                    y_pred_class = np.argmax(y_pred, axis=1)
                    y_true_class = np.argmax(y, axis=1)
                else:
                    # Binaria: arrotonda a 0/1
                    y_pred_class = np.round(y_pred).astype(int)
                    y_true_class = y
                
                metrics["accuracy"] = float(accuracy_score(y_true_class, y_pred_class))
                
                # Calcola precision, recall, F1 solo per problemi binari
                if len(np.unique(y_true_class)) <= 2:
                    metrics["precision"] = float(precision_score(y_true_class, y_pred_class, zero_division=0))
                    metrics["recall"] = float(recall_score(y_true_class, y_pred_class, zero_division=0))
                    metrics["f1"] = float(f1_score(y_true_class, y_pred_class, zero_division=0))
            else:
                # Regressione
                metrics["mse"] = float(mean_squared_error(y, y_pred))
                metrics["rmse"] = float(np.sqrt(metrics["mse"]))
                metrics["r2"] = float(r2_score(y, y_pred))
            
            return metrics
        
        except Exception as e:
            error_msg = f"Errore durante la valutazione del modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def _save_model_impl(self, filepath: str) -> None:
        """
        Implementazione del salvataggio per modelli scikit-learn.
        
        Args:
            filepath: Percorso del file
        """
        import joblib
        model_path = filepath.replace(".h5", ".joblib")
        joblib.dump(self.model, model_path)
    
    def _load_model_impl(self, filepath: str, custom_objects: Optional[Dict[str, Any]] = None) -> None:
        """
        Implementazione del caricamento per modelli scikit-learn.
        
        Args:
            filepath: Percorso del file
            custom_objects: Ignorato per scikit-learn
        """
        import joblib
        model_path = filepath.replace(".h5", ".joblib")
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"File del modello scikit-learn non trovato: {model_path}")
        
        self.model = joblib.load(model_path)
    
    def _reset_model_impl(self) -> None:
        """
        Implementazione del reset per modelli scikit-learn.
        """
        # Crea una nuova istanza con gli stessi parametri
        from sklearn.base import clone
        if self.model is not None:
            self.model = clone(self.model)


class TensorFlowModel(BaseModel):
    """
    Wrapper per modelli TensorFlow/Keras.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Inizializza un wrapper per modelli TensorFlow/Keras.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        self.config = {
            "model_type": "tensorflow",
            "parameters": {}
        }
    
    def build(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Costruisce l'architettura del modello TensorFlow/Keras.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi
            
        Raises:
            ModelError: Se la build fallisce
        """
        try:
            # Memorizza parametri di build
            self.config["parameters"].update({
                "input_shape": input_shape,
                "output_shape": output_shape
            })
            self.config["parameters"].update(kwargs)
            
            # Costruisci il modello specifico
            self._build_model_impl(input_shape, output_shape, **kwargs)
            
            model_logger.info(f"Modello {self.name} costruito con input_shape={input_shape}, output_shape={output_shape}")
        
        except Exception as e:
            error_msg = f"Errore durante la costruzione del modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise ModelError(error_msg)
    
    @abstractmethod
    def _build_model_impl(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione specifica per la costruzione del modello.
        Da sovrascrivere nelle sottoclassi.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi
        """
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello TensorFlow/Keras.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con metriche e risultati dell'addestramento
            
        Raises:
            TrainingError: Se l'addestramento fallisce
        """
        if self.model is None:
            raise TrainingError(f"Modello {self.name} non costruito")
        
        try:
            # Preprocessa i dati
            X_train_proc = self.preprocess_X(X_train, fit=True)
            y_train_proc = self.preprocess_y(y_train, fit=True)
            
            X_val_proc = None
            y_val_proc = None
            
            if X_val is not None and y_val is not None:
                X_val_proc = self.preprocess_X(X_val)
                y_val_proc = self.preprocess_y(y_val)
            
            # Memorizza i nomi delle feature e target se forniti
            feature_names = kwargs.pop("feature_names", None)
            target_names = kwargs.pop("target_names", None)
            
            if feature_names is not None:
                self.feature_names = feature_names
            
            if target_names is not None:
                self.target_names = target_names
            
            # Parametri di addestramento predefiniti
            epochs = kwargs.pop("epochs", DEFAULT_EPOCHS)
            batch_size = kwargs.pop("batch_size", DEFAULT_BATCH_SIZE)
            validation_split = kwargs.pop("validation_split", DEFAULT_VALIDATION_SPLIT if X_val is None else 0)
            
            # Prepara i dati di validazione
            validation_data = None
            if X_val_proc is not None and y_val_proc is not None:
                validation_data = (X_val_proc, y_val_proc)
            
            # Addestra il modello
            start_time = datetime.now()
            history = self.model.fit(
                X_train_proc, y_train_proc,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                validation_data=validation_data,
                **kwargs
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Memorizza la storia dell'addestramento
            self.training_history = history.history
            
            # Aggiorna i metadati
            self.is_trained = True
            self.metadata["last_trained"] = datetime.now().isoformat()
            
            # Ottieni metriche finali
            train_metrics = {}
            for metric_name, values in history.history.items():
                if not metric_name.startswith('val_'):
                    train_metrics[metric_name] = float(values[-1])
            
            val_metrics = {}
            for metric_name, values in history.history.items():
                if metric_name.startswith('val_'):
                    val_metrics[metric_name[4:]] = float(values[-1])
            
            self.metadata["training_metrics"] = train_metrics
            self.metadata["validation_metrics"] = val_metrics
            
            # Preparazione risultati
            results = {
                "training_time": training_time,
                "epochs": epochs,
                "training_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "history": history.history
            }
            
            model_logger.info(f"Modello {self.name} addestrato in {training_time:.2f} secondi")
            return results
        
        except Exception as e:
            error_msg = f"Errore durante l'addestramento del modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise TrainingError(error_msg)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Effettua previsioni con il modello TensorFlow/Keras.
        
        Args:
            X: Dati di input
            
        Returns:
            Previsioni del modello
            
        Raises:
            InferenceError: Se la previsione fallisce
        """
        if not self.is_trained:
            raise InferenceError(f"Modello {self.name} non addestrato")
        
        try:
            # Preprocessa l'input
            X_proc = self.preprocess_X(X)
            
            # Effettua la previsione
            y_pred = self.model.predict(X_proc)
            
            # Riconverti nella scala originale
            y_pred = self.postprocess_y(y_pred)
            
            return y_pred
        
        except Exception as e:
            error_msg = f"Errore durante la previsione con il modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Valuta le prestazioni del modello TensorFlow/Keras.
        
        Args:
            X: Dati di input
            y: Target reali
            
        Returns:
            Dizionario con metriche di valutazione
            
        Raises:
            InferenceError: Se la valutazione fallisce
        """
        if not self.is_trained:
            raise InferenceError(f"Modello {self.name} non addestrato")
        
        try:
            # Preprocessa i dati
            X_proc = self.preprocess_X(X)
            y_proc = self.preprocess_y(y)
            
            # Valuta il modello
            metrics = self.model.evaluate(X_proc, y_proc, verbose=0)
            
            # Converti in dizionario
            metric_names = self.model.metrics_names
            metrics_dict = {name: float(value) for name, value in zip(metric_names, metrics)}
            
            # Per modelli di classificazione, aggiungi metriche aggiuntive
            if 'accuracy' in metrics_dict:  # Classificazione
                # Effettua previsioni
                y_pred = self.predict(X)
                
                # Converti in classi discrete
                if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoding
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    y_true_classes = np.argmax(y, axis=1)
                else:
                    y_pred_classes = np.round(y_pred).astype(int)
                    y_true_classes = y
                
                # Calcola metriche di classificazione
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                # Solo per classificazione binaria
                if len(np.unique(y_true_classes)) <= 2:
                    metrics_dict['precision'] = float(precision_score(y_true_classes, y_pred_classes, zero_division=0))
                    metrics_dict['recall'] = float(recall_score(y_true_classes, y_pred_classes, zero_division=0))
                    metrics_dict['f1'] = float(f1_score(y_true_classes, y_pred_classes, zero_division=0))
            
            return metrics_dict
        
        except Exception as e:
            error_msg = f"Errore durante la valutazione del modello {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def _save_model_impl(self, filepath: str) -> None:
        """
        Implementazione del salvataggio per modelli TensorFlow/Keras.
        
        Args:
            filepath: Percorso del file
        """
        self.model.save(filepath, save_format='h5')
    
    def _load_model_impl(self, filepath: str, custom_objects: Optional[Dict[str, Any]] = None) -> None:
        """
        Implementazione del caricamento per modelli TensorFlow/Keras.
        
        Args:
            filepath: Percorso del file
            custom_objects: Oggetti personalizzati per il caricamento
        """
        self.model = tf.keras.models.load_model(filepath, custom_objects=custom_objects)
    
    def _reset_model_impl(self) -> None:
        """
        Implementazione del reset per modelli TensorFlow/Keras.
        """
        # Ricrea il modello
        if self.model is not None:
            input_shape = self.config["parameters"].get("input_shape")
            output_shape = self.config["parameters"].get("output_shape")
            
            if input_shape and output_shape:
                kwargs = {k: v for k, v in self.config["parameters"].items() 
                         if k not in ["input_shape", "output_shape"]}
                
                self._build_model_impl(input_shape, output_shape, **kwargs)


class ModelRegistry:
    """
    Registro dei modelli disponibili e loro metadata.
    """
    
    def __init__(self):
        """Inizializza il registro dei modelli."""
        self.models = {}
        self.metadata_cache = {}
        model_logger.info("Registro dei modelli inizializzato")
    
    def register_model(self, model_class: type, model_type: str, description: str) -> None:
        """
        Registra un nuovo tipo di modello.
        
        Args:
            model_class: Classe del modello
            model_type: Identificatore del tipo di modello
            description: Descrizione del modello
        """
        self.models[model_type] = {
            "class": model_class,
            "description": description
        }
        model_logger.info(f"Modello {model_type} registrato")
    
    def create_model(self, model_type: str, name: str, version: str = "1.0.0", **kwargs) -> BaseModel:
        """
        Crea una nuova istanza di un modello.
        
        Args:
            model_type: Tipo di modello da creare
            name: Nome del modello
            version: Versione del modello
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Istanza del modello
            
        Raises:
            ModelError: Se il tipo di modello non è registrato
        """
        if model_type not in self.models:
            raise ModelError(f"Tipo di modello non registrato: {model_type}")
        
        model_class = self.models[model_type]["class"]
        model = model_class(name=name, version=version, **kwargs)
        
        model_logger.info(f"Modello {name} (tipo {model_type}) creato")
        return model
    
    def get_available_models(self) -> Dict[str, str]:
        """
        Ottiene i tipi di modelli disponibili.
        
        Returns:
            Dizionario {model_type: description}
        """
        return {model_type: info["description"] 
                for model_type, info in self.models.items()}
    
    def scan_models_directory(self, directory: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Scansiona una directory per trovare modelli salvati.
        
        Args:
            directory: Directory da scansionare (default: MODELS_DIR)
            
        Returns:
            Dizionario con metadata dei modelli trovati
        """
        directory = directory or MODELS_DIR
        ensure_directory(directory)
        
        model_files = {}
        
        # Cerca file dei modelli
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.h5') or file.endswith('.keras'):
                    filepath = os.path.join(root, file)
                    
                    # Carica metadata
                    try:
                        config_path = filepath.replace('.h5', '.json').replace('.keras', '.json')
                        if os.path.exists(config_path):
                            config_data = load_json(config_path)
                            metadata = config_data.get("metadata", {})
                            
                            model_id = metadata.get("name", "unknown")
                            if model_id in model_files:
                                model_id = f"{model_id}_{os.path.basename(filepath)}"
                            
                            model_files[model_id] = {
                                "filepath": filepath,
                                "metadata": metadata
                            }
                    except Exception as e:
                        model_logger.warning(f"Errore nel caricamento dei metadata da {filepath}: {str(e)}")
        
        self.metadata_cache = model_files
        model_logger.info(f"Trovati {len(model_files)} modelli salvati in {directory}")
        
        return model_files
    
    def load_model_by_id(self, model_id: str) -> Optional[BaseModel]:
        """
        Carica un modello dal suo ID.
        
        Args:
            model_id: ID del modello
            
        Returns:
            Istanza del modello caricato o None se non trovato
        """
        # Scansiona i modelli se la cache è vuota
        if not self.metadata_cache:
            self.scan_models_directory()
        
        # Cerca il modello nella cache
        if model_id not in self.metadata_cache:
            model_logger.warning(f"Modello {model_id} non trovato nella cache")
            return None
        
        model_info = self.metadata_cache[model_id]
        filepath = model_info["filepath"]
        
        # Determina il tipo di modello
        config_path = filepath.replace('.h5', '.json').replace('.keras', '.json')
        config_data = load_json(config_path)
        config = config_data.get("config", {})
        
        model_type = config.get("model_type", "tensorflow")
        
        # Carica il modello appropriato
        try:
            if model_type == "sklearn":
                from .trainers import create_sklearn_model
                return create_sklearn_model(filepath=filepath)
            elif model_type == "tensorflow":
                from .trainers import load_keras_model
                return load_keras_model(filepath=filepath)
            else:
                model_logger.warning(f"Tipo di modello non supportato: {model_type}")
                return None
        
        except Exception as e:
            model_logger.error(f"Errore nel caricamento del modello {model_id}: {str(e)}")
            return None


# Inizializza il registro dei modelli
model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """
    Ottiene l'istanza del registro dei modelli.
    
    Returns:
        Istanza del registro dei modelli
    """
    return model_registry