# Sistema di addestramento

"""
Sistema di addestramento per i modelli nel progetto TradingLab.
Questo modulo fornisce classi e funzioni per addestrare modelli e preparare dataset.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, Type
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import tensorflow as tf

# Importazioni dal modulo base
from .base import BaseModel, SklearnModel, TensorFlowModel, model_registry

# Importazioni dal modulo utils
from ..utils import (
    app_logger, model_logger, ModelError, TrainingError, 
    time_it, ensure_directory, save_json, load_json
)

# Importazioni dal modulo config
from ..config import (
    DEFAULT_TRAIN_TEST_SPLIT, DEFAULT_VALIDATION_SPLIT, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS,
    MODELS_DIR, get_model_path
)


class SequenceGenerator:
    """
    Generatore di sequenze per modelli che richiedono dati sequenziali.
    Converte dati tabellari in sequenze con una finestra temporale scorrevole.
    """
    
    def __init__(self, window_size: int, forecast_horizon: int = 1, stride: int = 1):
        """
        Inizializza il generatore di sequenze.
        
        Args:
            window_size: Dimensione della finestra di lookback
            forecast_horizon: Orizzonte di previsione (quanti step nel futuro prevedere)
            stride: Passo per la finestra scorrevole
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        model_logger.info(f"Generatore di sequenze inizializzato con window_size={window_size}, "
                        f"forecast_horizon={forecast_horizon}, stride={stride}")
    
    def create_sequences(self, df: pd.DataFrame, feature_columns: List[str], 
                        target_columns: Optional[List[str]] = None,
                        date_column: Optional[str] = 'timestamp') -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea sequenze da un DataFrame.
        
        Args:
            df: DataFrame di input
            feature_columns: Colonne da usare come features
            target_columns: Colonne da usare come target (default: stesse feature_columns)
            date_column: Nome della colonna con date/timestamp (opzionale)
            
        Returns:
            Tuple con (X_sequences, y_targets)
            
        Raises:
            ValidationError: Se i dati di input non sono validi
        """
        if df is None or df.empty:
            raise ValueError("Il DataFrame è vuoto o None")
        
        # Verifica che le colonne richieste siano presenti
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colonne mancanti nel DataFrame: {missing_columns}")
        
        # Se target_columns non è specificato, usa le stesse feature_columns
        if target_columns is None:
            target_columns = feature_columns
        else:
            missing_targets = [col for col in target_columns if col not in df.columns]
            if missing_targets:
                raise ValueError(f"Colonne target mancanti nel DataFrame: {missing_targets}")
        
        # Estrai i dati di feature e target
        data = df[feature_columns].values
        targets = df[target_columns].values
        
        # Crea sequenze
        X_sequences = []
        y_targets = []
        
        # Conserva le date/timestamp se richiesto
        dates = None
        if date_column and date_column in df.columns:
            dates = df[date_column].values
        
        # Crea sequenze con finestra scorrevole
        for i in range(0, len(data) - self.window_size - self.forecast_horizon + 1, self.stride):
            if i + self.window_size + self.forecast_horizon <= len(data):
                # Sequenza di input
                X_sequences.append(data[i:i+self.window_size])
                
                # Target
                if self.forecast_horizon == 1:
                    y_targets.append(targets[i+self.window_size])
                else:
                    y_targets.append(targets[i+self.window_size:i+self.window_size+self.forecast_horizon])
        
        # Converti in array numpy
        X_sequences = np.array(X_sequences)
        y_targets = np.array(y_targets)
        
        model_logger.info(f"Create {len(X_sequences)} sequenze di forma {X_sequences.shape}")
        return X_sequences, y_targets
    
    def get_feature_importance(self, model: BaseModel, feature_names: List[str]) -> Dict[str, float]:
        """
        Calcola l'importanza delle feature per un modello.
        
        Args:
            model: Modello addestrato
            feature_names: Nomi delle feature
            
        Returns:
            Dizionario con importanza delle feature
            
        Note:
            Funziona solo con alcuni tipi di modelli (es. Random Forest).
        """
        # Verifica se il modello ha l'attributo feature_importances_
        if hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            
            # Replica i nomi delle feature per tutte le timestep
            expanded_feature_names = []
            for t in range(self.window_size):
                for name in feature_names:
                    expanded_feature_names.append(f"{name}_t-{self.window_size-t}")
            
            # Combina i nomi con le importanze
            if len(expanded_feature_names) == len(importances):
                return {name: float(imp) for name, imp in zip(expanded_feature_names, importances)}
            
            # Fallback se le dimensioni non corrispondono
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
        
        return {}


class DatasetSplitter:
    """
    Classe per la divisione dei dataset in training, validation e test set.
    """
    
    @staticmethod
    def train_validation_test_split(X: np.ndarray, y: np.ndarray, 
                                   test_size: float = DEFAULT_TRAIN_TEST_SPLIT,
                                   validation_size: float = DEFAULT_VALIDATION_SPLIT,
                                   shuffle: bool = True, random_state: int = 42) -> Tuple:
        """
        Divide i dati in training, validation e test set.
        
        Args:
            X: Features
            y: Target
            test_size: Frazione dei dati per il test set
            validation_size: Frazione dei dati di training per la validazione
            shuffle: Se mescolare i dati
            random_state: Seed per il generatore di numeri casuali
            
        Returns:
            Tuple con (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if X is None or y is None:
            raise ValueError("X e y non possono essere None")
        
        # Prima divisione: separazione del test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        
        # Seconda divisione: separazione del validation set dal training set
        # validation_size è relativo alla dimensione di X_train_val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=validation_size, 
            shuffle=shuffle, 
            random_state=random_state
        )
        
        model_logger.info(f"Dataset diviso in: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} sample")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def time_series_split(X: np.ndarray, y: np.ndarray, 
                         test_size: float = DEFAULT_TRAIN_TEST_SPLIT,
                         validation_size: float = DEFAULT_VALIDATION_SPLIT) -> Tuple:
        """
        Divide i dati in training, validation e test set mantenendo l'ordine temporale.
        
        Args:
            X: Features
            y: Target
            test_size: Frazione dei dati per il test set
            validation_size: Frazione dei dati di training per la validazione
            
        Returns:
            Tuple con (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if X is None or y is None:
            raise ValueError("X e y non possono essere None")
        
        n_samples = len(X)
        
        # Calcola gli indici di separazione
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))
        
        # Divisione temporale senza shuffle
        X_train = X[:val_idx]
        y_train = y[:val_idx]
        
        X_val = X[val_idx:test_idx]
        y_val = y[val_idx:test_idx]
        
        X_test = X[test_idx:]
        y_test = y[test_idx:]
        
        model_logger.info(f"Dataset diviso temporalmente in: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} sample")
        return X_train, X_val, X_test, y_train, y_val, y_test


class ModelTrainer:
    """
    Classe base per l'addestramento di modelli.
    """
    
    def __init__(self, model: BaseModel):
        """
        Inizializza il trainer.
        
        Args:
            model: Modello da addestrare
        """
        self.model = model
        self.training_history = None
        model_logger.info(f"Trainer inizializzato per il modello {model.name}")
    
    @time_it
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con risultati dell'addestramento
        """
        model_logger.info(f"Inizio addestramento del modello {self.model.name}")
        
        # Addestra il modello
        results = self.model.train(X_train, y_train, X_val, y_val, **kwargs)
        self.training_history = results
        
        # Log delle metriche principali
        metrics_log = []
        for name, value in results.get("training_metrics", {}).items():
            metrics_log.append(f"{name}={value:.4f}")
        
        if results.get("validation_metrics"):
            for name, value in results.get("validation_metrics", {}).items():
                metrics_log.append(f"val_{name}={value:.4f}")
        
        metrics_str = ", ".join(metrics_log)
        model_logger.info(f"Addestramento completato per {self.model.name}. Metriche: {metrics_str}")
        
        return results
    
    @time_it
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Valuta il modello su un test set.
        
        Args:
            X_test: Features di test
            y_test: Target di test
            
        Returns:
            Dizionario con metriche di valutazione
        """
        model_logger.info(f"Valutazione del modello {self.model.name} su {len(X_test)} sample")
        
        # Valuta il modello
        metrics = self.model.evaluate(X_test, y_test)
        
        # Log delle metriche
        metrics_str = ", ".join([f"{name}={value:.4f}" for name, value in metrics.items()])
        model_logger.info(f"Valutazione completata per {self.model.name}. Metriche: {metrics_str}")
        
        return metrics
    
    def save_training_report(self, filepath: Optional[str] = None) -> str:
        """
        Salva un report dell'addestramento.
        
        Args:
            filepath: Percorso del file (opzionale)
            
        Returns:
            Percorso del file salvato
        """
        if self.training_history is None:
            model_logger.warning(f"Nessuna storia di addestramento disponibile per il modello {self.model.name}")
            return ""
        
        # Genera percorso se non fornito
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(MODELS_DIR, f"{self.model.name}_training_report_{timestamp}.json")
        
        # Assicurati che la directory esista
        ensure_directory(os.path.dirname(filepath))
        
        # Prepara il report
        report = {
            "model_name": self.model.name,
            "model_version": self.model.version,
            "training_date": datetime.now().isoformat(),
            "training_metrics": self.model.metadata.get("training_metrics", {}),
            "validation_metrics": self.model.metadata.get("validation_metrics", {}),
            "training_history": {
                k: (v if isinstance(v, (int, float, str, bool)) else str(v)) 
                for k, v in self.training_history.items()
            }
        }
        
        # Salva il report
        save_json(filepath, report)
        model_logger.info(f"Report di addestramento salvato in {filepath}")
        
        return filepath


class SklearnTrainer(ModelTrainer):
    """
    Trainer specializzato per modelli scikit-learn.
    """
    
    @time_it
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            param_grid: Dict[str, List[Any]], cv: int = 5,
                            scoring: Optional[str] = None, n_jobs: int = -1) -> Dict[str, Any]:
        """
        Esegue l'ottimizzazione degli iperparametri con GridSearchCV.
        
        Args:
            X: Features
            y: Target
            param_grid: Griglia di parametri da testare
            cv: Numero di fold per la cross-validation
            scoring: Metrica di scoring (default: accuracy per classificazione, r2 per regressione)
            n_jobs: Numero di job paralleli
            
        Returns:
            Dizionario con risultati dell'ottimizzazione
        """
        if not isinstance(self.model, SklearnModel):
            raise TypeError("Il modello deve essere un'istanza di SklearnModel")
        
        model_logger.info(f"Inizio ottimizzazione iperparametri per {self.model.name}")
        
        # Preprocessa i dati
        X_proc = self.model.preprocess_X(X, fit=True)
        y_proc = self.model.preprocess_y(y, fit=True)
        
        # Configura GridSearchCV
        grid_search = GridSearchCV(
            self.model.model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Esegui la ricerca
        grid_search.fit(X_proc, y_proc)
        
        # Aggiorna il modello con i migliori parametri
        self.model.model = grid_search.best_estimator_
        
        # Aggiorna i metadati
        self.model.is_trained = True
        self.model.metadata["last_trained"] = datetime.now().isoformat()
        self.model.metadata["best_params"] = grid_search.best_params_
        self.model.metadata["best_score"] = float(grid_search.best_score_)
        
        # Preparazione risultati
        results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in grid_search.cv_results_.items()
            }
        }
        
        model_logger.info(f"Ottimizzazione completata per {self.model.name}. "
                         f"Miglior punteggio: {grid_search.best_score_:.4f}")
        
        return results


class KerasTrainer(ModelTrainer):
    """
    Trainer specializzato per modelli Keras/TensorFlow.
    """
    
    def __init__(self, model: TensorFlowModel):
        """
        Inizializza il trainer Keras.
        
        Args:
            model: Modello TensorFlow da addestrare
        """
        super().__init__(model)
        
        # Configura callback di early stopping
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Configura callback per ridurre il learning rate
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    
    @time_it
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello Keras/TensorFlow con callback.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con risultati dell'addestramento
        """
        # Configura callbacks
        callbacks = kwargs.pop("callbacks", [])
        use_early_stopping = kwargs.pop("use_early_stopping", True)
        use_reduce_lr = kwargs.pop("use_reduce_lr", True)
        
        if use_early_stopping:
            callbacks.append(self.early_stopping)
        
        if use_reduce_lr:
            callbacks.append(self.reduce_lr)
        
        # Addestra il modello
        return super().train(
            X_train, y_train, X_val, y_val, 
            callbacks=callbacks, 
            **kwargs
        )
    
    @time_it
    def hyperparameter_tuning(self, build_fn: Callable, X: np.ndarray, y: np.ndarray,
                            param_grid: Dict[str, List[Any]], X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Esegue l'ottimizzazione degli iperparametri per modelli Keras.
        
        Args:
            build_fn: Funzione che costruisce il modello
            X: Features di training
            y: Target di training
            param_grid: Griglia di parametri da testare
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            
        Returns:
            Dizionario con risultati dell'ottimizzazione
        """
        model_logger.info(f"Inizio ottimizzazione iperparametri per {self.model.name}")
        
        # Preprocessa i dati
        X_proc = self.model.preprocess_X(X, fit=True)
        y_proc = self.model.preprocess_y(y, fit=True)
        
        if X_val is not None and y_val is not None:
            X_val_proc = self.model.preprocess_X(X_val)
            y_val_proc = self.model.preprocess_y(y_val)
            validation_data = (X_val_proc, y_val_proc)
        else:
            validation_data = None
        
        # Parametri di base
        base_params = {
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS,
            "validation_split": DEFAULT_VALIDATION_SPLIT if validation_data is None else 0,
            "validation_data": validation_data,
            "callbacks": [self.early_stopping, self.reduce_lr]
        }
        
        # Genera tutte le combinazioni di parametri
        import itertools
        param_keys = param_grid.keys()
        param_values = [param_grid[key] for key in param_keys]
        param_combinations = list(itertools.product(*param_values))
        
        # Testa tutte le combinazioni
        best_val_loss = float('inf')
        best_params = None
        best_model = None
        results = []
        
        for params in param_combinations:
            # Crea dizionario dei parametri
            current_params = dict(zip(param_keys, params))
            model_params = {k: v for k, v in current_params.items() if k not in base_params}
            training_params = {k: v for k, v in current_params.items() if k in base_params}
            
            # Aggiorna parametri di base
            current_train_params = base_params.copy()
            current_train_params.update(training_params)
            
            # Costruisci il modello
            current_model = build_fn(**model_params)
            
            # Addestra il modello
            history = current_model.fit(
                X_proc, y_proc,
                **current_train_params
            )
            
            # Valuta il modello
            val_loss = min(history.history['val_loss'])
            
            # Salva risultati
            result = {
                "params": current_params,
                "val_loss": val_loss,
                "history": {k: v[-1] for k, v in history.history.items()}
            }
            results.append(result)
            
            # Aggiorna il miglior modello
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = current_params
                best_model = current_model
        
        # Aggiorna il modello con i migliori parametri
        if best_model is not None:
            # Copia i pesi
            self.model.model.set_weights(best_model.get_weights())
            
            # Aggiorna metadati
            self.model.is_trained = True
            self.model.metadata["last_trained"] = datetime.now().isoformat()
            self.model.metadata["best_params"] = best_params
            self.model.metadata["best_val_loss"] = best_val_loss
        
        # Preparazione risultati
        tuning_results = {
            "best_params": best_params,
            "best_val_loss": best_val_loss,
            "all_results": results
        }
        
        model_logger.info(f"Ottimizzazione completata per {self.model.name}. "
                         f"Miglior val_loss: {best_val_loss:.4f}")
        
        return tuning_results


# Factory functions

def create_sklearn_model(model_class: Optional[Any] = None, 
                       model_params: Optional[Dict[str, Any]] = None,
                       name: str = "sklearn_model", 
                       version: str = "1.0.0",
                       filepath: Optional[str] = None) -> SklearnModel:
    """
    Crea o carica un modello scikit-learn.
    
    Args:
        model_class: Classe del modello scikit-learn (ignorato se filepath è fornito)
        model_params: Parametri per il modello scikit-learn (ignorato se filepath è fornito)
        name: Nome del modello
        version: Versione del modello
        filepath: Percorso del file per caricare un modello esistente
        
    Returns:
        Istanza di SklearnModel
        
    Raises:
        ModelError: Se la creazione fallisce
    """
    try:
        if filepath:
            # Carica modello esistente
            model = SklearnModel.load(filepath)
            return model
        
        if model_class is None:
            raise ValueError("model_class deve essere specificato quando si crea un nuovo modello")
        
        # Crea nuovo modello
        params = model_params or {}
        sklearn_model = model_class(**params)
        
        model = SklearnModel(sklearn_model, name=name, version=version)
        
        model_logger.info(f"Modello scikit-learn {name} creato")
        return model
    
    except Exception as e:
        error_msg = f"Errore nella creazione del modello scikit-learn: {str(e)}"
        model_logger.error(error_msg)
        raise ModelError(error_msg)


def load_keras_model(filepath: str) -> TensorFlowModel:
    """
    Carica un modello Keras/TensorFlow.
    
    Args:
        filepath: Percorso del file
        
    Returns:
        Istanza di TensorFlowModel
        
    Raises:
        ModelError: Se il caricamento fallisce
    """
    try:
        # Determina il tipo di modello dalle proprietà del file
        config_path = filepath.replace('.h5', '.json').replace('.keras', '.json')
        
        if not os.path.exists(config_path):
            raise ModelError(f"File di configurazione non trovato: {config_path}")
        
        config_data = load_json(config_path)
        metadata = config_data.get("metadata", {})
        
        # Determina la classe concreta da usare
        model_name = metadata.get("name", "unknown")
        
        if "lstm" in model_name.lower():
            from .lstm import TALSTM
            return TALSTM.load(filepath)
        elif "transformer" in model_name.lower():
            from .transformer import TATransformer
            return TATransformer.load(filepath)
        else:
            # Fallback generico
            return TensorFlowModel.load(filepath)
    
    except Exception as e:
        error_msg = f"Errore nel caricamento del modello Keras: {str(e)}"
        model_logger.error(error_msg)
        raise ModelError(error_msg)


def prepare_classification_data(df: pd.DataFrame, feature_columns: List[str], 
                              target_column: str, threshold: float = 0.0,
                              sequence_length: Optional[int] = None,
                              forecast_horizon: int = 1) -> Tuple:
    """
    Prepara i dati per un problema di classificazione (es. previsione direzione).
    
    Args:
        df: DataFrame con i dati
        feature_columns: Colonne da usare come features
        target_column: Colonna da usare per il target (es. 'close')
        threshold: Soglia per determinare la classe (default: 0 = qualsiasi movimento)
        sequence_length: Lunghezza della sequenza (opzionale)
        forecast_horizon: Orizzonte di previsione
        
    Returns:
        Tuple con (X, y, feature_names) dove y è binario (0/1)
    """
    if df is None or df.empty:
        raise ValueError("Il DataFrame è vuoto o None")
    
    # Verifica che le colonne richieste siano presenti
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if target_column not in df.columns:
        missing_columns.append(target_column)
    
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel DataFrame: {missing_columns}")
    
    # Calcola il target (direzione del movimento)
    df = df.copy()
    df['target'] = df[target_column].shift(-forecast_horizon) - df[target_column]
    df['direction'] = (df['target'] > threshold).astype(int)
    
    # Rimuovi le righe con NaN
    df = df.dropna()
    
    if sequence_length:
        # Crea sequenze
        generator = SequenceGenerator(
            window_size=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        X, y = generator.create_sequences(
            df, 
            feature_columns=feature_columns, 
            target_columns=['direction']
        )
        
        # Espandi feature_names per ogni timestep
        feature_names = []
        for t in range(sequence_length):
            for name in feature_columns:
                feature_names.append(f"{name}_t-{sequence_length-t}")
    else:
        # Usa i dati senza sequenze
        X = df[feature_columns].values
        y = df['direction'].values
        feature_names = feature_columns
    
    return X, y, feature_names


def prepare_regression_data(df: pd.DataFrame, feature_columns: List[str], 
                          target_column: str, target_transform: Optional[str] = 'diff',
                          sequence_length: Optional[int] = None,
                          forecast_horizon: int = 1) -> Tuple:
    """
    Prepara i dati per un problema di regressione (es. previsione prezzi).
    
    Args:
        df: DataFrame con i dati
        feature_columns: Colonne da usare come features
        target_column: Colonna da usare per il target (es. 'close')
        target_transform: Trasformazione del target (None, 'diff', 'pct', 'log_diff')
        sequence_length: Lunghezza della sequenza (opzionale)
        forecast_horizon: Orizzonte di previsione
        
    Returns:
        Tuple con (X, y, feature_names, scaler_y) dove y è continuo
    """
    if df is None or df.empty:
        raise ValueError("Il DataFrame è vuoto o None")
    
    # Verifica che le colonne richieste siano presenti
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if target_column not in df.columns:
        missing_columns.append(target_column)
    
    if missing_columns:
        raise ValueError(f"Colonne mancanti nel DataFrame: {missing_columns}")
    
    # Calcola il target con la trasformazione appropriata
    df = df.copy()
    
    if target_transform == 'diff':
        # Differenza assoluta
        df['target'] = df[target_column].shift(-forecast_horizon) - df[target_column]
    elif target_transform == 'pct':
        # Variazione percentuale
        df['target'] = df[target_column].pct_change(periods=-forecast_horizon)
    elif target_transform == 'log_diff':
        # Log-return
        df['target'] = np.log(df[target_column].shift(-forecast_horizon) / df[target_column])
    else:
        # Nessuna trasformazione
        df['target'] = df[target_column].shift(-forecast_horizon)
    
    # Rimuovi le righe con NaN
    df = df.dropna()
    
    if sequence_length:
        # Crea sequenze
        generator = SequenceGenerator(
            window_size=sequence_length,
            forecast_horizon=forecast_horizon
        )
        
        X, y = generator.create_sequences(
            df, 
            feature_columns=feature_columns, 
            target_columns=['target']
        )
        
        # Espandi feature_names per ogni timestep
        feature_names = []
        for t in range(sequence_length):
            for name in feature_columns:
                feature_names.append(f"{name}_t-{sequence_length-t}")
    else:
        # Usa i dati senza sequenze
        X = df[feature_columns].values
        y = df['target'].values.reshape(-1, 1)  # Reshape per compatibilità
        feature_names = feature_columns
    
    # Crea scaler per il target
    from sklearn.preprocessing import StandardScaler
    scaler_y = StandardScaler()
    scaler_y.fit(y.reshape(-1, 1))
    
    return X, y, feature_names, scaler_y