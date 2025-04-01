# Ensemble di modelli

"""
Implementazione di modelli ensemble per previsioni finanziarie.
Questo modulo fornisce classi per combinare più modelli in approcci ensemble.
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime

# Importazioni dal modulo base
from .base import BaseModel, TensorFlowModel

# Importazioni dal modulo utils
from ..utils import (
    app_logger, model_logger, ModelError, TrainingError, InferenceError,
    time_it, save_json, load_json, ensure_directory
)

# Importazioni dal modulo config
from ..config import (
    MODELS_DIR, get_model_path
)


class EnsembleModel(BaseModel):
    """
    Classe base per modelli ensemble.
    """
    
    def __init__(self, name: str = "Ensemble", version: str = "1.0.0"):
        """
        Inizializza il modello ensemble.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        self.models: List[BaseModel] = []
        self.weights: Optional[np.ndarray] = None
        self.config["ensemble_type"] = "average"
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Aggiunge un modello all'ensemble.
        
        Args:
            model: Modello da aggiungere
            weight: Peso del modello nell'ensemble
        """
        self.models.append(model)
        
        # Aggiorna i pesi
        if self.weights is None:
            self.weights = np.array([weight])
        else:
            self.weights = np.append(self.weights, weight)
        
        # Normalizza i pesi
        self.weights = self.weights / np.sum(self.weights)
        
        model_logger.info(f"Aggiunto modello {model.name} all'ensemble {self.name} con peso {weight}")
    
    def build(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione vuota per compatibilità con l'interfaccia.
        I modelli ensemble non richiedono una build esplicita.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi
        """
        # Memorizza forma di input e output
        self.config["parameters"]["input_shape"] = input_shape
        self.config["parameters"]["output_shape"] = output_shape
        
        model_logger.info(f"Ensemble {self.name} configurato con {len(self.models)} modelli")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra tutti i modelli nell'ensemble.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con risultati dell'addestramento
            
        Raises:
            TrainingError: Se l'addestramento fallisce
        """
        if not self.models:
            raise TrainingError(f"Nessun modello nell'ensemble {self.name}")
        
        # Memorizza i nomi delle feature e target se forniti
        feature_names = kwargs.pop("feature_names", None)
        target_names = kwargs.pop("target_names", None)
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        if target_names is not None:
            self.target_names = target_names
        
        results_list = []
        
        # Addestra ogni modello
        for i, model in enumerate(self.models):
            try:
                model_logger.info(f"Addestramento del modello {i+1}/{len(self.models)} "
                               f"({model.name}) nell'ensemble {self.name}")
                
                # Passa feature_names e target_names anche ai modelli componenti
                model_kwargs = kwargs.copy()
                if feature_names is not None:
                    model_kwargs["feature_names"] = feature_names
                
                if target_names is not None:
                    model_kwargs["target_names"] = target_names
                
                results = model.train(X_train, y_train, X_val, y_val, **model_kwargs)
                results_list.append(results)
                
                model_logger.info(f"Addestramento del modello {model.name} completato")
            
            except Exception as e:
                error_msg = f"Errore durante l'addestramento del modello {model.name} nell'ensemble {self.name}: {str(e)}"
                model_logger.error(error_msg)
                raise TrainingError(error_msg)
        
        # Aggiorna i metadati
        self.is_trained = True
        self.metadata["last_trained"] = datetime.now().isoformat()
        
        # Valuta l'ensemble sul validation set
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.metadata["validation_metrics"] = val_metrics
        
        # Prepara i risultati dell'ensemble
        results = {
            "ensemble_name": self.name,
            "models_count": len(self.models),
            "model_results": results_list,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "validation_metrics": self.metadata.get("validation_metrics", {})
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Effettua previsioni combinando i risultati dei modelli nell'ensemble.
        
        Args:
            X: Dati di input
            
        Returns:
            Previsioni del modello ensemble
            
        Raises:
            InferenceError: Se la previsione fallisce
        """
        if not self.models:
            raise InferenceError(f"Nessun modello nell'ensemble {self.name}")
        
        if not self.is_trained:
            raise InferenceError(f"Ensemble {self.name} non addestrato")
        
        try:
            predictions = []
            
            # Ottieni previsioni da ogni modello
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            
            # Combina le previsioni in base al tipo di ensemble
            if self.config["ensemble_type"] == "average":
                # Media pesata
                combined = np.zeros_like(predictions[0])
                
                for i, pred in enumerate(predictions):
                    weight = self.weights[i] if self.weights is not None else 1.0 / len(self.models)
                    combined += pred * weight
            
            elif self.config["ensemble_type"] == "voting":
                # Votazione (per classificazione)
                # Converti previsioni in classi
                classes = []
                
                for pred in predictions:
                    if len(pred.shape) > 1 and pred.shape[1] > 1:
                        # Multi-classe
                        classes.append(np.argmax(pred, axis=1))
                    else:
                        # Binaria
                        classes.append((pred > 0.5).astype(int))
                
                # Converti in array per il conteggio
                classes = np.array(classes)
                
                # Conta i voti per ogni classe
                n_samples = classes.shape[1]
                n_classes = np.max(classes) + 1
                
                # Inizializza array per il conteggio dei voti
                votes = np.zeros((n_samples, n_classes))
                
                # Conta i voti pesati
                for i, class_pred in enumerate(classes):
                    weight = self.weights[i] if self.weights is not None else 1.0 / len(self.models)
                    for j, cls in enumerate(class_pred):
                        votes[j, cls] += weight
                
                # Classe con più voti
                combined = np.argmax(votes, axis=1)
                
                # Converti in one-hot se la prima previsione è one-hot
                if len(predictions[0].shape) > 1 and predictions[0].shape[1] > 1:
                    one_hot = np.zeros_like(predictions[0])
                    for i, cls in enumerate(combined):
                        one_hot[i, cls] = 1
                    combined = one_hot
                else:
                    # Reshape per match di dimensioni
                    combined = combined.reshape(-1, 1)
            
            elif self.config["ensemble_type"] == "stacking":
                # Stacking richiede un meta-modello aggiuntivo
                raise NotImplementedError("Stacking non ancora implementato")
            
            else:
                # Fallback a media semplice
                combined = np.mean(predictions, axis=0)
            
            return combined
        
        except Exception as e:
            error_msg = f"Errore durante la previsione con l'ensemble {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Valuta le prestazioni del modello ensemble.
        
        Args:
            X: Dati di input
            y: Target reali
            
        Returns:
            Dizionario con metriche di valutazione
            
        Raises:
            InferenceError: Se la valutazione fallisce
        """
        try:
            # Effettua previsioni
            y_pred = self.predict(X)
            
            # Determina il tipo di problema (classificazione o regressione)
            # basandosi sul primo modello
            if not self.models:
                raise InferenceError(f"Nessun modello nell'ensemble {self.name}")
            
            first_model_metrics = self.models[0].metadata.get("validation_metrics", {})
            
            if "accuracy" in first_model_metrics or hasattr(self.models[0].model, 'predict_proba'):
                # Classificazione
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                if len(y.shape) > 1 and y.shape[1] > 1:
                    # Multi-classe
                    y_true = np.argmax(y, axis=1)
                    y_pred_class = np.argmax(y_pred, axis=1)
                else:
                    # Binaria
                    y_true = y
                    y_pred_class = (y_pred > 0.5).astype(int)
                
                metrics = {
                    "accuracy": float(accuracy_score(y_true, y_pred_class))
                }
                
                # Calcola precision, recall, F1 solo per problemi binari
                if len(np.unique(y_true)) <= 2:
                    metrics["precision"] = float(precision_score(y_true, y_pred_class, zero_division=0))
                    metrics["recall"] = float(recall_score(y_true, y_pred_class, zero_division=0))
                    metrics["f1"] = float(f1_score(y_true, y_pred_class, zero_division=0))
            
            else:
                # Regressione
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                metrics = {
                    "mse": float(mean_squared_error(y, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                    "mae": float(mean_absolute_error(y, y_pred)),
                    "r2": float(r2_score(y, y_pred))
                }
            
            return metrics
        
        except Exception as e:
            error_msg = f"Errore durante la valutazione dell'ensemble {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def _save_model_impl(self, filepath: str) -> None:
        """
        Implementazione del salvataggio per modelli ensemble.
        
        Args:
            filepath: Percorso del file
        """
        # Crea directory per i modelli componenti
        ensemble_dir = os.path.dirname(filepath)
        models_dir = os.path.join(ensemble_dir, "models")
        ensure_directory(models_dir)
        
        # Salva ogni modello componente
        model_paths = []
        for i, model in enumerate(self.models):
            model_filename = f"{model.name}_{i}.h5"
            model_path = os.path.join(models_dir, model_filename)
            model.save(model_path)
            model_paths.append(model_path)
        
        # Salva meta-informazioni dell'ensemble
        ensemble_info = {
            "model_paths": model_paths,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "ensemble_type": self.config["ensemble_type"]
        }
        
        ensemble_info_path = filepath.replace(".h5", "_ensemble_info.json")
        save_json(ensemble_info_path, ensemble_info)
    
    def _load_model_impl(self, filepath: str, custom_objects: Optional[Dict[str, Any]] = None) -> None:
        """
        Implementazione del caricamento per modelli ensemble.
        
        Args:
            filepath: Percorso del file
            custom_objects: Oggetti personalizzati per il caricamento
        """
        # Carica meta-informazioni dell'ensemble
        ensemble_info_path = filepath.replace(".h5", "_ensemble_info.json")
        
        if not os.path.exists(ensemble_info_path):
            raise ModelError(f"File di meta-informazioni dell'ensemble non trovato: {ensemble_info_path}")
        
        ensemble_info = load_json(ensemble_info_path)
        
        # Carica ogni modello componente
        self.models = []
        for model_path in ensemble_info["model_paths"]:
            # Determina il tipo di modello dal percorso
            if "lstm" in model_path.lower():
                from .lstm import TALSTM
                model = TALSTM.load(model_path)
            elif "transformer" in model_path.lower():
                from .transformer import TATransformer
                model = TATransformer.load(model_path)
            else:
                # Tenta un caricamento generico
                model = BaseModel.load(model_path)
            
            self.models.append(model)
        
        # Carica i pesi
        self.weights = np.array(ensemble_info["weights"]) if ensemble_info["weights"] else None
        
        # Carica il tipo di ensemble
        self.config["ensemble_type"] = ensemble_info["ensemble_type"]
    
    def _reset_model_impl(self) -> None:
        """
        Implementazione del reset per modelli ensemble.
        """
        # Resetta ogni modello componente
        for model in self.models:
            model.reset()


class StackingEnsemble(EnsembleModel):
    """
    Ensemble di modelli con approccio stacking.
    Utilizza un meta-modello per combinare le previsioni dei modelli base.
    """
    
    def __init__(self, name: str = "StackingEnsemble", version: str = "1.0.0"):
        """
        Inizializza il modello ensemble con stacking.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        self.meta_model = None
        self.config["ensemble_type"] = "stacking"
    
    def set_meta_model(self, model: BaseModel) -> None:
        """
        Imposta il meta-modello per lo stacking.
        
        Args:
            model: Meta-modello da utilizzare
        """
        self.meta_model = model
        model_logger.info(f"Impostato meta-modello {model.name} per l'ensemble {self.name}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra l'ensemble con approccio stacking.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con risultati dell'addestramento
            
        Raises:
            TrainingError: Se l'addestramento fallisce
        """
        if not self.models:
            raise TrainingError(f"Nessun modello base nell'ensemble {self.name}")
        
        if self.meta_model is None:
            raise TrainingError(f"Meta-modello non impostato per l'ensemble {self.name}")
        
        try:
            # Memorizza i nomi delle feature e target se forniti
            feature_names = kwargs.pop("feature_names", None)
            target_names = kwargs.pop("target_names", None)
            
            if feature_names is not None:
                self.feature_names = feature_names
            
            if target_names is not None:
                self.target_names = target_names
            
            # Addestra i modelli base con cross-validation
            from sklearn.model_selection import KFold
            
            n_splits = kwargs.pop("n_splits", 5)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Prepara array per le previsioni dei modelli base
            n_samples = X_train.shape[0]
            
            # Determina la forma dell'output
            # Effettua una previsione di prova con il primo modello
            sample_pred = self.models[0].predict(X_train[:1])
            output_shape = sample_pred.shape[1:]
            output_size = np.prod(output_shape).astype(int)
            
            # Prepara array per le previsioni out-of-fold
            meta_features = np.zeros((n_samples, len(self.models) * output_size))
            
            # Addestra i modelli base con cross-validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                model_logger.info(f"Stacking: addestramento fold {fold+1}/{n_splits}")
                
                # Split dei dati
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Addestra ogni modello base
                for i, model in enumerate(self.models):
                    model_logger.info(f"Addestramento del modello {i+1}/{len(self.models)} "
                                   f"({model.name}) nel fold {fold+1}")
                    
                    # Addestra il modello
                    model_kwargs = kwargs.copy()
                    if feature_names is not None:
                        model_kwargs["feature_names"] = feature_names
                    
                    if target_names is not None:
                        model_kwargs["target_names"] = target_names
                    
                    model.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val, **model_kwargs)
                    
                    # Genera previsioni out-of-fold
                    pred = model.predict(X_fold_val)
                    
                    # Appiattisci le previsioni se necessario
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    
                    # Memorizza le previsioni come feature per il meta-modello
                    start_col = i * output_size
                    end_col = (i + 1) * output_size
                    meta_features[val_idx, start_col:end_col] = pred_flat
            
            # Ri-addestra i modelli base su tutti i dati
            base_results = []
            for i, model in enumerate(self.models):
                model_logger.info(f"Ri-addestramento completo del modello {i+1}/{len(self.models)} "
                               f"({model.name})")
                
                # Addestra il modello
                model_kwargs = kwargs.copy()
                if feature_names is not None:
                    model_kwargs["feature_names"] = feature_names
                
                if target_names is not None:
                    model_kwargs["target_names"] = target_names
                
                results = model.train(X_train, y_train, X_val, y_val, **model_kwargs)
                base_results.append(results)
            
            # Prepara meta-feature per il validation set
            if X_val is not None:
                meta_val_features = np.zeros((X_val.shape[0], len(self.models) * output_size))
                
                # Genera meta-features per il validation set
                for i, model in enumerate(self.models):
                    # Genera previsioni
                    pred = model.predict(X_val)
                    
                    # Appiattisci le previsioni se necessario
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    
                    # Memorizza le previsioni come feature per il meta-modello
                    start_col = i * output_size
                    end_col = (i + 1) * output_size
                    meta_val_features[:, start_col:end_col] = pred_flat
            
            # Addestra il meta-modello
            model_logger.info(f"Addestramento del meta-modello {self.meta_model.name}")
            
            # Crea un nome per le feature del meta-modello
            meta_feature_names = []
            if feature_names is not None:
                for model_idx, model in enumerate(self.models):
                    for feat_idx in range(output_size):
                        meta_feature_names.append(f"{model.name}_{feat_idx}")
            
            # Addestra il meta-modello
            meta_kwargs = kwargs.copy()
            if meta_feature_names:
                meta_kwargs["feature_names"] = meta_feature_names
            
            if target_names is not None:
                meta_kwargs["target_names"] = target_names
            
            meta_results = self.meta_model.train(
                meta_features, y_train, 
                meta_val_features if X_val is not None else None,
                y_val if X_val is not None else None,
                **meta_kwargs
            )
            
            # Aggiorna i metadati
            self.is_trained = True
            self.metadata["last_trained"] = datetime.now().isoformat()
            
            # Valuta l'ensemble sul validation set
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                self.metadata["validation_metrics"] = val_metrics
            
            # Prepara i risultati dell'ensemble
            results = {
                "ensemble_name": self.name,
                "models_count": len(self.models),
                "base_model_results": base_results,
                "meta_model_results": meta_results,
                "validation_metrics": self.metadata.get("validation_metrics", {})
            }
            
            return results
            
        except Exception as e:
            error_msg = f"Errore durante l'addestramento dell'ensemble stacking {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise TrainingError(error_msg)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Effettua previsioni combinando i risultati dei modelli base attraverso il meta-modello.
        
        Args:
            X: Dati di input
            
        Returns:
            Previsioni del modello ensemble
            
        Raises:
            InferenceError: Se la previsione fallisce
        """
        if not self.models:
            raise InferenceError(f"Nessun modello base nell'ensemble {self.name}")
        
        if self.meta_model is None:
            raise InferenceError(f"Meta-modello non impostato per l'ensemble {self.name}")
        
        if not self.is_trained:
            raise InferenceError(f"Ensemble {self.name} non addestrato")
        
        try:
            # Determina la forma dell'output dal primo modello
            sample_pred = self.models[0].predict(X[:1])
            output_shape = sample_pred.shape[1:]
            output_size = np.prod(output_shape).astype(int)
            
            # Prepara array per le meta-features
            n_samples = X.shape[0]
            meta_features = np.zeros((n_samples, len(self.models) * output_size))
            
            # Genera meta-features
            for i, model in enumerate(self.models):
                # Genera previsioni
                pred = model.predict(X)
                
                # Appiattisci le previsioni se necessario
                pred_flat = pred.reshape(pred.shape[0], -1)
                
                # Memorizza le previsioni come feature per il meta-modello
                start_col = i * output_size
                end_col = (i + 1) * output_size
                meta_features[:, start_col:end_col] = pred_flat
            
            # Effettua previsioni con il meta-modello
            meta_predictions = self.meta_model.predict(meta_features)
            
            return meta_predictions
            
        except Exception as e:
            error_msg = f"Errore durante la previsione con l'ensemble stacking {self.name}: {str(e)}"
            model_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def _save_model_impl(self, filepath: str) -> None:
        """
        Implementazione del salvataggio per modelli ensemble con stacking.
        
        Args:
            filepath: Percorso del file
        """
        # Crea directory per i modelli componenti
        ensemble_dir = os.path.dirname(filepath)
        models_dir = os.path.join(ensemble_dir, "models")
        ensure_directory(models_dir)
        
        # Salva ogni modello base
        model_paths = []
        for i, model in enumerate(self.models):
            model_filename = f"{model.name}_{i}.h5"
            model_path = os.path.join(models_dir, model_filename)
            model.save(model_path)
            model_paths.append(model_path)
        
        # Salva il meta-modello
        meta_model_path = None
        if self.meta_model is not None:
            meta_model_filename = f"meta_{self.meta_model.name}.h5"
            meta_model_path = os.path.join(models_dir, meta_model_filename)
            self.meta_model.save(meta_model_path)
        
        # Salva meta-informazioni dell'ensemble
        ensemble_info = {
            "model_paths": model_paths,
            "meta_model_path": meta_model_path,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "ensemble_type": self.config["ensemble_type"]
        }
        
        ensemble_info_path = filepath.replace(".h5", "_ensemble_info.json")
        save_json(ensemble_info_path, ensemble_info)
    
    def _load_model_impl(self, filepath: str, custom_objects: Optional[Dict[str, Any]] = None) -> None:
        """
        Implementazione del caricamento per modelli ensemble con stacking.
        
        Args:
            filepath: Percorso del file
            custom_objects: Oggetti personalizzati per il caricamento
        """
        # Carica meta-informazioni dell'ensemble
        ensemble_info_path = filepath.replace(".h5", "_ensemble_info.json")
        
        if not os.path.exists(ensemble_info_path):
            raise ModelError(f"File di meta-informazioni dell'ensemble non trovato: {ensemble_info_path}")
        
        ensemble_info = load_json(ensemble_info_path)
        
        # Carica ogni modello base
        self.models = []
        for model_path in ensemble_info["model_paths"]:
            # Determina il tipo di modello dal percorso
            if "lstm" in model_path.lower():
                from .lstm import TALSTM
                model = TALSTM.load(model_path)
            elif "transformer" in model_path.lower():
                from .transformer import TATransformer
                model = TATransformer.load(model_path)
            else:
                # Tenta un caricamento generico
                model = BaseModel.load(model_path)
            
            self.models.append(model)
        
        # Carica il meta-modello
        meta_model_path = ensemble_info.get("meta_model_path")
        if meta_model_path and os.path.exists(meta_model_path):
            # Determina il tipo di meta-modello dal percorso
            if "lstm" in meta_model_path.lower():
                from .lstm import TALSTM
                self.meta_model = TALSTM.load(meta_model_path)
            elif "transformer" in meta_model_path.lower():
                from .transformer import TATransformer
                self.meta_model = TATransformer.load(meta_model_path)
            else:
                # Tenta un caricamento generico
                self.meta_model = BaseModel.load(meta_model_path)
        else:
            model_logger.warning(f"Meta-modello non trovato per l'ensemble stacking {self.name}")
            self.meta_model = None
        
        # Carica i pesi
        self.weights = np.array(ensemble_info["weights"]) if ensemble_info["weights"] else None
        
        # Carica il tipo di ensemble
        self.config["ensemble_type"] = ensemble_info["ensemble_type"]