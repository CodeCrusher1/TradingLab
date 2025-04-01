# Modello TA-LSTM

"""
Implementazione del modello TA-LSTM (Technical Analysis LSTM) per previsioni finanziarie.
Questo modello utilizza indicatori di analisi tecnica come input per una rete LSTM.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, 
    Bidirectional, Conv1D, MaxPooling1D, Flatten, Attention,
    LayerNormalization, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

# Importazioni dal modulo base
from .base import TensorFlowModel

# Importazioni dal modulo utils
from ..utils import (
    app_logger, model_logger, ModelError, TrainingError, InferenceError,
    time_it
)


class TALSTM(TensorFlowModel):
    """
    Modello LSTM per analisi tecnica (Technical Analysis LSTM).
    """
    
    def __init__(self, name: str = "TA-LSTM", version: str = "1.0.0"):
        """
        Inizializza il modello TA-LSTM.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        
        # Parametri specifici del modello
        self.config["parameters"].update({
            "lstm_units": 64,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "bidirectional": False,
            "lstm_layers": 2,
            "activation": "relu",
            "recurrent_activation": "sigmoid",
            "output_activation": "linear"  # "sigmoid" per classificazione binaria
        })
    
    def _build_model_impl(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione della costruzione del modello LSTM.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi
        """
        # Aggiorna i parametri con quelli forniti
        for key, value in kwargs.items():
            if key in self.config["parameters"]:
                self.config["parameters"][key] = value
        
        # Estrai parametri
        lstm_units = self.config["parameters"]["lstm_units"]
        dropout_rate = self.config["parameters"]["dropout_rate"]
        learning_rate = self.config["parameters"]["learning_rate"]
        bidirectional = self.config["parameters"]["bidirectional"]
        lstm_layers = self.config["parameters"]["lstm_layers"]
        activation = self.config["parameters"]["activation"]
        recurrent_activation = self.config["parameters"]["recurrent_activation"]
        output_activation = self.config["parameters"]["output_activation"]
        
        # Crea il modello
        model = Sequential()
        
        # Input layer (LSTM richiede shape [batch, timesteps, features])
        if len(input_shape) == 2:
            # Se l'input è 2D, aggiungi una dimensione per le features
            input_shape = (input_shape[0], input_shape[1], 1)
        
        # Primo layer LSTM
        if bidirectional:
            model.add(Bidirectional(
                LSTM(lstm_units, activation=activation, recurrent_activation=recurrent_activation,
                     return_sequences=lstm_layers > 1),
                input_shape=input_shape[1:]))
        else:
            model.add(LSTM(lstm_units, activation=activation, recurrent_activation=recurrent_activation,
                           return_sequences=lstm_layers > 1,
                           input_shape=input_shape[1:]))
        
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Layer LSTM aggiuntivi
        for i in range(1, lstm_layers):
            if bidirectional:
                model.add(Bidirectional(
                    LSTM(lstm_units, activation=activation, recurrent_activation=recurrent_activation,
                         return_sequences=i < lstm_layers - 1)))
            else:
                model.add(LSTM(lstm_units, activation=activation, recurrent_activation=recurrent_activation,
                               return_sequences=i < lstm_layers - 1))
            
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(output_shape, activation=output_activation))
        
        # Compila il modello
        if output_activation == "sigmoid":
            # Classificazione binaria
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        elif output_activation == "softmax":
            # Classificazione multi-classe
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        else:
            # Regressione
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="mse",
                metrics=[RootMeanSquaredError(), "mae"]
            )
        
        # Memorizza il modello
        self.model = model
        model_logger.info(f"Modello {self.name} costruito: "
                       f"{lstm_layers} layer {'Bi' if bidirectional else ''}LSTM con {lstm_units} unità")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello LSTM con early stopping e learning rate reduction.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con risultati dell'addestramento
        """
        # Configura callback
        callbacks = kwargs.pop("callbacks", [])
        
        # Early stopping se non disabilitato esplicitamente
        if kwargs.pop("use_early_stopping", True):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=kwargs.pop("early_stopping_patience", 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Reduce LR on plateau se non disabilitato esplicitamente
        if kwargs.pop("use_reduce_lr", True):
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=kwargs.pop("lr_reduction_factor", 0.2),
                patience=kwargs.pop("lr_patience", 5),
                min_lr=kwargs.pop("min_lr", 1e-6)
            )
            callbacks.append(reduce_lr)
        
        # Richiama l'implementazione della classe base con i callback configurati
        return super().train(X_train, y_train, X_val, y_val, callbacks=callbacks, **kwargs)


class BiLSTM(TALSTM):
    """
    Modello LSTM bidirezionale per analisi tecnica.
    """
    
    def __init__(self, name: str = "TA-BiLSTM", version: str = "1.0.0"):
        """
        Inizializza il modello TA-BiLSTM.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        
        # Imposta il flag bidirectional
        self.config["parameters"]["bidirectional"] = True


class ConvLSTM(TensorFlowModel):
    """
    Modello CNN-LSTM per analisi tecnica.
    Utilizza strati convoluzionali 1D prima degli strati LSTM.
    """
    
    def __init__(self, name: str = "TA-ConvLSTM", version: str = "1.0.0"):
        """
        Inizializza il modello TA-ConvLSTM.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        
        # Parametri specifici del modello
        self.config["parameters"].update({
            "conv_filters": [32, 64],
            "conv_kernel_size": 3,
            "pool_size": 2,
            "lstm_units": 64,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "lstm_layers": 1,
            "activation": "relu",
            "output_activation": "linear"  # "sigmoid" per classificazione binaria
        })
    
    def _build_model_impl(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione della costruzione del modello CNN-LSTM.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi
        """
        # Aggiorna i parametri con quelli forniti
        for key, value in kwargs.items():
            if key in self.config["parameters"]:
                self.config["parameters"][key] = value
        
        # Estrai parametri
        conv_filters = self.config["parameters"]["conv_filters"]
        conv_kernel_size = self.config["parameters"]["conv_kernel_size"]
        pool_size = self.config["parameters"]["pool_size"]
        lstm_units = self.config["parameters"]["lstm_units"]
        dropout_rate = self.config["parameters"]["dropout_rate"]
        learning_rate = self.config["parameters"]["learning_rate"]
        lstm_layers = self.config["parameters"]["lstm_layers"]
        activation = self.config["parameters"]["activation"]
        output_activation = self.config["parameters"]["output_activation"]
        
        # Crea il modello
        model = Sequential()
        
        # Input layer (Conv1D richiede shape [batch, timesteps, features])
        if len(input_shape) == 2:
            # Se l'input è 2D, aggiungi una dimensione per le features
            input_shape = (input_shape[0], input_shape[1], 1)
        
        # Strati convoluzionali
        for i, filters in enumerate(conv_filters):
            if i == 0:
                # Primo strato con forma di input
                model.add(Conv1D(filters=filters, kernel_size=conv_kernel_size, activation=activation,
                                input_shape=input_shape[1:], padding='same'))
            else:
                model.add(Conv1D(filters=filters, kernel_size=conv_kernel_size, activation=activation,
                                padding='same'))
            
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=pool_size, padding='same'))
        
        # Strati LSTM
        for i in range(lstm_layers):
            model.add(LSTM(lstm_units, activation=activation, return_sequences=i < lstm_layers - 1))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(output_shape, activation=output_activation))
        
        # Compila il modello
        if output_activation == "sigmoid":
            # Classificazione binaria
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        elif output_activation == "softmax":
            # Classificazione multi-classe
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        else:
            # Regressione
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="mse",
                metrics=[RootMeanSquaredError(), "mae"]
            )
        
        # Memorizza il modello
        self.model = model
        model_logger.info(f"Modello {self.name} costruito: "
                       f"{len(conv_filters)} layer CNN + {lstm_layers} layer LSTM")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello CNN-LSTM con early stopping e learning rate reduction.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con risultati dell'addestramento
        """
        # Configura callback
        callbacks = kwargs.pop("callbacks", [])
        
        # Early stopping se non disabilitato esplicitamente
        if kwargs.pop("use_early_stopping", True):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=kwargs.pop("early_stopping_patience", 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Reduce LR on plateau se non disabilitato esplicitamente
        if kwargs.pop("use_reduce_lr", True):
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=kwargs.pop("lr_reduction_factor", 0.2),
                patience=kwargs.pop("lr_patience", 5),
                min_lr=kwargs.pop("min_lr", 1e-6)
            )
            callbacks.append(reduce_lr)
        
        # Richiama l'implementazione della classe base con i callback configurati
        return super().train(X_train, y_train, X_val, y_val, callbacks=callbacks, **kwargs)


class AttentionLSTM(TensorFlowModel):
    """
    Modello LSTM con meccanismo di attenzione per analisi tecnica.
    """
    
    def __init__(self, name: str = "TA-AttentionLSTM", version: str = "1.0.0"):
        """
        Inizializza il modello TA-AttentionLSTM.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        
        # Parametri specifici del modello
        self.config["parameters"].update({
            "lstm_units": 64,
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "lstm_layers": 1,
            "attention_units": 32,
            "activation": "relu",
            "output_activation": "linear"  # "sigmoid" per classificazione binaria
        })
    
    def _build_model_impl(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione della costruzione del modello LSTM con attenzione.
        
        Args:
            input_shape: Forma dell'input
            output_shape: Forma dell'output
            **kwargs: Parametri aggiuntivi
        """
        # Aggiorna i parametri con quelli forniti
        for key, value in kwargs.items():
            if key in self.config["parameters"]:
                self.config["parameters"][key] = value
        
        # Estrai parametri
        lstm_units = self.config["parameters"]["lstm_units"]
        dropout_rate = self.config["parameters"]["dropout_rate"]
        learning_rate = self.config["parameters"]["learning_rate"]
        lstm_layers = self.config["parameters"]["lstm_layers"]
        attention_units = self.config["parameters"]["attention_units"]
        activation = self.config["parameters"]["activation"]
        output_activation = self.config["parameters"]["output_activation"]
        
        # Input layer (LSTM richiede shape [batch, timesteps, features])
        if len(input_shape) == 2:
            # Se l'input è 2D, aggiungi una dimensione per le features
            input_shape = (input_shape[0], input_shape[1], 1)
        
        # Crea il modello con API funzionale di Keras
        inputs = Input(shape=input_shape[1:])
        
        # Strati LSTM
        x = inputs
        for i in range(lstm_layers):
            x = LSTM(lstm_units, activation=activation, return_sequences=True)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        # Meccanismo di attenzione
        # Implementazione semplificata usando layer di attenzione di Keras
        if hasattr(tf.keras.layers, 'Attention'):
            # Se disponibile, usa l'implementazione nativa
            attention_output = tf.keras.layers.Attention()([x, x])
        else:
            # Altrimenti implementa manualmente
            attention_scores = Dense(attention_units, activation='tanh')(x)
            attention_scores = Dense(1, activation='softmax')(attention_scores)
            attention_output = tf.keras.layers.multiply([x, attention_scores])
        
        # Global average pooling per ridurre la dimensione temporale
        x = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
        
        # Dense layers per output
        x = Dense(attention_units, activation=activation)(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(output_shape, activation=output_activation)(x)
        
        # Crea il modello
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compila il modello
        if output_activation == "sigmoid":
            # Classificazione binaria
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        elif output_activation == "softmax":
            # Classificazione multi-classe
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        else:
            # Regressione
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="mse",
                metrics=[RootMeanSquaredError(), "mae"]
            )
        
        # Memorizza il modello
        self.model = model
        model_logger.info(f"Modello {self.name} costruito: "
                       f"{lstm_layers} layer LSTM con meccanismo di attenzione")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello LSTM con attenzione.
        
        Args:
            X_train: Features di training
            y_train: Target di training
            X_val: Features di validazione (opzionale)
            y_val: Target di validazione (opzionale)
            **kwargs: Parametri aggiuntivi
            
        Returns:
            Dizionario con risultati dell'addestramento
        """
        # Configura callback
        callbacks = kwargs.pop("callbacks", [])
        
        # Early stopping se non disabilitato esplicitamente
        if kwargs.pop("use_early_stopping", True):
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=kwargs.pop("early_stopping_patience", 10),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Reduce LR on plateau se non disabilitato esplicitamente
        if kwargs.pop("use_reduce_lr", True):
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=kwargs.pop("lr_reduction_factor", 0.2),
                patience=kwargs.pop("lr_patience", 5),
                min_lr=kwargs.pop("min_lr", 1e-6)
            )
            callbacks.append(reduce_lr)
        
        # Richiama l'implementazione della classe base con i callback configurati
        return super().train(X_train, y_train, X_val, y_val, callbacks=callbacks, **kwargs)


# Registra i modelli nel registro
from .base import model_registry

model_registry.register_model(
    TALSTM,
    "ta_lstm",
    "LSTM per analisi tecnica"
)

model_registry.register_model(
    BiLSTM,
    "ta_bilstm",
    "LSTM bidirezionale per analisi tecnica"
)

model_registry.register_model(
    ConvLSTM,
    "ta_convlstm",
    "CNN-LSTM per analisi tecnica"
)

model_registry.register_model(
    AttentionLSTM,
    "ta_attention_lstm",
    "LSTM con attenzione per analisi tecnica"
)


# Factory functions

def create_lstm_model(model_type: str = "ta_lstm", name: Optional[str] = None, 
                    version: str = "1.0.0", **kwargs) -> TensorFlowModel:
    """
    Crea un modello LSTM per analisi tecnica.
    
    Args:
        model_type: Tipo di modello LSTM ('ta_lstm', 'ta_bilstm', 'ta_convlstm', 'ta_attention_lstm')
        name: Nome del modello (opzionale)
        version: Versione del modello
        **kwargs: Parametri aggiuntivi
        
    Returns:
        Istanza del modello LSTM
        
    Raises:
        ModelError: Se il tipo di modello non è supportato
    """
    if model_type not in ["ta_lstm", "ta_bilstm", "ta_convlstm", "ta_attention_lstm"]:
        raise ModelError(f"Tipo di modello LSTM non supportato: {model_type}")
    
    # Usa il nome predefinito se non fornito
    if name is None:
        name = {
            "ta_lstm": "TA-LSTM",
            "ta_bilstm": "TA-BiLSTM",
            "ta_convlstm": "TA-ConvLSTM",
            "ta_attention_lstm": "TA-AttentionLSTM"
        }[model_type]
    
    # Crea il modello appropriato
    if model_type == "ta_lstm":
        model = TALSTM(name=name, version=version)
    elif model_type == "ta_bilstm":
        model = BiLSTM(name=name, version=version)
    elif model_type == "ta_convlstm":
        model = ConvLSTM(name=name, version=version)
    elif model_type == "ta_attention_lstm":
        model = AttentionLSTM(name=name, version=version)
    
    # Aggiorna i parametri se forniti
    for key, value in kwargs.items():
        if key in model.config["parameters"]:
            model.config["parameters"][key] = value
    
    model_logger.info(f"Creato modello {name} di tipo {model_type}")
    return model