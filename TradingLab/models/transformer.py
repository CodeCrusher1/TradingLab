# Modello Transformer

"""
Implementazione del modello Transformer per previsioni finanziarie.
Questo modulo fornisce un'implementazione del modello Transformer adattato per serie temporali finanziarie.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Flatten, Concatenate,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
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


class TransformerBlock(tf.keras.layers.Layer):
    """
    Blocco Transformer con multi-head attention e feed-forward network.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        """
        Inizializza il blocco Transformer.
        
        Args:
            embed_dim: Dimensione dell'embedding
            num_heads: Numero di teste per l'attenzione
            ff_dim: Dimensione della feed-forward network
            rate: Tasso di dropout
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):
        """
        Forward pass del blocco Transformer.
        
        Args:
            inputs: Input tensor
            training: Flag di training
            
        Returns:
            Output tensor
        """
        # Implementazione del Transformer con connessioni residue
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Layer per aggiungere encoding posizionale agli input.
    """
    
    def __init__(self, max_steps: int, d_model: int):
        """
        Inizializza l'encoding posizionale.
        
        Args:
            max_steps: Numero massimo di step (lunghezza sequenza)
            d_model: Dimensione del modello (embedding)
        """
        super(PositionalEncoding, self).__init__()
        self.max_steps = max_steps
        self.d_model = d_model
        
        # Crea l'encoding posizionale fisso
        pos_encoding = self.positional_encoding(max_steps, d_model)
        self.pos_encoding = tf.cast(pos_encoding, tf.float32)
    
    def positional_encoding(self, max_steps: int, d_model: int) -> tf.Tensor:
        """
        Calcola l'encoding posizionale.
        
        Args:
            max_steps: Numero massimo di step
            d_model: Dimensione del modello
            
        Returns:
            Tensor con l'encoding posizionale
        """
        # Calcola l'encoding posizionale secondo la formula
        pos = np.arange(max_steps)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        
        # Applica seno alle posizioni pari e coseno alle posizioni dispari
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = np.zeros((max_steps, d_model))
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines
        
        # Aggiungi una dimensione di batch
        pos_encoding = pos_encoding[np.newaxis, ...]
        
        return tf.convert_to_tensor(pos_encoding)
    
    def call(self, inputs):
        """
        Forward pass con aggiunta dell'encoding posizionale.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Tensor con encoding posizionale aggiunto
        """
        seq_len = tf.shape(inputs)[1]
        
        # Assicurati che la sequenza non sia più lunga dell'encoding posizionale
        if seq_len > self.max_steps:
            raise ValueError(f"Lunghezza sequenza {seq_len} supera la lunghezza massima {self.max_steps}")
        
        # Slice dell'encoding posizionale per la lunghezza della sequenza attuale
        pos_encoding = self.pos_encoding[:, :seq_len, :]
        
        return inputs + pos_encoding


class TATransformer(TensorFlowModel):
    """
    Modello Transformer per analisi tecnica (Technical Analysis Transformer).
    """
    
    def __init__(self, name: str = "TA-Transformer", version: str = "1.0.0"):
        """
        Inizializza il modello TA-Transformer.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        
        # Parametri specifici del modello
        self.config["parameters"].update({
            "embed_dim": 64,
            "num_heads": 4,
            "ff_dim": 128,
            "transformer_blocks": 2,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "output_activation": "linear"  # "sigmoid" per classificazione binaria
        })
    
    def _build_model_impl(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione della costruzione del modello Transformer.
        
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
        embed_dim = self.config["parameters"]["embed_dim"]
        num_heads = self.config["parameters"]["num_heads"]
        ff_dim = self.config["parameters"]["ff_dim"]
        transformer_blocks = self.config["parameters"]["transformer_blocks"]
        dropout_rate = self.config["parameters"]["dropout_rate"]
        learning_rate = self.config["parameters"]["learning_rate"]
        output_activation = self.config["parameters"]["output_activation"]
        
        # Input layer (richiede shape [batch, timesteps, features])
        if len(input_shape) == 2:
            # Se l'input è 2D, aggiungi una dimensione per le features
            input_shape = (input_shape[0], input_shape[1], 1)
        
        # Crea il modello con API funzionale di Keras
        inputs = Input(shape=input_shape[1:])
        
        # Aggiungi encoding posizionale
        max_steps = input_shape[1]
        feature_dim = input_shape[2]
        pos_encoding = PositionalEncoding(max_steps, feature_dim)(inputs)
        
        # Proiezione a embed_dim se feature_dim è diverso
        if feature_dim != embed_dim:
            x = Dense(embed_dim)(pos_encoding)
        else:
            x = pos_encoding
        
        # Strati Transformer
        for _ in range(transformer_blocks):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        
        # Global average pooling per ridurre la dimensione temporale
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers per output
        x = Dense(ff_dim, activation="relu")(x)
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
                       f"{transformer_blocks} blocchi Transformer con {num_heads} teste di attenzione")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello Transformer con early stopping e learning rate reduction.
        
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
                patience=kwargs.pop("early_stopping_patience", 15),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Reduce LR on plateau se non disabilitato esplicitamente
        if kwargs.pop("use_reduce_lr", True):
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=kwargs.pop("lr_reduction_factor", 0.2),
                patience=kwargs.pop("lr_patience", 7),
                min_lr=kwargs.pop("min_lr", 1e-6)
            )
            callbacks.append(reduce_lr)
        
        # Richiama l'implementazione della classe base con i callback configurati
        return super().train(X_train, y_train, X_val, y_val, callbacks=callbacks, **kwargs)


class HybridTransformer(TensorFlowModel):
    """
    Modello ibrido Transformer-CNN per analisi tecnica.
    """
    
    def __init__(self, name: str = "TA-HybridTransformer", version: str = "1.0.0"):
        """
        Inizializza il modello TA-HybridTransformer.
        
        Args:
            name: Nome del modello
            version: Versione del modello
        """
        super().__init__(name, version)
        
        # Parametri specifici del modello
        self.config["parameters"].update({
            "embed_dim": 64,
            "num_heads": 4,
            "ff_dim": 128,
            "transformer_blocks": 1,
            "conv_filters": [32, 64],
            "conv_kernel_size": 3,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "output_activation": "linear"  # "sigmoid" per classificazione binaria
        })
    
    def _build_model_impl(self, input_shape: Tuple[int, ...], output_shape: int, **kwargs) -> None:
        """
        Implementazione della costruzione del modello Hybrid Transformer.
        
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
        embed_dim = self.config["parameters"]["embed_dim"]
        num_heads = self.config["parameters"]["num_heads"]
        ff_dim = self.config["parameters"]["ff_dim"]
        transformer_blocks = self.config["parameters"]["transformer_blocks"]
        conv_filters = self.config["parameters"]["conv_filters"]
        conv_kernel_size = self.config["parameters"]["conv_kernel_size"]
        dropout_rate = self.config["parameters"]["dropout_rate"]
        learning_rate = self.config["parameters"]["learning_rate"]
        output_activation = self.config["parameters"]["output_activation"]
        
        # Input layer (richiede shape [batch, timesteps, features])
        if len(input_shape) == 2:
            # Se l'input è 2D, aggiungi una dimensione per le features
            input_shape = (input_shape[0], input_shape[1], 1)
        
        # Crea il modello con API funzionale di Keras
        inputs = Input(shape=input_shape[1:])
        
        # Path Transformer
        max_steps = input_shape[1]
        feature_dim = input_shape[2]
        pos_encoding = PositionalEncoding(max_steps, feature_dim)(inputs)
        
        # Proiezione a embed_dim se feature_dim è diverso
        if feature_dim != embed_dim:
            transformer_path = Dense(embed_dim)(pos_encoding)
        else:
            transformer_path = pos_encoding
        
        # Strati Transformer
        for _ in range(transformer_blocks):
            transformer_path = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(transformer_path)
        
        # Global average pooling per il path Transformer
        transformer_output = GlobalAveragePooling1D()(transformer_path)
        
        # Path CNN
        cnn_path = inputs
        
        # Strati convoluzionali
        for filters in conv_filters:
            cnn_path = tf.keras.layers.Conv1D(filters=filters, kernel_size=conv_kernel_size, 
                                            activation="relu", padding="same")(cnn_path)
            cnn_path = BatchNormalization()(cnn_path)
            cnn_path = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(cnn_path)
        
        # Flatten per il path CNN
        cnn_output = Flatten()(cnn_path)
        
        # Concatena gli output dei due path
        combined = Concatenate()([transformer_output, cnn_output])
        
        # Dense layers per output
        x = Dense(ff_dim, activation="relu")(combined)
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
                       f"Hybrid con {transformer_blocks} blocchi Transformer e {len(conv_filters)} strati CNN")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Addestra il modello Hybrid Transformer.
        
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
                patience=kwargs.pop("early_stopping_patience", 15),
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Reduce LR on plateau se non disabilitato esplicitamente
        if kwargs.pop("use_reduce_lr", True):
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=kwargs.pop("lr_reduction_factor", 0.2),
                patience=kwargs.pop("lr_patience", 7),
                min_lr=kwargs.pop("min_lr", 1e-6)
            )
            callbacks.append(reduce_lr)
        
        # Richiama l'implementazione della classe base con i callback configurati
        return super().train(X_train, y_train, X_val, y_val, callbacks=callbacks, **kwargs)


# Registra i modelli nel registro
from .base import model_registry

model_registry.register_model(
    TATransformer,
    "ta_transformer",
    "Transformer per analisi tecnica"
)

model_registry.register_model(
    HybridTransformer,
    "ta_hybrid_transformer",
    "Modello ibrido Transformer-CNN per analisi tecnica"
)


# Factory functions

def create_transformer_model(model_type: str = "ta_transformer", name: Optional[str] = None, 
                            version: str = "1.0.0", **kwargs) -> TensorFlowModel:
    """
    Crea un modello Transformer per analisi tecnica.
    
    Args:
        model_type: Tipo di modello Transformer ('ta_transformer', 'ta_hybrid_transformer')
        name: Nome del modello (opzionale)
        version: Versione del modello
        **kwargs: Parametri aggiuntivi
        
    Returns:
        Istanza del modello Transformer
        
    Raises:
        ModelError: Se il tipo di modello non è supportato
    """
    if model_type not in ["ta_transformer", "ta_hybrid_transformer"]:
        raise ModelError(f"Tipo di modello Transformer non supportato: {model_type}")
    
    # Usa il nome predefinito se non fornito
    if name is None:
        name = {
            "ta_transformer": "TA-Transformer",
            "ta_hybrid_transformer": "TA-HybridTransformer"
        }[model_type]
    
    # Crea il modello appropriato
    if model_type == "ta_transformer":
        model = TATransformer(name=name, version=version)
    elif model_type == "ta_hybrid_transformer":
        model = HybridTransformer(name=name, version=version)
    
    # Aggiorna i parametri se forniti
    for key, value in kwargs.items():
        if key in model.config["parameters"]:
            model.config["parameters"][key] = value
    
    model_logger.info(f"Creato modello {name} di tipo {model_type}")
    return model