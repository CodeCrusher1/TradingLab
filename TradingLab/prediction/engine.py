# Motore di previsione

# Motore di previsione

"""
Sistema di previsione per il progetto TradingLab.
Questo modulo gestisce il processo di generazione previsioni basate su modelli e analisi tecnica.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
import json

# Importazioni dai moduli di progetto
from ..models.base import BaseModel, model_registry
from ..models.ensemble import EnsembleModel
from ..data.processor import DataProcessor
from ..data.database import get_default_repository
from ..data.file_storage import get_file_storage, get_persistence_manager
from ..analysis.indicators import TrendIndicators, MomentumIndicators
from ..analysis.patterns import CandlePatterns, HarmonicPatterns, ChartPatterns
from ..config import get_symbol, get_timeframe, DEFAULT_PREDICTION_TIMEFRAME
from ..utils import app_logger, ModelError, InferenceError


class PredictionResult:
    """
    Classe che rappresenta il risultato di una previsione.
    """
    
    def __init__(self, symbol: str, timeframe: str, timestamp: datetime):
        """
        Inizializza un nuovo risultato di previsione.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe della previsione
            timestamp: Timestamp della previsione
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = timestamp
        self.prediction_timestamp = datetime.now()
        
        # Previsione di direzione (1=up, 0=neutral, -1=down)
        self.direction = 0
        self.confidence = 0.0
        
        # Previsioni dei singoli modelli
        self.model_predictions = {}
        
        # Previsioni di distribuzione di probabilità
        self.probability_distribution = {
            "down": 0.0,
            "neutral": 0.0,
            "up": 0.0
        }
        
        # Livelli di prezzo previsti
        self.entry_price = None
        self.tp_price = None    # Take profit
        self.sl_price = None    # Stop loss
        self.risk_reward_ratio = None
        
        # Raccomandazione di azione
        self.action = "wait"    # buy, sell, wait
        self.order_type = None  # market, limit, stop
        
        # Informazioni sul modello
        self.model_name = None
        self.model_version = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte il risultato in un dizionario.
        
        Returns:
            Dizionario rappresentante il risultato
        """
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "direction": self.direction,
            "confidence": self.confidence,
            "model_predictions": self.model_predictions,
            "probability_distribution": self.probability_distribution,
            "entry_price": self.entry_price,
            "tp_price": self.tp_price,
            "sl_price": self.sl_price,
            "risk_reward_ratio": self.risk_reward_ratio,
            "action": self.action,
            "order_type": self.order_type,
            "model_name": self.model_name,
            "model_version": self.model_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """
        Crea un'istanza di PredictionResult da un dizionario.
        
        Args:
            data: Dizionario con i dati
            
        Returns:
            Istanza di PredictionResult
        """
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
            
        prediction = cls(symbol, timeframe, timestamp)
        
        # Carica i campi dal dizionario
        if data.get("prediction_timestamp"):
            prediction.prediction_timestamp = datetime.fromisoformat(data["prediction_timestamp"])
        
        prediction.direction = data.get("direction", 0)
        prediction.confidence = data.get("confidence", 0.0)
        prediction.model_predictions = data.get("model_predictions", {})
        prediction.probability_distribution = data.get("probability_distribution", 
                                                     {"down": 0.0, "neutral": 0.0, "up": 0.0})
        prediction.entry_price = data.get("entry_price")
        prediction.tp_price = data.get("tp_price")
        prediction.sl_price = data.get("sl_price")
        prediction.risk_reward_ratio = data.get("risk_reward_ratio")
        prediction.action = data.get("action", "wait")
        prediction.order_type = data.get("order_type")
        prediction.model_name = data.get("model_name")
        prediction.model_version = data.get("model_version")
        
        return prediction


class PredictionEngine:
    """
    Motore principale per le previsioni di mercato.
    Coordina modelli, analisi tecnica e regole per generare previsioni.
    """
    
    def __init__(self, model: Optional[BaseModel] = None, 
                processor: Optional[DataProcessor] = None):
        """
        Inizializza il motore di previsione.
        
        Args:
            model: Modello principale (opzionale)
            processor: Processore di dati (opzionale)
        """
        # Componenti principali
        self.model = model
        self.processor = processor or DataProcessor()
        self.db_repository = get_default_repository()
        self.feature_names = []
        
        # Impostazioni
        self.confidence_threshold = 0.6  # Soglia di confidenza per le previsioni
        self.risk_reward_ratio_min = 1.5  # Rapporto rischio/rendimento minimo
        
        # Inizializza calcolatori di indicatori
        self.trend_indicators = TrendIndicators()
        self.momentum_indicators = MomentumIndicators()
        self.candle_patterns = CandlePatterns()
        
        app_logger.info("PredictionEngine inizializzato")
    
    def load_model(self, model_id: str) -> None:
        """
        Carica un modello dal registro.
        
        Args:
            model_id: ID del modello
            
        Raises:
            ModelError: Se il modello non può essere caricato
        """
        try:
            # Ottieni il modello dal registro
            self.model = model_registry.load_model_by_id(model_id)
            
            if self.model is None:
                raise ModelError(f"Modello {model_id} non trovato")
            
            # Carica i nomi delle feature dal modello
            self.feature_names = self.model.feature_names or []
            
            app_logger.info(f"Modello {model_id} caricato nel PredictionEngine")
        
        except Exception as e:
            error_msg = f"Errore nel caricamento del modello {model_id}: {str(e)}"
            app_logger.error(error_msg)
            raise ModelError(error_msg)
    
    def predict(self, symbol: str, timeframe: Optional[str] = None, 
               lookback: int = 100) -> PredictionResult:
        """
        Genera una previsione per un simbolo e timeframe.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe per la previsione (default: DEFAULT_PREDICTION_TIMEFRAME)
            lookback: Numero di barre storiche da utilizzare
            
        Returns:
            Risultato della previsione
            
        Raises:
            InferenceError: Se la previsione fallisce
        """
        try:
            # Valida il simbolo e il timeframe
            symbol_obj = get_symbol(symbol)
            if symbol_obj is None:
                raise ValueError(f"Simbolo non valido: {symbol}")
            
            if timeframe is None:
                timeframe = DEFAULT_PREDICTION_TIMEFRAME
            
            timeframe_obj = get_timeframe(timeframe)
            if timeframe_obj is None:
                raise ValueError(f"Timeframe non valido: {timeframe}")
            
            # Ottieni i dati storici
            df = self.db_repository.fetch_raw_data(
                symbol, timeframe, limit=lookback
            )
            
            if df.empty:
                raise ValueError(f"Dati insufficienti per {symbol} con timeframe {timeframe}")
            
            # Elabora i dati
            processed_df = self.processor.process_dataframe(df)
            
            # Crea il risultato di previsione
            result = PredictionResult(symbol, timeframe, df['timestamp'].iloc[-1])
            
            # Esegui la previsione con il modello se disponibile
            if self.model is not None:
                model_prediction = self._predict_with_model(processed_df)
                self._update_result_with_model_prediction(result, model_prediction, processed_df)
            
            # Aggiungi analisi tecnica
            self._enhance_with_technical_analysis(result, processed_df)
            
            # Calcola livelli di prezzo e rapporto rischio/rendimento
            self._calculate_price_levels(result, processed_df)
            
            # Determina l'azione consigliata
            self._determine_action(result)
            
            app_logger.info(f"Previsione generata per {symbol} con timeframe {timeframe}, "
                          f"direzione: {result.direction}, confidenza: {result.confidence:.2f}")
            
            return result
        
        except Exception as e:
            error_msg = f"Errore durante la previsione per {symbol} con timeframe {timeframe}: {str(e)}"
            app_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def _predict_with_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Effettua la previsione utilizzando il modello.
        
        Args:
            df: DataFrame elaborato con indicatori
            
        Returns:
            Dizionario con risultati della previsione del modello
        """
        if self.model is None:
            return {}
        
        try:
            # Prepara i dati di input
            if not self.feature_names:
                # Usa tutte le colonne numeriche se non sono specificate feature
                X = df.select_dtypes(include=['number']).values
            else:
                # Usa solo le feature specificate
                available_features = [f for f in self.feature_names if f in df.columns]
                if not available_features:
                    raise ValueError("Nessuna feature richiesta disponibile nel DataFrame")
                
                X = df[available_features].values
            
            # Reshape se necessario (per modelli sequenziali)
            if hasattr(self.model, 'config') and self.model.config.get('parameters', {}).get('input_shape'):
                input_shape = self.model.config['parameters']['input_shape']
                if len(input_shape) > 2:  # Modello sequenziale
                    # Prendi gli ultimi N record corrispondenti alla sequenza
                    seq_length = input_shape[0]
                    if len(X) < seq_length:
                        raise ValueError(f"Dati insufficienti per sequenza di lunghezza {seq_length}")
                    
                    X = X[-seq_length:].reshape(1, *input_shape[1:])
            else:
                # Per modelli non sequenziali, prendi solo l'ultima riga
                X = X[-1:].reshape(1, -1)
            
            # Effettua la previsione
            y_pred = self.model.predict(X)
            
            # Interpreta la previsione
            if y_pred.shape[1] > 1:  # Multi-classe (probabilities)
                probabilities = y_pred[0]
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
                
                # Converti classe in direzione (-1, 0, 1)
                if predicted_class == 0:
                    direction = -1  # Down
                elif predicted_class == 1:
                    direction = 0   # Neutral
                else:
                    direction = 1   # Up
                
                prediction = {
                    "direction": direction,
                    "confidence": confidence,
                    "raw_prediction": y_pred[0].tolist(),
                    "model_name": self.model.name,
                    "model_version": self.model.version
                }
            else:  # Regressione o classificazione binaria
                value = float(y_pred[0][0])
                
                # Interpreta come classificazione binaria o regressione normalizzata
                if -1.5 < value < 1.5:  # Probabile classificazione/regressione normalizzata
                    # Converti in direzione (-1, 0, 1) con zona neutra
                    if value < -0.3:
                        direction = -1  # Down
                        confidence = min(abs(value), 1.0)
                    elif value > 0.3:
                        direction = 1   # Up
                        confidence = min(abs(value), 1.0)
                    else:
                        direction = 0   # Neutral
                        confidence = 1.0 - min(abs(value) * 3, 0.9)  # Più vicino a 0, più confidenza nel neutrale
                else:  # Probabile previsione di prezzo
                    # In questo caso consideriamo il valore come una previsione di variazione percentuale
                    if value < -0.1:
                        direction = -1  # Down
                        confidence = min(abs(value) * 5, 1.0)  # Scala la confidenza
                    elif value > 0.1:
                        direction = 1   # Up
                        confidence = min(abs(value) * 5, 1.0)  # Scala la confidenza
                    else:
                        direction = 0   # Neutral
                        confidence = 0.7  # Confidenza media per previsioni vicine allo zero
                
                prediction = {
                    "direction": direction,
                    "confidence": confidence,
                    "raw_prediction": float(value),
                    "model_name": self.model.name,
                    "model_version": self.model.version
                }
            
            return prediction
        
        except Exception as e:
            app_logger.error(f"Errore nella previsione con modello: {e}")
            return {}
    
    def _update_result_with_model_prediction(self, result: PredictionResult, 
                                           model_prediction: Dict[str, Any],
                                           df: pd.DataFrame) -> None:
        """
        Aggiorna il risultato con la previsione del modello.
        
        Args:
            result: Risultato da aggiornare
            model_prediction: Previsione del modello
            df: DataFrame elaborato
        """
        if not model_prediction:
            return
        
        # Aggiorna il risultato con la previsione del modello
        result.direction = model_prediction.get("direction", 0)
        result.confidence = model_prediction.get("confidence", 0.0)
        result.model_name = model_prediction.get("model_name")
        result.model_version = model_prediction.get("model_version")
        
        # Se il modello è un ensemble, aggiungi dettagli per ogni modello componente
        if isinstance(self.model, EnsembleModel):
            result.model_predictions = {}
            for i, component_model in enumerate(self.model.models):
                result.model_predictions[component_model.name] = {
                    "weight": float(self.model.weights[i]) if self.model.weights is not None else 1.0 / len(self.model.models),
                    "confidence": model_prediction.get("component_confidences", {}).get(component_model.name, 0.0)
                }
        
        # Aggiorna la distribuzione di probabilità in base alla confidenza
        if result.direction == 1:  # Up
            result.probability_distribution = {
                "down": (1.0 - result.confidence) * 0.5,
                "neutral": (1.0 - result.confidence) * 0.5,
                "up": result.confidence
            }
        elif result.direction == -1:  # Down
            result.probability_distribution = {
                "down": result.confidence,
                "neutral": (1.0 - result.confidence) * 0.5,
                "up": (1.0 - result.confidence) * 0.5
            }
        else:  # Neutral
            result.probability_distribution = {
                "down": (1.0 - result.confidence) * 0.5,
                "neutral": result.confidence,
                "up": (1.0 - result.confidence) * 0.5
            }
    
    def _enhance_with_technical_analysis(self, result: PredictionResult, df: pd.DataFrame) -> None:
        """
        Migliora la previsione con analisi tecnica.
        
        Args:
            result: Risultato da migliorare
            df: DataFrame elaborato
        """
        if df.empty:
            return
        
        # Ottieni l'ultima riga di dati
        last_row = df.iloc[-1]
        
        # Analizza i trend
        trend_direction = 0
        trend_confidence = 0.0
        
        # Verifica trend con indicatori multipli
        
        # 1. Verifica Supertrend
        supertrend_present = all(c in df.columns for c in ['supertrend_direction', 'supertrend_uptrend', 'supertrend_downtrend'])
        if supertrend_present:
            if last_row['supertrend_uptrend']:
                trend_direction += 1
                trend_confidence += 0.5
            elif last_row['supertrend_downtrend']:
                trend_direction -= 1
                trend_confidence += 0.5
        
        # 2. Verifica Parabolic SAR
        psar_present = all(c in df.columns for c in ['psar_direction', 'psar_uptrend', 'psar_downtrend'])
        if psar_present:
            if last_row['psar_uptrend']:
                trend_direction += 1
                trend_confidence += 0.3
            elif last_row['psar_downtrend']:
                trend_direction -= 1
                trend_confidence += 0.3
        
        # 3. Verifica EMA
        ema_present = all(c in df.columns for c in ['ema_50', 'ema_200'])
        if ema_present:
            if last_row['ema_50'] > last_row['ema_200']:
                trend_direction += 1
                trend_confidence += 0.4
            elif last_row['ema_50'] < last_row['ema_200']:
                trend_direction -= 1
                trend_confidence += 0.4
        
        # 4. Verifica ADX per forza del trend
        adx_present = all(c in df.columns for c in ['adx', 'adx_bullish', 'adx_bearish'])
        if adx_present:
            # ADX > 25 indica un trend forte
            if last_row['adx'] > 25:
                trend_confidence += 0.2
                
                if last_row['adx_bullish']:
                    trend_direction += 1
                elif last_row['adx_bearish']:
                    trend_direction -= 1
        
        # Normalizza la direzione del trend e la confidenza
        if trend_direction != 0:
            trend_direction = 1 if trend_direction > 0 else -1
            trend_confidence = min(trend_confidence, 1.0)
            
            # Se non abbiamo una previsione dal modello o la confidenza è bassa
            if result.confidence < 0.5:
                # Aggiorna la previsione con il trend tecnico
                result.direction = trend_direction
                result.confidence = trend_confidence
                result.model_name = "Technical Analysis"
                result.model_version = "1.0"
                
                # Aggiorna la distribuzione di probabilità
                if trend_direction == 1:  # Up
                    result.probability_distribution = {
                        "down": (1.0 - trend_confidence) * 0.5,
                        "neutral": (1.0 - trend_confidence) * 0.5,
                        "up": trend_confidence
                    }
                else:  # Down
                    result.probability_distribution = {
                        "down": trend_confidence,
                        "neutral": (1.0 - trend_confidence) * 0.5,
                        "up": (1.0 - trend_confidence) * 0.5
                    }
            elif result.direction == trend_direction:
                # Se la previsione del modello concorda con il trend, aumenta la confidenza
                result.confidence = min(result.confidence + 0.1, 1.0)
            else:
                # Se la previsione del modello è in contrasto con il trend, riduci la confidenza
                result.confidence = max(result.confidence - 0.2, 0.0)
        
        # Analizza pattern di candele
        candle_patterns_present = any(col in df.columns for col in [
            'doji', 'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star'
        ])
        
        if candle_patterns_present:
            # Pattern rialzisti
            bullish_pattern = any([
                last_row.get('hammer', False),
                last_row.get('bullish_engulfing', False),
                last_row.get('morning_star', False),
                last_row.get('tweezer_bottom', False)
            ])
            
            # Pattern ribassisti
            bearish_pattern = any([
                last_row.get('shooting_star', False),
                last_row.get('bearish_engulfing', False),
                last_row.get('evening_star', False),
                last_row.get('tweezer_top', False)
            ])
            
            # Aggiusta la confidenza in base ai pattern
            if bullish_pattern and result.direction == 1:
                result.confidence = min(result.confidence + 0.15, 1.0)
            elif bearish_pattern and result.direction == -1:
                result.confidence = min(result.confidence + 0.15, 1.0)
            elif bullish_pattern and result.direction == -1:
                result.confidence = max(result.confidence - 0.15, 0.0)
            elif bearish_pattern and result.direction == 1:
                result.confidence = max(result.confidence - 0.15, 0.0)
    
    def _calculate_price_levels(self, result: PredictionResult, df: pd.DataFrame) -> None:
        """
        Calcola livelli di prezzo, take profit e stop loss.
        
        Args:
            result: Risultato da aggiornare
            df: DataFrame elaborato
        """
        if df.empty:
            return
        
        # Ottieni l'ultimo prezzo disponibile
        last_price = df['close'].iloc[-1]
        result.entry_price = last_price
        
        # Ottieni ATR per dimensionare stop loss e take profit
        atr = None
        atr_cols = [c for c in df.columns if c.startswith('atr_') and c.replace('atr_', '').isdigit()]
        if atr_cols:
            atr = df[atr_cols[0]].iloc[-1]
        else:
            # Calcola ATR se non presente
            df_temp = df.copy()
            df_temp['tr'] = np.maximum(
                df_temp['high'] - df_temp['low'],
                np.maximum(
                    np.abs(df_temp['high'] - df_temp['close'].shift()),
                    np.abs(df_temp['low'] - df_temp['close'].shift())
                )
            )
            atr = df_temp['tr'].rolling(window=14).mean().iloc[-1]
        
        if atr is None or np.isnan(atr):
            # Fallback: usa una percentuale del prezzo
            atr = last_price * 0.01  # 1% del prezzo
        
        # Calcola stop loss e take profit in base alla direzione prevista
        if result.direction == 1:  # Up
            # Per rialzo: SL sotto il prezzo attuale, TP sopra
            result.sl_price = last_price - (atr * 2)
            result.tp_price = last_price + (atr * 3)
        elif result.direction == -1:  # Down
            # Per ribasso: SL sopra il prezzo attuale, TP sotto
            result.sl_price = last_price + (atr * 2)
            result.tp_price = last_price - (atr * 3)
        else:  # Neutral
            result.sl_price = None
            result.tp_price = None
        
        # Calcola il rapporto rischio/rendimento
        if result.sl_price is not None and result.tp_price is not None:
            potential_profit = abs(result.tp_price - last_price)
            potential_loss = abs(result.sl_price - last_price)
            
            if potential_loss > 0:
                result.risk_reward_ratio = potential_profit / potential_loss
            else:
                result.risk_reward_ratio = 0
    
    def _determine_action(self, result: PredictionResult) -> None:
        """
        Determina l'azione consigliata in base alla previsione.
        
        Args:
            result: Risultato da aggiornare
        """
        # Default: wait
        result.action = "wait"
        result.order_type = None
        
        # Se la confidenza è sotto la soglia, attendi
        if result.confidence < self.confidence_threshold:
            return
        
        # Se il rapporto rischio/rendimento è inadeguato, attendi
        if result.risk_reward_ratio is not None and result.risk_reward_ratio < self.risk_reward_ratio_min:
            return
        
        # Determina l'azione in base alla direzione
        if result.direction == 1:  # Up
            result.action = "buy"
            result.order_type = "market"
        elif result.direction == -1:  # Down
            result.action = "sell"
            result.order_type = "market"
    
    def save_prediction(self, prediction: PredictionResult) -> bool:
        """
        Salva una previsione nel database.
        
        Args:
            prediction: Risultato della previsione
            
        Returns:
            True se il salvataggio ha successo
        """
        try:
            # Converti la previsione in dizionario
            prediction_dict = prediction.to_dict()
            
            # Salva nel database
            is_advanced = isinstance(self.model, EnsembleModel)
            success = self.db_repository.store_model_prediction(prediction_dict, advanced=is_advanced)
            
            if success:
                app_logger.info(f"Previsione salvata per {prediction.symbol} con timeframe {prediction.timeframe}")
            else:
                app_logger.warning(f"Impossibile salvare la previsione per {prediction.symbol}")
            
            return success
        
        except Exception as e:
            app_logger.error(f"Errore durante il salvataggio della previsione: {e}")
            return False
    
    def get_predictions(self, symbol: str, timeframe: str, 
                       limit: int = 10, is_advanced: bool = False) -> List[PredictionResult]:
        """
        Recupera le previsioni precedenti dal database.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe
            limit: Numero massimo di previsioni da recuperare
            is_advanced: Se recuperare previsioni avanzate
            
        Returns:
            Lista di oggetti PredictionResult
        """
        try:
            # Recupera previsioni dal database
            predictions_df = self.db_repository.fetch_model_predictions(
                symbol, timeframe, advanced=is_advanced, limit=limit
            )
            
            if predictions_df.empty:
                return []
            
            # Converti in oggetti PredictionResult
            results = []
            for _, row in predictions_df.iterrows():
                # Converti la riga in dizionario
                row_dict = row.to_dict()
                
                # Aggiungi distribuzione di probabilità se è una previsione avanzata
                if is_advanced:
                    row_dict["probability_distribution"] = {
                        "down": row.get("probability_down", 0),
                        "neutral": row.get("probability_neutral", 0),
                        "up": row.get("probability_up", 0)
                    }
                
                # Crea l'oggetto PredictionResult
                prediction = PredictionResult.from_dict(row_dict)
                results.append(prediction)
            
            app_logger.info(f"Recuperate {len(results)} previsioni per {symbol} con timeframe {timeframe}")
            return results
        
        except Exception as e:
            app_logger.error(f"Errore durante il recupero delle previsioni: {e}")
            return []


class MultiModelPredictionEngine(PredictionEngine):
    """
    Motore di previsione che utilizza più modelli.
    """
    
    def __init__(self, models: Optional[List[BaseModel]] = None, 
                weights: Optional[List[float]] = None,
                processor: Optional[DataProcessor] = None):
        """
        Inizializza il motore di previsione multi-modello.
        
        Args:
            models: Lista di modelli da utilizzare
            weights: Pesi per ciascun modello (opzionale)
            processor: Processore di dati (opzionale)
        """
        # Inizializza la classe base
        super().__init__(model=None, processor=processor)
        
        # Inizializza la lista dei modelli
        self.models = models or []
        
        # Valida e normalizza i pesi
        if weights is not None:
            if len(weights) != len(self.models):
                raise ValueError(f"Il numero di pesi ({len(weights)}) non corrisponde al numero di modelli ({len(self.models)})")
            
            # Normalizza i pesi per garantire che sommino a 1
            total_weight = sum(weights)
            if total_weight > 0:
                self.weights = [w / total_weight for w in weights]
            else:
                # Pesi uniformi se la somma è 0
                self.weights = [1.0 / len(self.models) for _ in self.models]
        else:
            # Pesi uniformi se non specificati
            self.weights = [1.0 / len(self.models) for _ in self.models] if self.models else []
        
        # Aggiungi dizionario per tenere traccia delle previsioni dei singoli modelli
        self.model_predictions = {}
        
        app_logger.info(f"MultiModelPredictionEngine inizializzato con {len(self.models)} modelli")
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """
        Aggiunge un modello al motore di previsione.
        
        Args:
            model: Modello da aggiungere
            weight: Peso del modello (default: 1.0)
        """
        if model is None:
            raise ValueError("Impossibile aggiungere un modello nullo")
        
        self.models.append(model)
        
        # Aggiungi il nuovo peso e normalizza
        total_weight = sum(self.weights) + weight
        self.weights = [(w / total_weight) for w in self.weights]
        self.weights.append(weight / total_weight)
        
        app_logger.info(f"Modello {model.name} (v{model.version}) aggiunto al MultiModelPredictionEngine "
                      f"con peso {weight / total_weight:.2f}")
    
    def remove_model(self, model_name: str) -> bool:
        """
        Rimuove un modello dal motore di previsione.
        
        Args:
            model_name: Nome del modello da rimuovere
            
        Returns:
            True se il modello è stato rimosso, False altrimenti
        """
        for i, model in enumerate(self.models):
            if model.name == model_name:
                # Rimuovi il modello
                self.models.pop(i)
                removed_weight = self.weights.pop(i)
                
                # Rinormalizza i pesi se necessario
                if self.models:
                    total_weight = sum(self.weights)
                    if total_weight > 0:
                        self.weights = [w / total_weight for w in self.weights]
                    else:
                        # Reset a pesi uniformi
                        self.weights = [1.0 / len(self.models) for _ in self.models]
                
                app_logger.info(f"Modello {model_name} rimosso dal MultiModelPredictionEngine")
                return True
        
        app_logger.warning(f"Modello {model_name} non trovato nel MultiModelPredictionEngine")
        return False
    
    def set_model_weight(self, model_name: str, weight: float) -> bool:
        """
        Imposta il peso di un modello specifico.
        
        Args:
            model_name: Nome del modello
            weight: Nuovo peso
            
        Returns:
            True se il peso è stato aggiornato, False altrimenti
        """
        if weight < 0:
            raise ValueError("Il peso non può essere negativo")
        
        for i, model in enumerate(self.models):
            if model.name == model_name:
                # Aggiorna il peso
                self.weights[i] = weight
                
                # Rinormalizza i pesi
                total_weight = sum(self.weights)
                if total_weight > 0:
                    self.weights = [w / total_weight for w in self.weights]
                else:
                    # Reset a pesi uniformi
                    self.weights = [1.0 / len(self.models) for _ in self.models]
                
                app_logger.info(f"Peso del modello {model_name} aggiornato a {self.weights[i]:.2f}")
                return True
        
        app_logger.warning(f"Modello {model_name} non trovato nel MultiModelPredictionEngine")
        return False
    
    def predict(self, symbol: str, timeframe: Optional[str] = None, 
               lookback: int = 100) -> PredictionResult:
        """
        Genera una previsione per un simbolo e timeframe utilizzando tutti i modelli.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe per la previsione (default: DEFAULT_PREDICTION_TIMEFRAME)
            lookback: Numero di barre storiche da utilizzare
            
        Returns:
            Risultato della previsione aggregata
            
        Raises:
            InferenceError: Se la previsione fallisce
        """
        try:
            # Valida il simbolo e il timeframe
            symbol_obj = get_symbol(symbol)
            if symbol_obj is None:
                raise ValueError(f"Simbolo non valido: {symbol}")
            
            if timeframe is None:
                timeframe = DEFAULT_PREDICTION_TIMEFRAME
            
            timeframe_obj = get_timeframe(timeframe)
            if timeframe_obj is None:
                raise ValueError(f"Timeframe non valido: {timeframe}")
            
            # Ottieni i dati storici
            df = self.db_repository.fetch_raw_data(
                symbol, timeframe, limit=lookback
            )
            
            if df.empty:
                raise ValueError(f"Dati insufficienti per {symbol} con timeframe {timeframe}")
            
            # Elabora i dati
            processed_df = self.processor.process_dataframe(df)
            
            # Crea il risultato di previsione
            result = PredictionResult(symbol, timeframe, df['timestamp'].iloc[-1])
            
            # Esegui le previsioni con tutti i modelli
            if not self.models:
                app_logger.warning("Nessun modello disponibile per la previsione")
            else:
                self._predict_with_multiple_models(processed_df, result)
            
            # Aggiungi analisi tecnica
            self._enhance_with_technical_analysis(result, processed_df)
            
            # Calcola livelli di prezzo e rapporto rischio/rendimento
            self._calculate_price_levels(result, processed_df)
            
            # Determina l'azione consigliata
            self._determine_action(result)
            
            app_logger.info(f"Previsione generata con MultiModelPredictionEngine per {symbol} con timeframe {timeframe}, "
                          f"direzione: {result.direction}, confidenza: {result.confidence:.2f}")
            
            return result
        
        except Exception as e:
            error_msg = f"Errore durante la previsione multi-modello per {symbol} con timeframe {timeframe}: {str(e)}"
            app_logger.error(error_msg)
            raise InferenceError(error_msg)
    
    def _predict_with_multiple_models(self, df: pd.DataFrame, result: PredictionResult) -> None:
        """
        Effettua previsioni utilizzando tutti i modelli disponibili e aggrega i risultati.
        
        Args:
            df: DataFrame elaborato con indicatori
            result: Risultato da aggiornare con la previsione aggregata
        """
        model_predictions = {}
        aggregated_direction = 0
        confidence_sum = 0
        
        # Ottieni le previsioni per ciascun modello
        for i, model in enumerate(self.models):
            try:
                # Salva temporaneamente il modello corrente per usare il metodo della classe base
                self.model = model
                model_pred = self._predict_with_model(df)
                self.model = None  # Resetta il modello
                
                if model_pred:
                    model_name = model.name
                    direction = model_pred.get("direction", 0)
                    confidence = model_pred.get("confidence", 0.0)
                    weight = self.weights[i]
                    
                    # Salva la previsione del modello
                    model_predictions[model_name] = {
                        "direction": direction,
                        "confidence": confidence,
                        "weight": weight,
                        "raw_prediction": model_pred.get("raw_prediction", None)
                    }
                    
                    # Aggrega la direzione ponderata per peso
                    aggregated_direction += direction * weight * confidence
                    confidence_sum += weight
            
            except Exception as e:
                app_logger.error(f"Errore nella previsione con il modello {model.name}: {str(e)}")
        
        # Calcola la direzione aggregata finale
        if confidence_sum > 0:
            # Normalizza per ottenere un valore tra -1 e 1
            normalized_direction = aggregated_direction / confidence_sum
            
            # Converti in direzione discreta (-1, 0, 1)
            if normalized_direction > 0.3:
                result.direction = 1
                result.confidence = min(abs(normalized_direction), 1.0)
            elif normalized_direction < -0.3:
                result.direction = -1
                result.confidence = min(abs(normalized_direction), 1.0)
            else:
                result.direction = 0
                result.confidence = 1.0 - min(abs(normalized_direction) * 3, 0.9)
        else:
            # Default se non ci sono modelli validi
            result.direction = 0
            result.confidence = 0.0
        
        # Aggiorna il risultato con le previsioni dei modelli
        result.model_predictions = model_predictions
        result.model_name = "MultiModel"
        result.model_version = "1.0"
        
        # Calcola la distribuzione di probabilità
        up_prob = 0.0
        down_prob = 0.0
        neutral_prob = 0.0
        
        for pred_info in model_predictions.values():
            weight = pred_info["weight"]
            confidence = pred_info["confidence"]
            direction = pred_info["direction"]
            
            if direction == 1:  # Up
                up_prob += weight * confidence
                neutral_prob += weight * (1 - confidence) * 0.5
                down_prob += weight * (1 - confidence) * 0.5
            elif direction == -1:  # Down
                down_prob += weight * confidence
                neutral_prob += weight * (1 - confidence) * 0.5
                up_prob += weight * (1 - confidence) * 0.5
            else:  # Neutral
                neutral_prob += weight * confidence
                up_prob += weight * (1 - confidence) * 0.5
                down_prob += weight * (1 - confidence) * 0.5
        
        # Normalizza le probabilità per assicurarsi che sommino a 1
        total_prob = up_prob + down_prob + neutral_prob
        if total_prob > 0:
            result.probability_distribution = {
                "up": up_prob / total_prob,
                "down": down_prob / total_prob,
                "neutral": neutral_prob / total_prob
            }