# Strategie di trading

"""
Implementazione di strategie di trading per il progetto TradingLab.
Questo modulo fornisce classi base e strategie concrete per il trading algoritmico.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

# Importazioni dai moduli di progetto
from ..analysis.indicators import TrendIndicators, MomentumIndicators, VolumeIndicators
from ..analysis.patterns import CandlePatterns, ChartPatterns
from ..utils import app_logger, StrategyError


class Strategy(ABC):
    """Classe base astratta per le strategie di trading."""
    
    def __init__(self, name: str = "Base Strategy"):
        """
        Inizializza una nuova strategia.
        
        Args:
            name: Nome della strategia
        """
        self.name = name
        self.description = "Strategia base astratta"
        self.parameters = {}
        
        # Metriche di performance
        self.performance = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
        
        app_logger.info(f"Strategia {name} inizializzata")
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali di trading sul DataFrame fornito.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading aggiunti
        """
        pass
    
    def set_parameters(self, **kwargs) -> None:
        """
        Imposta i parametri della strategia.
        
        Args:
            **kwargs: Parametri della strategia come coppie chiave-valore
        """
        self.parameters.update(kwargs)
        app_logger.info(f"Parametri aggiornati per {self.name}: {kwargs}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Ottiene i parametri della strategia.
        
        Returns:
            Dizionario con i parametri della strategia
        """
        return self.parameters
    
    def reset_performance(self) -> None:
        """Resetta le metriche di performance."""
        self.performance = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0
        }
    
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """
        Aggiorna le metriche di performance.
        
        Args:
            metrics: Dizionario con nuove metriche
        """
        self.performance.update(metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte la strategia in un dizionario.
        
        Returns:
            Dizionario con proprietà della strategia
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "performance": self.performance
        }


class MovingAverageCrossover(Strategy):
    """Strategia di crossover delle medie mobili."""
    
    def __init__(self, name: str = "Moving Average Crossover"):
        """
        Inizializza la strategia di crossover delle medie mobili.
        
        Args:
            name: Nome della strategia
        """
        super().__init__(name)
        self.description = "Genera segnali di acquisto/vendita in base al crossover di medie mobili"
        
        # Parametri predefiniti
        self.parameters = {
            "fast_period": 20,
            "slow_period": 50,
            "signal_column": "close",
            "ma_type": "ema"  # 'ema' o 'sma'
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali di trading basati su crossover di medie mobili.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading aggiunti
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Estrai parametri
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        signal_column = self.parameters["signal_column"]
        ma_type = self.parameters["ma_type"]
        
        # Verifica che la colonna del segnale esista
        if signal_column not in result.columns:
            raise StrategyError(f"Colonna {signal_column} non trovata nel DataFrame")
        
        # Calcola le medie mobili se non presenti
        fast_ma_col = f"{ma_type}_{fast_period}"
        slow_ma_col = f"{ma_type}_{slow_period}"
        
        if fast_ma_col not in result.columns:
            if ma_type == "ema":
                result[fast_ma_col] = result[signal_column].ewm(span=fast_period, adjust=False).mean()
            else:  # sma
                result[fast_ma_col] = result[signal_column].rolling(window=fast_period).mean()
        
        if slow_ma_col not in result.columns:
            if ma_type == "ema":
                result[slow_ma_col] = result[signal_column].ewm(span=slow_period, adjust=False).mean()
            else:  # sma
                result[slow_ma_col] = result[signal_column].rolling(window=slow_period).mean()
        
        # Genera segnali
        result['signal'] = 0
        
        # Crossover della media veloce sopra la media lenta (segnale di acquisto)
        result.loc[(result[fast_ma_col] > result[slow_ma_col]) & 
                 (result[fast_ma_col].shift(1) <= result[slow_ma_col].shift(1)), 'signal'] = 1
        
        # Crossover della media veloce sotto la media lenta (segnale di vendita)
        result.loc[(result[fast_ma_col] < result[slow_ma_col]) & 
                 (result[fast_ma_col].shift(1) >= result[slow_ma_col].shift(1)), 'signal'] = -1
        
        # Aggiungi informazioni di trend
        result['trend'] = np.where(result[fast_ma_col] > result[slow_ma_col], 1, -1)
        
        app_logger.info(f"Generati segnali di MA Crossover con fast_period={fast_period}, slow_period={slow_period}")
        return result


class RSIStrategy(Strategy):
    """Strategia basata sull'indicatore RSI (Relative Strength Index)."""
    
    def __init__(self, name: str = "RSI Strategy"):
        """
        Inizializza la strategia RSI.
        
        Args:
            name: Nome della strategia
        """
        super().__init__(name)
        self.description = "Genera segnali di acquisto/vendita basati sull'indicatore RSI"
        
        # Parametri predefiniti
        self.parameters = {
            "rsi_period": 14,
            "overbought_threshold": 70,
            "oversold_threshold": 30,
            "signal_column": "close"
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali di trading basati su RSI.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading aggiunti
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Estrai parametri
        rsi_period = self.parameters["rsi_period"]
        overbought = self.parameters["overbought_threshold"]
        oversold = self.parameters["oversold_threshold"]
        signal_column = self.parameters["signal_column"]
        
        # Verifica che la colonna del segnale esista
        if signal_column not in result.columns:
            raise StrategyError(f"Colonna {signal_column} non trovata nel DataFrame")
        
        # Verifica se RSI è già calcolato
        rsi_col = f"rsi_{rsi_period}"
        
        if rsi_col not in result.columns:
            # Calcola RSI
            momentum = MomentumIndicators()
            result = momentum.calculate_rsi(result, periods=[rsi_period])
        
        # Genera segnali
        result['signal'] = 0
        
        # Segnale di acquisto quando RSI esce dalla zona di ipervenduto
        result.loc[(result[rsi_col] > oversold) & 
                 (result[rsi_col].shift(1) <= oversold), 'signal'] = 1
        
        # Segnale di vendita quando RSI entra nella zona di ipercomprato
        result.loc[(result[rsi_col] < overbought) & 
                 (result[rsi_col].shift(1) >= overbought), 'signal'] = -1
        
        # Aggiungi informazioni di trend
        result['trend'] = np.where(result[rsi_col] > 50, 1, -1)
        
        app_logger.info(f"Generati segnali RSI con period={rsi_period}, overbought={overbought}, oversold={oversold}")
        return result


class SupertrendStrategy(Strategy):
    """Strategia basata sull'indicatore Supertrend."""
    
    def __init__(self, name: str = "Supertrend Strategy"):
        """
        Inizializza la strategia Supertrend.
        
        Args:
            name: Nome della strategia
        """
        super().__init__(name)
        self.description = "Genera segnali di acquisto/vendita basati sull'indicatore Supertrend"
        
        # Parametri predefiniti
        self.parameters = {
            "atr_period": 10,
            "multiplier": 3.0
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali di trading basati su Supertrend.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading aggiunti
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Estrai parametri
        atr_period = self.parameters["atr_period"]
        multiplier = self.parameters["multiplier"]
        
        # Verifica se Supertrend è già calcolato
        if 'supertrend_direction' not in result.columns:
            # Calcola Supertrend
            trends = TrendIndicators()
            result = trends.supertrend(result, atr_period=atr_period, multiplier=multiplier)
        
        # Genera segnali
        result['signal'] = 0
        
        # Segnale di acquisto quando Supertrend cambia da ribassista a rialzista
        result.loc[result['supertrend_buy_signal'], 'signal'] = 1
        
        # Segnale di vendita quando Supertrend cambia da rialzista a ribassista
        result.loc[result['supertrend_sell_signal'], 'signal'] = -1
        
        # Aggiungi informazioni di trend
        result['trend'] = result['supertrend_direction']
        
        app_logger.info(f"Generati segnali Supertrend con atr_period={atr_period}, multiplier={multiplier}")
        return result


class BBandRSIStrategy(Strategy):
    """Strategia combinata Bollinger Band e RSI."""
    
    def __init__(self, name: str = "BBand-RSI Strategy"):
        """
        Inizializza la strategia combinata Bollinger Band e RSI.
        
        Args:
            name: Nome della strategia
        """
        super().__init__(name)
        self.description = "Genera segnali combinando Bollinger Bands e RSI"
        
        # Parametri predefiniti
        self.parameters = {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "signal_column": "close"
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali di trading basati su Bollinger Bands e RSI.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading aggiunti
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Estrai parametri
        bb_period = self.parameters["bb_period"]
        bb_std = self.parameters["bb_std"]
        rsi_period = self.parameters["rsi_period"]
        rsi_overbought = self.parameters["rsi_overbought"]
        rsi_oversold = self.parameters["rsi_oversold"]
        signal_column = self.parameters["signal_column"]
        
        # Verifica che la colonna del segnale esista
        if signal_column not in result.columns:
            raise StrategyError(f"Colonna {signal_column} non trovata nel DataFrame")
        
        # Calcola Bollinger Bands se non presenti
        bb_middle_col = f"bb_middle_{bb_period}"
        bb_upper_col = f"bb_upper_{bb_period}"
        bb_lower_col = f"bb_lower_{bb_period}"
        
        if bb_middle_col not in result.columns:
            # Calcola BB
            result[bb_middle_col] = result[signal_column].rolling(window=bb_period).mean()
            std = result[signal_column].rolling(window=bb_period).std()
            result[bb_upper_col] = result[bb_middle_col] + (std * bb_std)
            result[bb_lower_col] = result[bb_middle_col] - (std * bb_std)
        
        # Calcola RSI se non presente
        rsi_col = f"rsi_{rsi_period}"
        
        if rsi_col not in result.columns:
            # Calcola RSI
            momentum = MomentumIndicators()
            result = momentum.calculate_rsi(result, periods=[rsi_period])
        
        # Genera segnali
        result['signal'] = 0
        
        # Segnale di acquisto: prezzo tocca la banda inferiore + RSI in ipervenduto
        result.loc[(result[signal_column] <= result[bb_lower_col]) & 
                 (result[rsi_col] < rsi_oversold), 'signal'] = 1
        
        # Segnale di vendita: prezzo tocca la banda superiore + RSI in ipercomprato
        result.loc[(result[signal_column] >= result[bb_upper_col]) & 
                 (result[rsi_col] > rsi_overbought), 'signal'] = -1
        
        # Aggiungi informazioni di trend
        result['trend'] = np.where(result[signal_column] > result[bb_middle_col], 1, -1)
        
        app_logger.info(f"Generati segnali BBand-RSI con bb_period={bb_period}, rsi_period={rsi_period}")
        return result


class PatternStrategy(Strategy):
    """Strategia basata sul riconoscimento di pattern."""
    
    def __init__(self, name: str = "Pattern Strategy"):
        """
        Inizializza la strategia basata su pattern.
        
        Args:
            name: Nome della strategia
        """
        super().__init__(name)
        self.description = "Genera segnali basati sul riconoscimento di pattern di candele"
        
        # Parametri predefiniti
        self.parameters = {
            "pattern_types": ["candle", "chart"],  # "candle", "chart", "harmonic"
            "confirmation_window": 1,  # Barre per confermare il pattern
            "use_volume_filter": True  # Filtra con volume
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali di trading basati sul riconoscimento di pattern.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading aggiunti
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Estrai parametri
        pattern_types = self.parameters["pattern_types"]
        confirmation_window = self.parameters["confirmation_window"]
        use_volume_filter = self.parameters["use_volume_filter"]
        
        # Calcola pattern se necessario
        
        # Pattern di candele
        if "candle" in pattern_types and not any(col.startswith("bullish_") or col.startswith("bearish_") for col in result.columns):
            candle_patterns = CandlePatterns()
            result = candle_patterns.recognizer(result)
        
        # Pattern di grafici
        if "chart" in pattern_types and not any(col in result.columns for col in ["head_and_shoulders", "double_top", "double_bottom"]):
            chart_patterns = ChartPatterns()
            result = chart_patterns.detect_head_and_shoulders(result)
            result = chart_patterns.detect_double_top_bottom(result)
        
        # Genera segnali
        result['signal'] = 0
        
        # Segnali rialzisti (pattern bullish)
        bullish_patterns = [c for c in result.columns if c.startswith("bullish_") or c in ["double_bottom", "inverse_head_and_shoulders"]]
        for pattern in bullish_patterns:
            if pattern in result.columns:
                # Applica filtro del volume se richiesto
                if use_volume_filter and 'volume' in result.columns:
                    avg_volume = result['volume'].rolling(window=20).mean()
                    volume_condition = result['volume'] > avg_volume
                    result.loc[result[pattern] & volume_condition, 'signal'] = 1
                else:
                    result.loc[result[pattern], 'signal'] = 1
        
        # Segnali ribassisti (pattern bearish)
        bearish_patterns = [c for c in result.columns if c.startswith("bearish_") or c in ["double_top", "head_and_shoulders"]]
        for pattern in bearish_patterns:
            if pattern in result.columns:
                # Applica filtro del volume se richiesto
                if use_volume_filter and 'volume' in result.columns:
                    avg_volume = result['volume'].rolling(window=20).mean()
                    volume_condition = result['volume'] > avg_volume
                    result.loc[result[pattern] & volume_condition, 'signal'] = -1
                else:
                    result.loc[result[pattern], 'signal'] = -1
        
        # Richiedi conferma se il parametro confirmation_window > 1
        if confirmation_window > 1:
            # Crea un segnale combinato che richiede pattern consecutivi
            result['temp_signal'] = result['signal'].copy()
            result['signal'] = 0
            
            for i in range(confirmation_window, len(result)):
                # Verifica se ci sono segnali coerenti nelle ultime N barre
                window = result['temp_signal'].iloc[i-confirmation_window+1:i+1]
                if (window > 0).all():
                    result.loc[result.index[i], 'signal'] = 1
                elif (window < 0).all():
                    result.loc[result.index[i], 'signal'] = -1
            
            # Rimuovi colonna temporanea
            result.drop('temp_signal', axis=1, inplace=True)
        
        app_logger.info(f"Generati segnali Pattern con tipi={pattern_types}")
        return result


class SmartMoneyStrategy(Strategy):
    """Strategia basata su concetti "Smart Money" (SMC)."""
    
    def __init__(self, name: str = "Smart Money Strategy"):
        """
        Inizializza la strategia Smart Money.
        
        Args:
            name: Nome della strategia
        """
        super().__init__(name)
        self.description = "Genera segnali basati sui concetti 'Smart Money' (SMC)"
        
        # Parametri predefiniti
        self.parameters = {
            "use_order_blocks": True,
            "use_fair_value_gaps": True,
            "use_liquidity_levels": True,
            "atr_multiple": 2.0  # Multiplo dell'ATR per definire stop loss
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali di trading basati su concetti SMC.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading aggiunti
        """
        if df.empty:
            return df
        
        result = df.copy()
        
        # Estrai parametri
        use_order_blocks = self.parameters["use_order_blocks"]
        use_fair_value_gaps = self.parameters["use_fair_value_gaps"]
        use_liquidity_levels = self.parameters["use_liquidity_levels"]
        atr_multiple = self.parameters["atr_multiple"]
        
        # Calcola indicatori SMC se necessario
        from ..analysis.indicators import MarketStructureIndicators
        
        market_structure = MarketStructureIndicators()
        
        # Calcola tutti gli indicatori necessari se non presenti
        smc_indicators_present = all(col in result.columns for col in [
            'bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
            'htf_liquidity_high', 'htf_liquidity_low'
        ])
        
        if not smc_indicators_present:
            result = market_structure.smart_money_concepts(result)
        
        # Calcola zone di domanda e offerta se non presenti
        demand_supply_present = all(col in result.columns for col in [
            'demand_zone_top', 'demand_zone_bottom', 'supply_zone_top', 'supply_zone_bottom'
        ])
        
        if not demand_supply_present:
            result = market_structure.detect_demand_supply_zones(result)
        
        # Genera segnali
        result['signal'] = 0
        
        # Order Blocks (OB)
        if use_order_blocks:
            # Aggiungi segnali di acquisto per Order Block rialzisti
            result.loc[result['bullish_ob'], 'signal'] = 1
            
            # Aggiungi segnali di vendita per Order Block ribassisti
            result.loc[result['bearish_ob'], 'signal'] = -1
        
        # Fair Value Gaps (FVG)
        if use_fair_value_gaps:
            # Aggiungi segnali per Fair Value Gaps
            if 'bullish_fvg' in result.columns and 'bearish_fvg' in result.columns:
                result.loc[result['bullish_fvg'], 'signal'] = 1
                result.loc[result['bearish_fvg'], 'signal'] = -1
        
        # Liquidity Levels
        if use_liquidity_levels:
            # Segnali per zone di liquidità
            if 'in_demand_zone' in result.columns:
                result.loc[result['in_demand_zone'], 'signal'] = 1
            
            if 'in_supply_zone' in result.columns:
                result.loc[result['in_supply_zone'], 'signal'] = -1
        
        # Imposta stop loss e take profit
        result['stop_loss'] = None
        result['take_profit'] = None
        
        # Calcola ATR se non presente
        atr_col = 'atr_14'
        if atr_col not in result.columns:
            result['tr'] = np.maximum(
                result['high'] - result['low'],
                np.maximum(
                    np.abs(result['high'] - result['close'].shift(1)),
                    np.abs(result['low'] - result['close'].shift(1))
                )
            )
            result[atr_col] = result['tr'].rolling(window=14).mean()
            result.drop('tr', axis=1, inplace=True)
        
        # Imposta stop loss e take profit basati su ATR
        for i in range(len(result)):
            if result['signal'].iloc[i] == 1:  # Segnale di acquisto
                atr = result[atr_col].iloc[i]
                result.loc[result.index[i], 'stop_loss'] = result['close'].iloc[i] - (atr * atr_multiple)
                result.loc[result.index[i], 'take_profit'] = result['close'].iloc[i] + (atr * atr_multiple * 1.5)
            elif result['signal'].iloc[i] == -1:  # Segnale di vendita
                atr = result[atr_col].iloc[i]
                result.loc[result.index[i], 'stop_loss'] = result['close'].iloc[i] + (atr * atr_multiple)
                result.loc[result.index[i], 'take_profit'] = result['close'].iloc[i] - (atr * atr_multiple * 1.5)
        
        app_logger.info(f"Generati segnali Smart Money con order_blocks={use_order_blocks}, fvg={use_fair_value_gaps}")
        return result


class CompositeStrategy(Strategy):
    """Strategia che combina più strategie."""
    
    def __init__(self, name: str = "Composite Strategy"):
        """
        Inizializza una strategia composita.
        
        Args:
            name: Nome della strategia
        """
        super().__init__(name)
        self.description = "Strategia che combina più strategie"
        
        # Parametri predefiniti
        self.parameters = {
            "combination_method": "majority",  # 'majority', 'unanimous', 'weighted'
            "weights": None  # Pesi per la combinazione (solo per 'weighted')
        }
        
        # Lista di strategie
        self.strategies: List[Strategy] = []
    
    def add_strategy(self, strategy: Strategy, weight: float = 1.0) -> None:
        """
        Aggiunge una strategia alla combinazione.
        
        Args:
            strategy: Strategia da aggiungere
            weight: Peso della strategia (solo per combinazione pesata)
        """
        self.strategies.append(strategy)
        
        # Aggiorna i pesi se necessario
        if self.parameters["combination_method"] == "weighted":
            weights = self.parameters["weights"] or []
            weights.append(weight)
            self.parameters["weights"] = weights
        
        app_logger.info(f"Strategia {strategy.name} aggiunta alla strategia composita {self.name}")
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """
        Rimuove una strategia dalla combinazione.
        
        Args:
            strategy_name: Nome della strategia da rimuovere
            
        Returns:
            True se la strategia è stata rimossa
        """
        for i, strategy in enumerate(self.strategies):
            if strategy.name == strategy_name:
                self.strategies.pop(i)
                
                # Aggiorna i pesi se necessario
                if self.parameters["combination_method"] == "weighted":
                    weights = self.parameters["weights"] or []
                    if i < len(weights):
                        weights.pop(i)
                        self.parameters["weights"] = weights
                
                app_logger.info(f"Strategia {strategy_name} rimossa dalla strategia composita {self.name}")
                return True
        
        return False
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera segnali combinando quelli delle strategie componenti.
        
        Args:
            df: DataFrame con dati di mercato
            
        Returns:
            DataFrame con segnali di trading combinati
        """
        if df.empty or not self.strategies:
            return df
        
        result = df.copy()
        combined_signals = pd.DataFrame(index=result.index)
        
        # Genera segnali per ogni strategia componente
        for i, strategy in enumerate(self.strategies):
            strategy_df = strategy.generate_signals(df)
            combined_signals[f"signal_{i}"] = strategy_df['signal']
        
        # Determina il metodo di combinazione
        combination_method = self.parameters["combination_method"]
        
        # Combina i segnali in base al metodo scelto
        if combination_method == "unanimous":
            # Segnale di acquisto solo se tutte le strategie concordano
            result['signal'] = np.where(
                (combined_signals > 0).all(axis=1), 1,
                np.where((combined_signals < 0).all(axis=1), -1, 0)
            )
        
        elif combination_method == "weighted":
            # Combinazione pesata dei segnali
            weights = self.parameters["weights"]
            if not weights or len(weights) != len(self.strategies):
                weights = [1.0] * len(self.strategies)
            
            # Moltiplica ogni segnale per il suo peso
            weighted_sum = pd.Series(0, index=result.index)
            for i, weight in enumerate(weights):
                weighted_sum += combined_signals[f"signal_{i}"] * weight
            
            # Normalizza per il numero di strategie e arrotonda
            threshold = sum(weights) * 0.5
            result['signal'] = np.where(
                weighted_sum > threshold, 1,
                np.where(weighted_sum < -threshold, -1, 0)
            )
        
        else:  # "majority" (default)
            # Segnale basato sulla maggioranza
            signal_sum = combined_signals.sum(axis=1)
            result['signal'] = np.where(
                signal_sum > 0, 1,
                np.where(signal_sum < 0, -1, 0)
            )
        
        # Aggiorna il trend come la media dei trend delle strategie componenti
        if 'trend' in result.columns:
            trend_columns = []
            for i, strategy in enumerate(self.strategies):
                strategy_df = strategy.generate_signals(df)
                if 'trend' in strategy_df.columns:
                    trend_columns.append(strategy_df['trend'])
            
            if trend_columns:
                result['trend'] = pd.concat(trend_columns, axis=1).mean(axis=1).apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )
        
        app_logger.info(f"Generati segnali compositi con metodo {combination_method} da {len(self.strategies)} strategie")
        return result
    
    def get_strategies(self) -> List[Dict[str, Any]]:
        """
        Ottiene le informazioni sulle strategie componenti.
        
        Returns:
            Lista di dizionari con informazioni sulle strategie
        """
        return [strategy.to_dict() for strategy in self.strategies]
    
    def reset(self) -> None:
        """Resetta tutte le strategie componenti."""
        for strategy in self.strategies:
            strategy.reset_performance()
        self.reset_performance()
        app_logger.info(f"Reset di tutte le strategie in {self.name}")


class StrategyFactory:
    """Factory per creare istanze di strategie."""
    
    _strategies = {
        "moving_average_crossover": MovingAverageCrossover,
        "rsi": RSIStrategy,
        "supertrend": SupertrendStrategy,
        "bband_rsi": BBandRSIStrategy,
        "pattern": PatternStrategy,
        "smart_money": SmartMoneyStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_type: str, **kwargs) -> Strategy:
        """
        Crea una nuova istanza di strategia.
        
        Args:
            strategy_type: Tipo di strategia
            **kwargs: Parametri aggiuntivi per la strategia
            
        Returns:
            Istanza della strategia
            
        Raises:
            StrategyError: Se il tipo di strategia non è supportato
        """
        if strategy_type not in cls._strategies:
            raise StrategyError(f"Tipo di strategia non supportato: {strategy_type}")
        
        strategy_class = cls._strategies[strategy_type]
        strategy = strategy_class()
        
        # Imposta parametri aggiuntivi
        if kwargs:
            strategy.set_parameters(**kwargs)
        
        return strategy
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: type) -> None:
        """
        Registra una nuova classe di strategia.
        
        Args:
            strategy_type: Identificatore della strategia
            strategy_class: Classe della strategia
        """
        if not issubclass(strategy_class, Strategy):
            raise TypeError("La classe deve essere una sottoclasse di Strategy")
        
        cls._strategies[strategy_type] = strategy_class
        app_logger.info(f"Strategia {strategy_type} registrata")
    
    @classmethod
    def get_available_strategies(cls) -> Dict[str, str]:
        """
        Ottiene le strategie disponibili.
        
        Returns:
            Dizionario di {type: description}
        """
        return {
            strategy_type: strategy_class().description
            for strategy_type, strategy_class in cls._strategies.items()
        }


def backtest_strategy(strategy: Strategy, df: pd.DataFrame, initial_capital: float = 10000.0, 
                     commission: float = 0.0, slippage: float = 0.0, position_size: Optional[float] = None) -> Dict[str, Any]:
    """
    Esegue un backtest della strategia sui dati forniti.
    
    Args:
        strategy: Strategia da testare
        df: DataFrame con dati di mercato
        initial_capital: Capitale iniziale
        commission: Commissione percentuale per trade
        slippage: Slippage percentuale per trade
        position_size: Percentuale del capitale da utilizzare per ogni trade (se None, usa tutto)
        
    Returns:
        Dizionario con i risultati del backtest
    """
    # Genera segnali
    result_df = strategy.generate_signals(df)
    
    # Inizializza le variabili del backtest
    capital = initial_capital
    position = 0
    entry_price = 0.0
    trades = []
    equity_curve = []
    drawdown = []
    max_capital = initial_capital
    
    # Esegui il backtest
    for i in range(1, len(result_df)):
        date = result_df.index[i]
        signal = result_df['signal'].iloc[i]
        close = result_df['close'].iloc[i]
        
        # Calcola equity corrente
        current_equity = capital + position * close
        equity_curve.append((date, current_equity))
        
        # Calcola drawdown
        if current_equity > max_capital:
            max_capital = current_equity
        current_drawdown = (max_capital - current_equity) / max_capital * 100
        drawdown.append((date, current_drawdown))
        
        # Gestione del segnale
        if signal == 1 and position == 0:  # Segnale di acquisto
            # Calcola capitale da investire
            trade_capital = capital if position_size is None else capital * position_size
            
            # Calcola costi
            commission_cost = trade_capital * commission / 100
            slippage_cost = trade_capital * slippage / 100
            
            # Calcola posizione effettiva
            effective_capital = trade_capital - commission_cost - slippage_cost
            position = effective_capital / close
            
            entry_price = close
            capital -= effective_capital
            
            trades.append({
                'type': 'buy',
                'date': date,
                'price': close,
                'position': position,
                'cost': commission_cost + slippage_cost
            })
            
        elif signal == -1 and position > 0:  # Segnale di vendita
            # Calcola valore di vendita
            sale_value = position * close
            
            # Calcola costi
            commission_cost = sale_value * commission / 100
            slippage_cost = sale_value * slippage / 100
            
            # Calcola capitale effettivo restituito
            effective_value = sale_value - commission_cost - slippage_cost
            
            # Calcola profitto
            profit = effective_value - (position * entry_price)
            
            capital += effective_value
            position = 0
            
            trades.append({
                'type': 'sell',
                'date': date,
                'price': close,
                'profit': profit,
                'cost': commission_cost + slippage_cost
            })
    
    # Chiudi la posizione finale
    if position > 0:
        # Calcola valore di vendita
        sale_value = position * result_df['close'].iloc[-1]
        
        # Calcola costi
        commission_cost = sale_value * commission / 100
        slippage_cost = sale_value * slippage / 100
        
        # Calcola capitale effettivo restituito
        effective_value = sale_value - commission_cost - slippage_cost
        
        # Calcola profitto
        profit = effective_value - (position * entry_price)
        
        capital += effective_value
        
        trades.append({
            'type': 'close',
            'date': result_df.index[-1],
            'price': result_df['close'].iloc[-1],
            'profit': profit,
            'cost': commission_cost + slippage_cost
        })
    
    # Calcola le metriche di performance
    final_equity = capital
    total_return = (final_equity / initial_capital - 1) * 100
    
    # Calcola profitti/perdite
    win_trades = [t for t in trades if t.get('profit', 0) > 0]
    lose_trades = [t for t in trades if t.get('profit', 0) < 0]
    
    # Win rate
    win_rate = len(win_trades) / max(1, len(trades) - 1) * 100
    
    # Calcola profit factor
    total_profit = sum(t.get('profit', 0) for t in win_trades)
    total_loss = abs(sum(t.get('profit', 0) for t in lose_trades))
    profit_factor = total_profit / max(1, total_loss)
    
    # Calcola max drawdown
    max_drawdown = max([dd[1] for dd in drawdown]) if drawdown else 0
    
    # Calcola Sharpe Ratio (approssimativo)
    if len(equity_curve) > 1:
        returns = [(equity_curve[i][1] / equity_curve[i-1][1]) - 1 for i in range(1, len(equity_curve))]
        avg_return = sum(returns) / len(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.0001
        sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Aggiorna le metriche di performance nella strategia
    strategy.update_performance({
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "total_trades": len(trades) - 1,
        "winning_trades": len(win_trades),
        "losing_trades": len(lose_trades),
        "total_return": total_return,
        "final_equity": final_equity
    })
    
    app_logger.info(f"Backtest completato per {strategy.name}: Win Rate={win_rate:.2f}%, Profit Factor={profit_factor:.2f}")
    
    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown,
        "metrics": strategy.performance
    }


def optimize_strategy(strategy: Strategy, df: pd.DataFrame, param_grid: Dict[str, List[Any]], 
                     metric: str = "total_return", initial_capital: float = 10000.0) -> Dict[str, Any]:
    """
    Ottimizza i parametri della strategia utilizzando grid search.
    
    Args:
        strategy: Strategia da ottimizzare
        df: DataFrame con dati di mercato
        param_grid: Griglia di parametri da testare (dict di liste)
        metric: Metrica da ottimizzare
        initial_capital: Capitale iniziale per i backtest
        
    Returns:
        Dizionario con i risultati dell'ottimizzazione
    """
    import itertools
    
    app_logger.info(f"Avvio ottimizzazione per {strategy.name} sulla metrica {metric}")
    
    # Genera tutte le combinazioni di parametri
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    best_params = None
    best_metric_value = -float('inf')
    results = []
    
    total_combinations = len(param_combinations)
    app_logger.info(f"Testando {total_combinations} combinazioni di parametri")
    
    # Esegui backtest per ogni combinazione
    for i, combination in enumerate(param_combinations):
        # Crea dizionario dei parametri correnti
        current_params = dict(zip(param_names, combination))
        
        # Applica parametri alla strategia
        strategy.set_parameters(**current_params)
        
        # Esegui backtest
        backtest_result = backtest_strategy(strategy, df, initial_capital=initial_capital)
        metrics = backtest_result["metrics"]
        
        # Verifica che la metrica esista
        if metric not in metrics:
            raise StrategyError(f"Metrica {metric} non trovata nei risultati del backtest")
        
        # Valuta metrica
        metric_value = metrics[metric]
        
        # Inverte segno per metriche che devono essere minimizzate
        if metric == "max_drawdown":
            metric_value = -metric_value
        
        # Salva risultato
        results.append({
            "params": current_params,
            "metrics": metrics
        })
        
        # Aggiorna miglior risultato
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_params = current_params
        
        # Log di progresso
        if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
            app_logger.info(f"Progresso: {i + 1}/{total_combinations} combinazioni testate")
    
    # Applica i migliori parametri trovati
    strategy.set_parameters(**best_params)
    
    app_logger.info(f"Ottimizzazione completata. Migliori parametri: {best_params}")
    
    return {
        "best_params": best_params,
        "best_metrics": backtest_strategy(strategy, df, initial_capital=initial_capital)["metrics"],
        "all_results": results
    }


# Aggiungi la strategia composita al factory
StrategyFactory._strategies["composite"] = CompositeStrategy