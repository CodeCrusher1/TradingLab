# Elaborazione dei dati e indicatori

"""
Modulo per l'elaborazione dei dati nel progetto TradingLab.
Fornisce funzioni e classi per calcolare indicatori tecnici e pattern di mercato.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta

# Importazioni dal modulo config
from ..config import INDICATOR_PARAMS, get_symbol, get_timeframe

# Importazioni dal modulo utils
from ..utils import (
    app_logger, data_logger, ProcessingError,
    time_it, handle_exception
)

# Importazioni dal modulo validators
from .validators import MarketDataValidator


class TechnicalIndicators:
    """
    Classe per il calcolo di indicatori tecnici.
    """
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calcola EMA (Exponential Moving Average) per diverse finestre temporali.
        
        Args:
            df: DataFrame con colonna 'close'
            periods: Lista di periodi per cui calcolare l'EMA
            
        Returns:
            DataFrame con colonne EMA aggiunte
        """
        if 'close' not in df.columns:
            raise ProcessingError("La colonna 'close' è richiesta per calcolare EMA")
        
        result = df.copy()
        
        if periods is None:
            periods = INDICATOR_PARAMS.get('ema', [21, 50, 200])
        
        for period in periods:
            column_name = f'ema_{period}'
            result[column_name] = result['close'].ewm(span=period, adjust=False).mean()
        
        return result
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calcola SMA (Simple Moving Average) per diverse finestre temporali.
        
        Args:
            df: DataFrame con colonna 'close'
            periods: Lista di periodi per cui calcolare la SMA
            
        Returns:
            DataFrame con colonne SMA aggiunte
        """
        if 'close' not in df.columns:
            raise ProcessingError("La colonna 'close' è richiesta per calcolare SMA")
        
        result = df.copy()
        
        if periods is None:
            periods = [20, 50, 200]
        
        for period in periods:
            column_name = f'sma_{period}'
            result[column_name] = result['close'].rolling(period).mean()
        
        return result
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calcola RSI (Relative Strength Index) per diverse finestre temporali.
        
        Args:
            df: DataFrame con colonna 'close'
            periods: Lista di periodi per cui calcolare l'RSI
            
        Returns:
            DataFrame con colonne RSI aggiunte
        """
        if 'close' not in df.columns:
            raise ProcessingError("La colonna 'close' è richiesta per calcolare RSI")
        
        result = df.copy()
        
        if periods is None:
            periods = INDICATOR_PARAMS.get('rsi', [14, 21, 50])
        
        for period in periods:
            # Calcola la variazione
            delta = result['close'].diff()
            
            # Separa guadagni e perdite
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calcola medie esponenziali di guadagni e perdite
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            
            # Calcola RS e RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            column_name = f'rsi_{period}'
            result[column_name] = rsi
        
        return result
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calcola ATR (Average True Range) per diverse finestre temporali.
        
        Args:
            df: DataFrame con colonne 'high', 'low', 'close'
            periods: Lista di periodi per cui calcolare l'ATR
            
        Returns:
            DataFrame con colonne ATR aggiunte
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per calcolare ATR")
        
        result = df.copy()
        
        if periods is None:
            periods = INDICATOR_PARAMS.get('atr', [14, 21, 50])
        
        # Calcola il True Range
        result['tr'] = np.maximum(
            result['high'] - result['low'],
            np.maximum(
                np.abs(result['high'] - result['close'].shift()),
                np.abs(result['low'] - result['close'].shift())
            )
        )
        
        for period in periods:
            # Calcola l'ATR come media esponenziale del True Range
            column_name = f'atr_{period}'
            # Usa rolling per evitare bias iniziale e poi ewm per vera ATR
            # I primi `period` valori sono calcolati con SMA per inizializzare ATR
            result[column_name] = result['tr'].rolling(window=period).mean()
            # I valori successivi sono calcolati con EMA per l'effettivo ATR
            first_atr = result[column_name].iloc[period-1]
            for i in range(period, len(result)):
                current_tr = result['tr'].iloc[i]
                prev_atr = result[column_name].iloc[i-1]
                result[column_name].iloc[i] = (prev_atr * (period - 1) + current_tr) / period
            
            # Aggiungi anche versione normalizzata (ATR diviso per prezzo)
            result[f'atr_{period}_normalized'] = result[column_name] / result['close']
        
        # Rimuovi la colonna TR temporanea
        result.drop('tr', axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Calcola le Bande di Bollinger.
        
        Args:
            df: DataFrame con colonna 'close'
            period: Periodo per la media mobile
            num_std: Numero di deviazioni standard
            
        Returns:
            DataFrame con colonne per le Bande di Bollinger
        """
        if 'close' not in df.columns:
            raise ProcessingError("La colonna 'close' è richiesta per calcolare le Bande di Bollinger")
        
        result = df.copy()
        
        # Calcola la SMA
        result[f'bb_middle_{period}'] = result['close'].rolling(period).mean()
        
        # Calcola la deviazione standard
        std = result['close'].rolling(period).std()
        
        # Calcola banda superiore e inferiore
        result[f'bb_upper_{period}'] = result[f'bb_middle_{period}'] + (std * num_std)
        result[f'bb_lower_{period}'] = result[f'bb_middle_{period}'] - (std * num_std)
        
        # Larghezza delle bande (utile per identificare compressione della volatilità)
        result[f'bb_width_{period}'] = (result[f'bb_upper_{period}'] - result[f'bb_lower_{period}']) / result[f'bb_middle_{period}']
        
        return result
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calcola MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame con colonna 'close'
            fast_period: Periodo per EMA veloce
            slow_period: Periodo per EMA lenta
            signal_period: Periodo per la linea di segnale
            
        Returns:
            DataFrame con colonne MACD aggiunte
        """
        if 'close' not in df.columns:
            raise ProcessingError("La colonna 'close' è richiesta per calcolare MACD")
        
        result = df.copy()
        
        # Calcola EMA veloce e lenta
        fast_ema = result['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calcola MACD e linea di segnale
        result['macd'] = fast_ema - slow_ema
        result['macd_signal'] = result['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calcola istogramma MACD
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        return result
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, 
                            d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
        """
        Calcola l'oscillatore stocastico.
        
        Args:
            df: DataFrame con colonne 'high', 'low', 'close'
            k_period: Periodo per %K
            d_period: Periodo per %D
            slowing: Periodo per rallentare %K
            
        Returns:
            DataFrame con colonne stocastiche aggiunte
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per calcolare l'oscillatore stocastico")
        
        result = df.copy()
        
        # Calcola il minimo e il massimo nel periodo
        low_min = result['low'].rolling(window=k_period).min()
        high_max = result['high'].rolling(window=k_period).max()
        
        # Calcola %K grezzo
        raw_k = 100 * ((result['close'] - low_min) / (high_max - low_min))
        
        # Applica rallentamento a %K
        result[f'stoch_{k_period}'] = raw_k.rolling(window=slowing).mean()
        
        # Calcola %D come media mobile di %K
        result[f'stoch_signal_{k_period}'] = result[f'stoch_{k_period}'].rolling(window=d_period).mean()
        
        return result
    
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, 
                          kijun_period: int = 26, senkou_b_period: int = 52, 
                          displacement: int = 26) -> pd.DataFrame:
        """
        Calcola l'indicatore Ichimoku Cloud.
        
        Args:
            df: DataFrame con colonne 'high', 'low', 'close'
            tenkan_period: Periodo per Tenkan-sen (linea di conversione)
            kijun_period: Periodo per Kijun-sen (linea base)
            senkou_b_period: Periodo per Senkou Span B
            displacement: Periodo di spostamento per Kumo (nuvola)
            
        Returns:
            DataFrame con colonne Ichimoku aggiunte
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per calcolare Ichimoku")
        
        result = df.copy()
        
        # Calcola Tenkan-sen (linea di conversione)
        high_tenkan = result['high'].rolling(window=tenkan_period).max()
        low_tenkan = result['low'].rolling(window=tenkan_period).min()
        result['ichimoku_tenkan'] = (high_tenkan + low_tenkan) / 2
        
        # Calcola Kijun-sen (linea base)
        high_kijun = result['high'].rolling(window=kijun_period).max()
        low_kijun = result['low'].rolling(window=kijun_period).min()
        result['ichimoku_kijun'] = (high_kijun + low_kijun) / 2
        
        # Calcola Senkou Span A (prima linea della nuvola)
        result['ichimoku_senkou_a'] = ((result['ichimoku_tenkan'] + result['ichimoku_kijun']) / 2).shift(displacement)
        
        # Calcola Senkou Span B (seconda linea della nuvola)
        high_senkou = result['high'].rolling(window=senkou_b_period).max()
        low_senkou = result['low'].rolling(window=senkou_b_period).min()
        result['ichimoku_senkou_b'] = ((high_senkou + low_senkou) / 2).shift(displacement)
        
        # Calcola Chikou Span (linea di ritardo)
        result['ichimoku_chikou'] = result['close'].shift(-displacement)
        
        return result
    
    @staticmethod
    def calculate_premium_discount(df: pd.DataFrame, reference_column: str = 'ema_50') -> pd.DataFrame:
        """
        Calcola il premium/discount percentuale rispetto a una media mobile o altro riferimento.
        
        Args:
            df: DataFrame con colonna 'close' e la colonna di riferimento
            reference_column: Colonna da usare come riferimento
            
        Returns:
            DataFrame con colonna premium_discount aggiunta
        """
        required_columns = ['close', reference_column]
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per calcolare premium/discount")
        
        result = df.copy()
        
        # Calcola il premium/discount come percentuale
        result['premium_discount'] = ((result['close'] - result[reference_column]) / result[reference_column]) * 100
        
        return result
    
    @staticmethod
    def calculate_volume_indicators(df: pd.DataFrame, periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Calcola indicatori basati sul volume.
        
        Args:
            df: DataFrame con colonne 'close', 'volume'
            periods: Lista di periodi per le medie mobili del volume
            
        Returns:
            DataFrame con indicatori di volume aggiunti
        """
        required_columns = ['close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per calcolare indicatori di volume")
        
        result = df.copy()
        
        if periods is None:
            periods = [10, 20, 50]
        
        # Media mobile del volume
        for period in periods:
            result[f'volume_sma_{period}'] = result['volume'].rolling(window=period).mean()
        
        # Volume relativo (rispetto alla media a 20 periodi)
        result['volume_ratio_20'] = result['volume'] / result['volume_sma_20']
        
        # Money Flow Index (MFI) - 14 periodi
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        raw_money_flow = typical_price * result['volume']
        
        delta = typical_price.diff()
        positive_flow = np.where(delta > 0, raw_money_flow, 0)
        negative_flow = np.where(delta < 0, raw_money_flow, 0)
        
        period = 14
        positive_mf = pd.Series(positive_flow).rolling(period).sum()
        negative_mf = pd.Series(negative_flow).rolling(period).sum()
        
        money_ratio = positive_mf / negative_mf
        result['mfi_14'] = 100 - (100 / (1 + money_ratio))
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(result)):
            if result['close'].iloc[i] > result['close'].iloc[i-1]:
                obv.append(obv[-1] + result['volume'].iloc[i])
            elif result['close'].iloc[i] < result['close'].iloc[i-1]:
                obv.append(obv[-1] - result['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        result['obv'] = obv
        
        return result


class MarketPatterns:
    """
    Classe per il riconoscimento di pattern di mercato.
    """
    
    @staticmethod
    def detect_fair_value_gaps(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
        """
        Rileva Fair Value Gaps (FVG).
        
        Args:
            df: DataFrame con colonne 'high', 'low'
            lookback: Numero di candele da considerare
            
        Returns:
            DataFrame con colonna 'fair_value_gap' aggiunta
        """
        required_columns = ['high', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per rilevare Fair Value Gaps")
        
        result = df.copy()
        result['fair_value_gap'] = False
        
        # Dobbiamo avere almeno lookback+1 candele
        if len(result) <= lookback:
            return result
        
        for i in range(lookback, len(result)):
            # Ottieni il massimo high delle ultime lookback candele
            past_highs = result['high'].iloc[i-lookback:i].max()
            
            # Ottieni il minimo low delle ultime lookback candele
            past_lows = result['low'].iloc[i-lookback:i].min()
            
            # Verifica le condizioni per FVG (gap rispetto ai massimi o minimi precedenti)
            bullish_gap = result['low'].iloc[i] > past_highs
            bearish_gap = result['high'].iloc[i] < past_lows
            
            # Segnala FVG se una delle condizioni è soddisfatta
            result.loc[result.index[i], 'fair_value_gap'] = bullish_gap or bearish_gap
        
        return result
    
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """
        Rileva Order Blocks (OB) secondo la metodologia ICT.
        
        Args:
            df: DataFrame con colonne 'open', 'close', 'high', 'low'
            lookback: Numero di candele da considerare
            
        Returns:
            DataFrame con colonne 'order_block_bullish' e 'order_block_bearish' aggiunte
        """
        required_columns = ['open', 'close', 'high', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per rilevare Order Blocks")
        
        result = df.copy()
        result['order_block_bullish'] = False
        result['order_block_bearish'] = False
        
        # Dobbiamo avere almeno lookback+1 candele
        if len(result) <= lookback:
            return result
        
        for i in range(lookback, len(result)):
            # Esamina le ultime lookback candele per bullish OB (candele rosse prima di un movimento al rialzo)
            if result['close'].iloc[i] > result['high'].iloc[i-lookback:i].max():
                # Cerca candele rosse (bearish) nelle ultime lookback candele
                for j in range(i-lookback, i):
                    if result['close'].iloc[j] < result['open'].iloc[j]:
                        result.loc[result.index[j], 'order_block_bullish'] = True
            
            # Esamina le ultime lookback candele per bearish OB (candele verdi prima di un movimento al ribasso)
            if result['close'].iloc[i] < result['low'].iloc[i-lookback:i].min():
                # Cerca candele verdi (bullish) nelle ultime lookback candele
                for j in range(i-lookback, i):
                    if result['close'].iloc[j] > result['open'].iloc[j]:
                        result.loc[result.index[j], 'order_block_bearish'] = True
        
        return result
    
    @staticmethod
    def detect_breaker_blocks(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """
        Rileva Breaker Blocks secondo la metodologia ICT.
        
        Args:
            df: DataFrame con colonne 'open', 'close', 'high', 'low'
            lookback: Numero di candele da considerare
            
        Returns:
            DataFrame con colonna 'breaker_block' aggiunta
        """
        required_columns = ['open', 'close', 'high', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per rilevare Breaker Blocks")
        
        result = df.copy()
        result['breaker_block'] = False
        
        # Dobbiamo avere almeno lookback+1 candele
        if len(result) <= lookback:
            return result
        
        for i in range(lookback, len(result)):
            # Esamina le ultime lookback candele per trovare Order Blocks
            for j in range(i-lookback, i):
                # Se abbiamo identificato un bullish Order Block
                if result.loc[result.index[j], 'order_block_bullish']:
                    # Verifica se il prezzo è tornato indietro e ha rotto sotto il Block
                    if result['low'].iloc[i] < result['low'].iloc[j]:
                        result.loc[result.index[i], 'breaker_block'] = True
                
                # Se abbiamo identificato un bearish Order Block
                elif result.loc[result.index[j], 'order_block_bearish']:
                    # Verifica se il prezzo è tornato indietro e ha rotto sopra il Block
                    if result['high'].iloc[i] > result['high'].iloc[j]:
                        result.loc[result.index[i], 'breaker_block'] = True
        
        return result
    
    @staticmethod
    def detect_liquidity_levels(df: pd.DataFrame, n_bars: int = 5) -> pd.DataFrame:
        """
        Rileva livelli di liquidità (swing high/low).
        
        Args:
            df: DataFrame con colonne 'high', 'low'
            n_bars: Numero di barre per definire uno swing
            
        Returns:
            DataFrame con colonne 'liquidity_level_above' e 'liquidity_level_below' aggiunte
        """
        required_columns = ['high', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per rilevare livelli di liquidità")
        
        result = df.copy()
        result['liquidity_level_above'] = np.nan
        result['liquidity_level_below'] = np.nan
        
        # Dobbiamo avere almeno 2*n_bars+1 candele
        if len(result) < 2*n_bars+1:
            return result
        
        for i in range(n_bars, len(result) - n_bars):
            # Verifica se abbiamo uno swing high
            swing_high = True
            for j in range(1, n_bars+1):
                if (result['high'].iloc[i] <= result['high'].iloc[i-j] or 
                    result['high'].iloc[i] <= result['high'].iloc[i+j]):
                    swing_high = False
                    break
            
            if swing_high:
                result.loc[result.index[i], 'liquidity_level_above'] = result['high'].iloc[i]
            
            # Verifica se abbiamo uno swing low
            swing_low = True
            for j in range(1, n_bars+1):
                if (result['low'].iloc[i] >= result['low'].iloc[i-j] or 
                    result['low'].iloc[i] >= result['low'].iloc[i+j]):
                    swing_low = False
                    break
            
            if swing_low:
                result.loc[result.index[i], 'liquidity_level_below'] = result['low'].iloc[i]
        
        return result
    
    @staticmethod
    def detect_market_structure(df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
        """
        Rileva la struttura di mercato identificando higher highs, higher lows, 
        lower highs, lower lows.
        
        Args:
            df: DataFrame con colonne 'high', 'low'
            swing_length: Numero di barre per definire uno swing
            
        Returns:
            DataFrame con colonne struttura di mercato aggiunte
        """
        required_columns = ['high', 'low']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per rilevare la struttura di mercato")
        
        result = df.copy()
        
        # Rileva swing highs e swing lows
        df_with_swings = MarketPatterns.detect_liquidity_levels(result, n_bars=swing_length)
        
        # Prepara colonne per la struttura
        result['market_structure_high'] = np.nan
        result['market_structure_low'] = np.nan
        
        # Estrai swing points
        swing_highs = df_with_swings[~df_with_swings['liquidity_level_above'].isna()]
        swing_lows = df_with_swings[~df_with_swings['liquidity_level_below'].isna()]
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return result
        
        # Analizza swing highs per HH/LH
        prev_high = None
        high_idx = 0
        
        for idx, row in swing_highs.iterrows():
            current_high = row['liquidity_level_above']
            
            if prev_high is not None:
                if current_high > prev_high:
                    # Higher High
                    result.loc[idx, 'market_structure_high'] = current_high
                else:
                    # Lower High
                    result.loc[idx, 'market_structure_high'] = current_high
            
            prev_high = current_high
            high_idx = idx
        
        # Analizza swing lows per HL/LL
        prev_low = None
        low_idx = 0
        
        for idx, row in swing_lows.iterrows():
            current_low = row['liquidity_level_below']
            
            if prev_low is not None:
                if current_low > prev_low:
                    # Higher Low
                    result.loc[idx, 'market_structure_low'] = current_low
                else:
                    # Lower Low
                    result.loc[idx, 'market_structure_low'] = current_low
            
            prev_low = current_low
            low_idx = idx
        
        return result
    
    @staticmethod
    def detect_sr_zones(df: pd.DataFrame, lookback: int = 100, zone_threshold: float = 0.01) -> pd.DataFrame:
        """
        Rileva zone di supporto e resistenza.
        
        Args:
            df: DataFrame con colonne 'high', 'low', 'close'
            lookback: Numero massimo di barre per l'analisi
            zone_threshold: Soglia percentuale per raggruppare i livelli
            
        Returns:
            DataFrame con colonne sr_support e sr_resistance aggiunte
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                raise ProcessingError(f"La colonna '{col}' è richiesta per rilevare zone S/R")
        
        result = df.copy()
        result['sr_support'] = np.nan
        result['sr_resistance'] = np.nan
        
        # Usa i livelli di liquidità come base per i livelli S/R
        df_with_levels = MarketPatterns.detect_liquidity_levels(result)
        
        # Estrai livelli di liquidità
        liq_above = df_with_levels[~df_with_levels['liquidity_level_above'].isna()]['liquidity_level_above'].tolist()
        liq_below = df_with_levels[~df_with_levels['liquidity_level_below'].isna()]['liquidity_level_below'].tolist()
        
        # Raggruppa livelli simili per resistenze
        resistance_zones = []
        
        for level in sorted(liq_above):
            # Se il livello è vicino a una zona esistente, aggiorna la zona
            added_to_existing = False
            
            for i, zone in enumerate(resistance_zones):
                zone_avg = sum(zone) / len(zone)
                if abs(level - zone_avg) / zone_avg < zone_threshold:
                    resistance_zones[i].append(level)
                    added_to_existing = True
                    break
            
            # Se non è stato aggiunto a una zona esistente, crea una nuova zona
            if not added_to_existing:
                resistance_zones.append([level])
        
        # Raggruppa livelli simili per supporti
        support_zones = []
        
        for level in sorted(liq_below, reverse=True):
            # Se il livello è vicino a una zona esistente, aggiorna la zona
            added_to_existing = False
            
            for i, zone in enumerate(support_zones):
                zone_avg = sum(zone) / len(zone)
                if abs(level - zone_avg) / zone_avg < zone_threshold:
                    support_zones[i].append(level)
                    added_to_existing = True
                    break
            
            # Se non è stato aggiunto a una zona esistente, crea una nuova zona
            if not added_to_existing:
                support_zones.append([level])
        
        # Calcola il valore medio di ogni zona
        resistance_levels = [sum(zone) / len(zone) for zone in resistance_zones]
        support_levels = [sum(zone) / len(zone) for zone in support_zones]
        
        # Applica i livelli alle ultime candele
        last_price = result['close'].iloc[-1]
        
        # Trova i supporti sotto il prezzo attuale
        supports_below = [level for level in support_levels if level < last_price]
        
        if supports_below:
            # Prendi il supporto più vicino al prezzo attuale
            closest_support = max(supports_below)
            result.loc[result.index[-1], 'sr_support'] = closest_support
        
        # Trova le resistenze sopra il prezzo attuale
        resistances_above = [level for level in resistance_levels if level > last_price]
        
        if resistances_above:
            # Prendi la resistenza più vicina al prezzo attuale
            closest_resistance = min(resistances_above)
            result.loc[result.index[-1], 'sr_resistance'] = closest_resistance
        
        return result


class DataProcessor:
    """
    Classe principale per l'elaborazione dei dati di mercato.
    Coordina il calcolo di indicatori e pattern.
    """
    
    def __init__(self):
        """Inizializza il processor."""
        self.indicators = TechnicalIndicators()
        self.patterns = MarketPatterns()
        self.validator = MarketDataValidator()
    
    @time_it
    def process_dataframe(self, df: pd.DataFrame, calculate_patterns: bool = True) -> pd.DataFrame:
        """
        Elabora un DataFrame calcolando indicatori e pattern.
        
        Args:
            df: DataFrame con dati OHLCV
            calculate_patterns: Se calcolare i pattern (richiede più tempo)
            
        Returns:
            DataFrame elaborato con indicatori e pattern
        """
        try:
            # Valida il DataFrame
            df = self.validator.validate_ohlcv(df)
            
            # Calcola gli indicatori tecnici di base
            df = self.indicators.calculate_ema(df)
            df = self.indicators.calculate_rsi(df)
            df = self.indicators.calculate_atr(df)
            df = self.indicators.calculate_premium_discount(df)
            
            # Calcola i pattern se richiesto
            if calculate_patterns:
                df = self.patterns.detect_fair_value_gaps(df)
                df = self.patterns.detect_order_blocks(df)
                df = self.patterns.detect_breaker_blocks(df)
                df = self.patterns.detect_liquidity_levels(df)
                df = self.patterns.detect_market_structure(df)
            
            return df
        
        except Exception as e:
            raise ProcessingError("Errore durante l'elaborazione del DataFrame", 
                                 details={"error": str(e)})
    
    @time_it
    def process_symbol_data(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                           calculate_patterns: bool = True) -> pd.DataFrame:
        """
        Elabora i dati di un simbolo specifico.
        
        Args:
            df: DataFrame con dati OHLCV
            symbol: Nome del simbolo
            timeframe: Timeframe
            calculate_patterns: Se calcolare i pattern
            
        Returns:
            DataFrame elaborato
        """
        try:
            data_logger.info(f"Elaborazione dati per {symbol} con timeframe {timeframe}")
            
            # Esegui elaborazione standard
            processed_df = self.process_dataframe(df, calculate_patterns)
            
            # Aggiungi eventuali elaborazioni specifiche per simbolo/timeframe
            symbol_obj = get_symbol(symbol)
            timeframe_obj = get_timeframe(timeframe)
            
            if symbol_obj and timeframe_obj:
                # Esempio: calcola indicatori aggiuntivi per timeframe più lunghi
                if timeframe in ['daily', 'weekly', 'monthly']:
                    processed_df = self.indicators.calculate_bollinger_bands(processed_df)
                    processed_df = self.indicators.calculate_macd(processed_df)
                
                # Esempio: calcola indicatori specifici per l'asset class
                if symbol_obj.asset_class.value == "Precious Metal":
                    processed_df = self.indicators.calculate_stochastic(processed_df)
            
            data_logger.info(f"Elaborazione completata per {symbol} con timeframe {timeframe}")
            return processed_df
        
        except Exception as e:
            error_msg = f"Errore nell'elaborazione dei dati per {symbol} con timeframe {timeframe}: {e}"
            data_logger.error(error_msg)
            raise ProcessingError(error_msg)
    
    @time_it
    def batch_process(self, data_dict: Dict[str, pd.DataFrame], 
                     calculate_patterns: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Elabora dati per più simboli/timeframe in batch.
        
        Args:
            data_dict: Dizionario {symbol_timeframe: dataframe}
            calculate_patterns: Se calcolare i pattern
            
        Returns:
            Dizionario con DataFrame elaborati
        """
        results = {}
        errors = []
        
        for key, df in data_dict.items():
            try:
                # Estrai simbolo e timeframe dalla chiave (es. "Gold_daily")
                symbol, timeframe = key.split('_')
                
                # Elabora i dati
                processed_df = self.process_symbol_data(df, symbol, timeframe, calculate_patterns)
                
                # Salva il risultato
                results[key] = processed_df
            
            except Exception as e:
                errors.append(f"{key}: {e}")
                data_logger.error(f"Errore nell'elaborazione batch per {key}: {e}")
                # Aggiungiamo il DataFrame originale come fallback
                results[key] = df
        
        if errors:
            data_logger.warning(f"Errori nell'elaborazione batch: {errors}")
        
        return results


class MarketDataBatcher:
    """
    Classe per gestire il batch processing di dati di mercato.
    """
    
    def __init__(self, db_repository=None, processor=None):
        """
        Inizializza il batcher.
        
        Args:
            db_repository: Repository del database
            processor: Processor per l'elaborazione
        """
        # Importa qui per evitare dipendenze circolari
        from .database import get_default_repository
        
        self.db_repository = db_repository or get_default_repository()
        self.processor = processor or DataProcessor()
    
    @time_it
    def process_and_store(self, symbol: str, timeframe: str, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         limit: int = 5000) -> int:
        """
        Elabora e salva i dati per un simbolo e timeframe specifico.
        
        Args:
            symbol: Simbolo
            timeframe: Timeframe
            start_date: Data di inizio (opzionale)
            end_date: Data di fine (opzionale)
            limit: Limite di record da elaborare
            
        Returns:
            Numero di record elaborati e salvati
        """
        try:
            data_logger.info(f"Avvio elaborazione e salvataggio per {symbol} con timeframe {timeframe}")
            
            # Recupera i dati grezzi
            raw_df = self.db_repository.fetch_raw_data(symbol, timeframe, start_date, end_date, limit)
            
            if raw_df.empty:
                data_logger.warning(f"Nessun dato grezzo trovato per {symbol} con timeframe {timeframe}")
                return 0
            
            # Elabora i dati
            processed_df = self.processor.process_symbol_data(raw_df, symbol, timeframe)
            
            # Salva i dati elaborati
            records_affected = self.db_repository.store_processed_data(processed_df, symbol, timeframe)
            
            data_logger.info(f"Elaborati e salvati {records_affected} record per {symbol} con timeframe {timeframe}")
            return records_affected
        
        except Exception as e:
            error_msg = f"Errore durante l'elaborazione e salvataggio per {symbol} con timeframe {timeframe}: {e}"
            data_logger.error(error_msg)
            return 0
    
    @time_it
    def process_all(self, symbols: Optional[List[str]] = None, 
                   timeframes: Optional[List[str]] = None,
                   days_back: int = 90) -> Dict[str, int]:
        """
        Elabora e salva i dati per tutti i simboli e timeframe specificati.
        
        Args:
            symbols: Lista di simboli (default: tutti)
            timeframes: Lista di timeframe (default: tutti)
            days_back: Numero di giorni indietro da elaborare
            
        Returns:
            Dizionario con numero di record elaborati per ogni combinazione
        """
        from ..config import SYMBOLS, TIMEFRAMES
        
        # Usa tutti i simboli e timeframe disponibili se non specificati
        if symbols is None:
            symbols = [s for s in SYMBOLS.keys()]
        
        if timeframes is None:
            timeframes = [t for t in TIMEFRAMES.keys()]
        
        # Calcola la data di inizio
        start_date = datetime.now() - timedelta(days=days_back)
        
        results = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                try:
                    records = self.process_and_store(symbol, timeframe, start_date=start_date)
                    results[key] = records
                except Exception as e:
                    data_logger.error(f"Errore nell'elaborazione completa per {key}: {e}")
                    results[key] = -1
        
        return results


# Funzioni factory

def get_processor() -> DataProcessor:
    """
    Ottiene un processor per l'elaborazione dei dati.
    
    Returns:
        Istanza del DataProcessor
    """
    return DataProcessor()


def get_batcher(db_repository=None) -> MarketDataBatcher:
    """
    Ottiene un batcher per l'elaborazione batch dei dati.
    
    Args:
        db_repository: Repository del database (opzionale)
        
    Returns:
        Istanza del MarketDataBatcher
    """
    return MarketDataBatcher(db_repository=db_repository)