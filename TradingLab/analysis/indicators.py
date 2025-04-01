# Indicatori tecnici

"""
Indicatori tecnici avanzati per il progetto TradingLab.
Questo modulo fornisce implementazioni di indicatori tecnici avanzati per l'analisi dei mercati finanziari.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import math
from datetime import datetime, timedelta

# Importazioni dal modulo utils
from ..utils import app_logger, ProcessingError, time_it


class TrendIndicators:
    """Indicatori per l'analisi dei trend di mercato."""
    
    @staticmethod
    def supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calcola l'indicatore Supertrend.
        
        Args:
            df: DataFrame con colonne high, low, close
            atr_period: Periodo per il calcolo dell'ATR
            multiplier: Moltiplicatore per le bande
            
        Returns:
            DataFrame con colonne supertrend, supertrend_upper, supertrend_lower, supertrend_direction
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Supertrend")
        
        # Calcola True Range
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift(1))
            ),
            np.abs(result['low'] - result['close'].shift(1))
        )
        
        # Calcola ATR
        result['atr'] = result['tr'].rolling(window=atr_period).mean()
        
        # Calcola le bande
        result['upperband'] = ((result['high'] + result['low']) / 2) + (multiplier * result['atr'])
        result['lowerband'] = ((result['high'] + result['low']) / 2) - (multiplier * result['atr'])
        
        # Inizializza le colonne per l'indicatore Supertrend
        result['supertrend'] = 0.0
        result['supertrend_direction'] = 0  # 1 per trend rialzista, -1 per trend ribassista
        
        # Calcola Supertrend
        for i in range(1, len(result)):
            curr_close = result['close'].iloc[i]
            prev_close = result['close'].iloc[i-1]
            
            # Calcola le bande aggiornate
            if result['upperband'].iloc[i] < result['upperband'].iloc[i-1] or prev_close > result['upperband'].iloc[i-1]:
                result.loc[result.index[i], 'upperband'] = result['upperband'].iloc[i]
            else:
                result.loc[result.index[i], 'upperband'] = result['upperband'].iloc[i-1]
            
            if result['lowerband'].iloc[i] > result['lowerband'].iloc[i-1] or prev_close < result['lowerband'].iloc[i-1]:
                result.loc[result.index[i], 'lowerband'] = result['lowerband'].iloc[i]
            else:
                result.loc[result.index[i], 'lowerband'] = result['lowerband'].iloc[i-1]
            
            # Calcola la direzione del trend
            if prev_close <= result['supertrend'].iloc[i-1] and curr_close > result['upperband'].iloc[i]:
                result.loc[result.index[i], 'supertrend_direction'] = 1
            elif prev_close >= result['supertrend'].iloc[i-1] and curr_close < result['lowerband'].iloc[i]:
                result.loc[result.index[i], 'supertrend_direction'] = -1
            else:
                result.loc[result.index[i], 'supertrend_direction'] = result['supertrend_direction'].iloc[i-1]
                
                if result['supertrend_direction'].iloc[i-1] == 1 and curr_close < result['lowerband'].iloc[i]:
                    result.loc[result.index[i], 'supertrend_direction'] = -1
                elif result['supertrend_direction'].iloc[i-1] == -1 and curr_close > result['upperband'].iloc[i]:
                    result.loc[result.index[i], 'supertrend_direction'] = 1
            
            # Determina il valore Supertrend
            if result['supertrend_direction'].iloc[i] == 1:
                result.loc[result.index[i], 'supertrend'] = result['lowerband'].iloc[i]
            else:
                result.loc[result.index[i], 'supertrend'] = result['upperband'].iloc[i]
        
        # Pulisci le colonne temporanee
        result.drop(['tr', 'atr'], axis=1, inplace=True)
        
        # Aggiungi colonne per identificare crossover
        result['supertrend_uptrend'] = result['supertrend_direction'] == 1
        result['supertrend_downtrend'] = result['supertrend_direction'] == -1
        
        # Trend change signals
        result['supertrend_buy_signal'] = (result['supertrend_direction'] == 1) & (result['supertrend_direction'].shift(1) == -1)
        result['supertrend_sell_signal'] = (result['supertrend_direction'] == -1) & (result['supertrend_direction'].shift(1) == 1)
        
        return result
    
    @staticmethod
    def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.DataFrame:
        """
        Calcola l'indicatore Parabolic SAR (Stop and Reverse).
        
        Args:
            df: DataFrame con colonne high, low, close
            af_start: Acceleration factor iniziale
            af_step: Incremento dell'acceleration factor
            af_max: Valore massimo dell'acceleration factor
            
        Returns:
            DataFrame con colonna psar e direzione del trend
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Parabolic SAR")
        
        # Inizializza colonne per PSAR
        result['psar'] = np.nan
        result['psar_direction'] = np.nan  # 1 per uptrend, -1 per downtrend
        result['psar_ep'] = np.nan  # Extreme Point
        result['psar_af'] = np.nan  # Acceleration Factor
        
        # Determina trend iniziale (assumiamo un uptrend iniziale)
        init_trend = 1
        
        # Imposta PSAR iniziale (primo punto)
        result.loc[result.index[0], 'psar'] = result['low'].iloc[0] if init_trend == 1 else result['high'].iloc[0]
        result.loc[result.index[0], 'psar_direction'] = init_trend
        result.loc[result.index[0], 'psar_ep'] = result['high'].iloc[0] if init_trend == 1 else result['low'].iloc[0]
        result.loc[result.index[0], 'psar_af'] = af_start
        
        # Calcola PSAR per i punti successivi
        for i in range(1, len(result)):
            # Ottieni valori dalla riga precedente
            prev_psar = result['psar'].iloc[i-1]
            prev_direction = result['psar_direction'].iloc[i-1]
            prev_ep = result['psar_ep'].iloc[i-1]
            prev_af = result['psar_af'].iloc[i-1]
            
            # Calcola nuovo PSAR in base alla direzione del trend
            if prev_direction == 1:  # Uptrend
                # PSAR = Prev PSAR + AF * (EP - Prev PSAR)
                psar = prev_psar + prev_af * (prev_ep - prev_psar)
                # Assicurati che PSAR non sia superiore ai minimi dei due periodi precedenti
                psar = min(psar, result['low'].iloc[max(0, i-2):i].min())
            else:  # Downtrend
                psar = prev_psar + prev_af * (prev_ep - prev_psar)
                # Assicurati che PSAR non sia inferiore ai massimi dei due periodi precedenti
                psar = max(psar, result['high'].iloc[max(0, i-2):i].max())
            
            # Verifica se c'è un'inversione del trend
            curr_high = result['high'].iloc[i]
            curr_low = result['low'].iloc[i]
            
            if prev_direction == 1 and curr_low < psar:  # Da uptrend a downtrend
                direction = -1
                psar = prev_ep  # Imposta PSAR al precedente Extreme Point
                ep = curr_low
                af = af_start
            elif prev_direction == -1 and curr_high > psar:  # Da downtrend a uptrend
                direction = 1
                psar = prev_ep  # Imposta PSAR al precedente Extreme Point
                ep = curr_high
                af = af_start
            else:  # Nessuna inversione
                direction = prev_direction
                
                # Aggiorna Extreme Point se necessario
                if direction == 1 and curr_high > prev_ep:
                    ep = curr_high
                    af = min(prev_af + af_step, af_max)
                elif direction == -1 and curr_low < prev_ep:
                    ep = curr_low
                    af = min(prev_af + af_step, af_max)
                else:
                    ep = prev_ep
                    af = prev_af
            
            # Aggiorna i valori nel DataFrame
            result.loc[result.index[i], 'psar'] = psar
            result.loc[result.index[i], 'psar_direction'] = direction
            result.loc[result.index[i], 'psar_ep'] = ep
            result.loc[result.index[i], 'psar_af'] = af
        
        # Aggiungi colonne per identificare gli stati del trend
        result['psar_uptrend'] = result['psar_direction'] == 1
        result['psar_downtrend'] = result['psar_direction'] == -1
        
        # Trend change signals
        result['psar_buy_signal'] = (result['psar_direction'] == 1) & (result['psar_direction'].shift(1) == -1)
        result['psar_sell_signal'] = (result['psar_direction'] == -1) & (result['psar_direction'].shift(1) == 1)
        
        # Rimuovi colonne temporanee
        result.drop(['psar_ep', 'psar_af'], axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calcola l'indicatore ADX (Average Directional Index).
        
        Args:
            df: DataFrame con colonne high, low, close
            period: Periodo per il calcolo dell'ADX
            
        Returns:
            DataFrame con colonne ADX, +DI, -DI
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per ADX")
        
        # Calcola True Range (TR)
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift(1))
            ),
            np.abs(result['low'] - result['close'].shift(1))
        )
        
        # Calcola +DM e -DM (Directional Movement)
        result['+dm'] = np.where(
            (result['high'] - result['high'].shift(1) > result['low'].shift(1) - result['low']) & 
            (result['high'] - result['high'].shift(1) > 0),
            result['high'] - result['high'].shift(1),
            0
        )
        
        result['-dm'] = np.where(
            (result['low'].shift(1) - result['low'] > result['high'] - result['high'].shift(1)) & 
            (result['low'].shift(1) - result['low'] > 0),
            result['low'].shift(1) - result['low'],
            0
        )
        
        # Calcola TR, +DM e -DM come medie mobili esponenziali
        result['tr_ema'] = result['tr'].ewm(alpha=1/period, adjust=False).mean()
        result['+dm_ema'] = result['+dm'].ewm(alpha=1/period, adjust=False).mean()
        result['-dm_ema'] = result['-dm'].ewm(alpha=1/period, adjust=False).mean()
        
        # Calcola +DI e -DI (Directional Indicators)
        result['+di'] = (result['+dm_ema'] / result['tr_ema']) * 100
        result['-di'] = (result['-dm_ema'] / result['tr_ema']) * 100
        
        # Calcola DX (Directional Index)
        result['dx'] = np.abs(result['+di'] - result['-di']) / (result['+di'] + result['-di']) * 100
        
        # Calcola ADX (Average Directional Index)
        result['adx'] = result['dx'].ewm(alpha=1/period, adjust=False).mean()
        
        # Pulisci le colonne temporanee
        columns_to_drop = ['tr', '+dm', '-dm', 'tr_ema', '+dm_ema', '-dm_ema', 'dx']
        result.drop(columns_to_drop, axis=1, inplace=True)
        
        # Aggiungi colonne per identificare trend forti
        result['adx_strong_trend'] = result['adx'] > 25  # ADX > 25 indica un trend forte
        result['adx_weak_trend'] = result['adx'] < 20    # ADX < 20 indica un trend debole
        
        # Identifica trend rialzista/ribassista
        result['adx_bullish'] = (result['+di'] > result['-di']) & result['adx_strong_trend']
        result['adx_bearish'] = (result['+di'] < result['-di']) & result['adx_strong_trend']
        
        return result
    
    @staticmethod
    def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calcola i Canali di Keltner.
        
        Args:
            df: DataFrame con colonne high, low, close
            ema_period: Periodo per il calcolo dell'EMA centrale
            atr_period: Periodo per il calcolo dell'ATR
            multiplier: Moltiplicatore dell'ATR per le bande
            
        Returns:
            DataFrame con colonne kc_middle, kc_upper, kc_lower
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Keltner Channels")
        
        # Calcola l'EMA del prezzo tipico
        result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
        result['kc_middle'] = result['typical_price'].ewm(span=ema_period, adjust=False).mean()
        
        # Calcola ATR
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift(1))
            ),
            np.abs(result['low'] - result['close'].shift(1))
        )
        result['atr'] = result['tr'].rolling(window=atr_period).mean()
        
        # Calcola le bande
        result['kc_upper'] = result['kc_middle'] + (multiplier * result['atr'])
        result['kc_lower'] = result['kc_middle'] - (multiplier * result['atr'])
        
        # Rimuovi colonne temporanee
        result.drop(['typical_price', 'tr', 'atr'], axis=1, inplace=True)
        
        # Aggiungi indicatori di stato
        result['kc_above_upper'] = result['close'] > result['kc_upper']  # Prezzo sopra banda superiore
        result['kc_below_lower'] = result['close'] < result['kc_lower']  # Prezzo sotto banda inferiore
        result['kc_inside'] = (result['close'] >= result['kc_lower']) & (result['close'] <= result['kc_upper'])
        
        # Contrazione/espansione dei canali
        result['kc_width'] = (result['kc_upper'] - result['kc_lower']) / result['kc_middle']
        
        return result


class MomentumIndicators:
    """Indicatori per l'analisi del momentum di mercato."""
    
    @staticmethod
    def rsi_divergence(df: pd.DataFrame, rsi_period: int = 14, price_type: str = 'close', window: int = 10) -> pd.DataFrame:
        """
        Calcola le divergenze dell'RSI.
        
        Args:
            df: DataFrame con colonne OHLC e RSI
            rsi_period: Periodo utilizzato per calcolare l'RSI
            price_type: Tipo di prezzo da utilizzare ('close', 'high', 'low')
            window: Finestra per cercare divergenze
            
        Returns:
            DataFrame con colonne di divergenza RSI aggiunte
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = [price_type, f'rsi_{rsi_period}']
        if not all(col in result.columns for col in required_cols):
            # Se l'RSI non è presente, calcolalo
            if f'rsi_{rsi_period}' not in result.columns and 'close' in result.columns:
                # Calcola la variazione
                delta = result['close'].diff()
                
                # Separa guadagni e perdite
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # Calcola medie esponenziali di guadagni e perdite
                avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
                
                # Calcola RS e RSI
                rs = avg_gain / avg_loss
                result[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))
            else:
                raise ProcessingError(f"Colonne {required_cols} richieste per divergenze RSI")
        
        # Inizializza colonne di divergenza
        result['rsi_bullish_div'] = False  # Divergenza rialzista
        result['rsi_bearish_div'] = False  # Divergenza ribassista
        
        # Trova i pivot (massimi e minimi locali) per il prezzo
        result['price_pivot_high'] = False
        result['price_pivot_low'] = False
        
        # Trova i pivot per l'RSI
        result[f'rsi_{rsi_period}_pivot_high'] = False
        result[f'rsi_{rsi_period}_pivot_low'] = False
        
        # Identifica i pivot per il prezzo
        for i in range(window, len(result) - window):
            # Pivot alto per il prezzo (un massimo locale)
            price_window = result[price_type].iloc[i-window:i+window+1]
            if result[price_type].iloc[i] == price_window.max():
                result.loc[result.index[i], 'price_pivot_high'] = True
            
            # Pivot basso per il prezzo (un minimo locale)
            if result[price_type].iloc[i] == price_window.min():
                result.loc[result.index[i], 'price_pivot_low'] = True
            
            # Pivot alto per l'RSI
            rsi_window = result[f'rsi_{rsi_period}'].iloc[i-window:i+window+1]
            if result[f'rsi_{rsi_period}'].iloc[i] == rsi_window.max():
                result.loc[result.index[i], f'rsi_{rsi_period}_pivot_high'] = True
            
            # Pivot basso per l'RSI
            if result[f'rsi_{rsi_period}'].iloc[i] == rsi_window.min():
                result.loc[result.index[i], f'rsi_{rsi_period}_pivot_low'] = True
        
        # Cerca divergenze
        price_pivots_high = result[result['price_pivot_high']].index
        price_pivots_low = result[result['price_pivot_low']].index
        rsi_pivots_high = result[result[f'rsi_{rsi_period}_pivot_high']].index
        rsi_pivots_low = result[result[f'rsi_{rsi_period}_pivot_low']].index
        
        for i in range(1, len(price_pivots_high)):
            # Cerca divergenze ribassiste (prezzi più alti, RSI più bassi)
            if price_pivots_high[i-1] in price_pivots_high and price_pivots_high[i] in price_pivots_high:
                price_idx1 = result.index.get_loc(price_pivots_high[i-1])
                price_idx2 = result.index.get_loc(price_pivots_high[i])
                
                # Trova i pivot RSI più vicini
                rsi_high_around_1 = [idx for idx in rsi_pivots_high if abs(result.index.get_loc(idx) - price_idx1) <= window]
                rsi_high_around_2 = [idx for idx in rsi_pivots_high if abs(result.index.get_loc(idx) - price_idx2) <= window]
                
                if rsi_high_around_1 and rsi_high_around_2:
                    rsi_idx1 = result.index.get_loc(rsi_high_around_1[0])
                    rsi_idx2 = result.index.get_loc(rsi_high_around_2[0])
                    
                    # Verifica divergenza ribassista (prezzo più alto, RSI più basso)
                    if (result[price_type].iloc[price_idx2] > result[price_type].iloc[price_idx1] and
                        result[f'rsi_{rsi_period}'].iloc[rsi_idx2] < result[f'rsi_{rsi_period}'].iloc[rsi_idx1]):
                        result.loc[result.index[price_idx2], 'rsi_bearish_div'] = True
        
        for i in range(1, len(price_pivots_low)):
            # Cerca divergenze rialziste (prezzi più bassi, RSI più alti)
            if price_pivots_low[i-1] in price_pivots_low and price_pivots_low[i] in price_pivots_low:
                price_idx1 = result.index.get_loc(price_pivots_low[i-1])
                price_idx2 = result.index.get_loc(price_pivots_low[i])
                
                # Trova i pivot RSI più vicini
                rsi_low_around_1 = [idx for idx in rsi_pivots_low if abs(result.index.get_loc(idx) - price_idx1) <= window]
                rsi_low_around_2 = [idx for idx in rsi_pivots_low if abs(result.index.get_loc(idx) - price_idx2) <= window]
                
                if rsi_low_around_1 and rsi_low_around_2:
                    rsi_idx1 = result.index.get_loc(rsi_low_around_1[0])
                    rsi_idx2 = result.index.get_loc(rsi_low_around_2[0])
                    
                    # Verifica divergenza rialzista (prezzo più basso, RSI più alto)
                    if (result[price_type].iloc[price_idx2] < result[price_type].iloc[price_idx1] and
                        result[f'rsi_{rsi_period}'].iloc[rsi_idx2] > result[f'rsi_{rsi_period}'].iloc[rsi_idx1]):
                        result.loc[result.index[price_idx2], 'rsi_bullish_div'] = True
        
        # Rimuovi colonne temporanee
        result.drop(['price_pivot_high', 'price_pivot_low', 
                     f'rsi_{rsi_period}_pivot_high', f'rsi_{rsi_period}_pivot_low'], 
                   axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def awesome_oscillator(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 34) -> pd.DataFrame:
        """
        Calcola l'Awesome Oscillator.
        
        Args:
            df: DataFrame con colonne high e low
            fast_period: Periodo per la media veloce
            slow_period: Periodo per la media lenta
            
        Returns:
            DataFrame con colonna ao (Awesome Oscillator) aggiunta
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['high', 'low']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high e low richieste per Awesome Oscillator")
        
        # Calcola il prezzo mediano (median price)
        result['median_price'] = (result['high'] + result['low']) / 2
        
        # Calcola le medie mobili semplici
        result['ao_fast'] = result['median_price'].rolling(window=fast_period).mean()
        result['ao_slow'] = result['median_price'].rolling(window=slow_period).mean()
        
        # Calcola l'Awesome Oscillator
        result['ao'] = result['ao_fast'] - result['ao_slow']
        
        # Rimuovi colonne temporanee
        result.drop(['median_price', 'ao_fast', 'ao_slow'], axis=1, inplace=True)
        
        # Aggiungi informazioni sull'andamento dell'oscillatore
        result['ao_positive'] = result['ao'] > 0
        result['ao_negative'] = result['ao'] < 0
        
        # Identifica i segnali "Saucer" (3 barre nella stessa direzione)
        result['ao_green'] = result['ao'] > result['ao'].shift(1)
        result['ao_red'] = result['ao'] < result['ao'].shift(1)
        
        # Segnali di trading
        # Saucer: tre barre nella stessa direzione con la barra centrale di direzione opposta
        result['ao_bullish_saucer'] = (
            result['ao_positive'] &
            result['ao_red'].shift(2) &
            result['ao_red'].shift(1) &
            result['ao_green']
        )
        
        result['ao_bearish_saucer'] = (
            result['ao_negative'] &
            result['ao_green'].shift(2) &
            result['ao_green'].shift(1) &
            result['ao_red']
        )
        
        # Twin Peaks: due picchi nella zona opposta alla tendenza del prezzo
        # (Implementazione semplificata)
        for i in range(5, len(result)):
            window = result['ao'].iloc[i-5:i]
            if (result['ao'].iloc[i] > 0 and  # AO positivo
                (window < 0).any() and  # Alcune barre negative nel passato recente
                window.min() < window.iloc[0]):  # Il secondo minimo è più basso
                result.loc[result.index[i], 'ao_bullish_twin_peaks'] = True
            else:
                result.loc[result.index[i], 'ao_bullish_twin_peaks'] = False
                
            if (result['ao'].iloc[i] < 0 and  # AO negativo
                (window > 0).any() and  # Alcune barre positive nel passato recente
                window.max() > window.iloc[0]):  # Il secondo massimo è più alto
                result.loc[result.index[i], 'ao_bearish_twin_peaks'] = True
            else:
                result.loc[result.index[i], 'ao_bearish_twin_peaks'] = False
        
        # Rimuovi colonne temporanee
        result.drop(['ao_green', 'ao_red'], axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calcola l'indicatore Williams %R.
        
        Args:
            df: DataFrame con colonne high, low, close
            period: Periodo di lookback
            
        Returns:
            DataFrame con colonna williams_r aggiunta
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Williams %R")
        
        # Calcola massimo e minimo nel periodo
        highest_high = result['high'].rolling(window=period).max()
        lowest_low = result['low'].rolling(window=period).min()
        
        # Calcola Williams %R
        result[f'williams_r_{period}'] = -100 * (highest_high - result['close']) / (highest_high - lowest_low)
        
        # Identifica livelli di ipercomprato/ipervenduto
        result[f'williams_r_{period}_oversold'] = result[f'williams_r_{period}'] <= -80
        result[f'williams_r_{period}_overbought'] = result[f'williams_r_{period}'] >= -20
        
        # Identifica crossover
        result[f'williams_r_{period}_buy_signal'] = (
            (result[f'williams_r_{period}'] > -80) & 
            (result[f'williams_r_{period}'].shift(1) <= -80)
        )
        
        result[f'williams_r_{period}_sell_signal'] = (
            (result[f'williams_r_{period}'] < -20) & 
            (result[f'williams_r_{period}'].shift(1) >= -20)
        )
        
        return result


class VolumeIndicators:
    """Indicatori basati sul volume."""
    
    @staticmethod
    def volume_profile(df: pd.DataFrame, bins: int = 10, window: Optional[int] = None) -> pd.DataFrame:
        """
        Calcola il Volume Profile.
        
        Args:
            df: DataFrame con colonne high, low, close e volume
            bins: Numero di livelli di prezzo (bins)
            window: Finestra di tempo da considerare (None per tutti i dati)
            
        Returns:
            DataFrame con Volume Profile per livelli di prezzo
        """
        if 'volume' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
            raise ProcessingError("Colonne volume, high, low richieste per Volume Profile")
        
        # Usa gli ultimi 'window' periodi se specificato
        data = df.copy()
        if window is not None:
            data = data.iloc[-window:]
        
        # Calcola il range di prezzo
        price_range = data['high'].max() - data['low'].min()
        
        # Definisci i bin di prezzo
        price_bins = np.linspace(data['low'].min(), data['high'].max(), bins + 1)
        
        # Inizializza array per volume per bin
        volume_profile = np.zeros(bins)
        
        # Calcola il volume associato a ciascuna barra per ogni bin di prezzo
        for i in range(len(data)):
            bar_high = data['high'].iloc[i]
            bar_low = data['low'].iloc[i]
            bar_volume = data['volume'].iloc[i]
            
            # Calcola la percentuale di ciascuna barra che interseca ogni bin
            for j in range(bins):
                bin_low = price_bins[j]
                bin_high = price_bins[j + 1]
                
                # Caso 1: La barra è completamente all'interno del bin
                if bar_low >= bin_low and bar_high <= bin_high:
                    volume_profile[j] += bar_volume
                # Caso 2: La barra interseca parzialmente il bin
                elif bar_low < bin_high and bar_high > bin_low:
                    # Calcola l'intersezione
                    intersection = min(bar_high, bin_high) - max(bar_low, bin_low)
                    bar_range = bar_high - bar_low
                    
                    # Distribuisci il volume proporzionalmente
                    if bar_range > 0:
                        volume_profile[j] += bar_volume * (intersection / bar_range)
        
        # Crea DataFrame con il risultato
        volume_prof_df = pd.DataFrame({
            'price_level': [(price_bins[i] + price_bins[i+1]) / 2 for i in range(bins)],
            'volume': volume_profile
        })
        
        # Ordina per volume per trovare i livelli di massimo volume
        volume_prof_df = volume_prof_df.sort_values('volume', ascending=False)
        
        # Trova il Point of Control (POC) - livello di prezzo con il massimo volume
        poc = volume_prof_df.iloc[0]['price_level']
        
        # Aggiungi il POC al DataFrame originale come colonna
        result = df.copy()
        result['volume_poc'] = poc
        
        # Value Area - regione che contiene il 70% del volume totale
        total_volume = volume_prof_df['volume'].sum()
        cumulative_volume = 0
        value_area_levels = []
        
        for _, row in volume_prof_df.iterrows():
            cumulative_volume += row['volume']
            value_area_levels.append(row['price_level'])
            
            if cumulative_volume >= 0.7 * total_volume:
                break
        
        result['volume_va_high'] = max(value_area_levels)
        result['volume_va_low'] = min(value_area_levels)
        
        return result
    
    @staticmethod
    def on_balance_volume(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola l'indicatore On Balance Volume (OBV).
        
        Args:
            df: DataFrame con colonne close e volume
            
        Returns:
            DataFrame con colonna obv aggiunta
        """
        result = df.copy()
        
        if 'close' not in result.columns or 'volume' not in result.columns:
            raise ProcessingError("Colonne close e volume richieste per On Balance Volume")
        
        # Inizializza OBV
        result['obv'] = 0
        
        # Calcola OBV
        for i in range(1, len(result)):
            if result['close'].iloc[i] > result['close'].iloc[i-1]:
                result.loc[result.index[i], 'obv'] = result['obv'].iloc[i-1] + result['volume'].iloc[i]
            elif result['close'].iloc[i] < result['close'].iloc[i-1]:
                result.loc[result.index[i], 'obv'] = result['obv'].iloc[i-1] - result['volume'].iloc[i]
            else:
                result.loc[result.index[i], 'obv'] = result['obv'].iloc[i-1]
        
        # Aggiungi SMA dell'OBV
        result['obv_sma'] = result['obv'].rolling(window=20).mean()
        
        # Segnali di trading
        result['obv_increasing'] = result['obv'] > result['obv'].shift(1)
        result['obv_decreasing'] = result['obv'] < result['obv'].shift(1)
        
        # Divergenza OBV/prezzo
        result['obv_bullish_div'] = (result['close'] < result['close'].shift(1)) & (result['obv'] > result['obv'].shift(1))
        result['obv_bearish_div'] = (result['close'] > result['close'].shift(1)) & (result['obv'] < result['obv'].shift(1))
        
        return result
    
    @staticmethod
    def accumulation_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola l'indicatore Accumulation/Distribution Line.
        
        Args:
            df: DataFrame con colonne high, low, close e volume
            
        Returns:
            DataFrame con colonna ad_line aggiunta
        """
        result = df.copy()
        
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close, volume richieste per Accumulation/Distribution")
        
        # Calcola Money Flow Multiplier
        result['mf_multiplier'] = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (result['high'] - result['low'])
        
        # Gestisci i casi in cui high = low
        result['mf_multiplier'] = result['mf_multiplier'].replace([np.inf, -np.inf], 0)
        result['mf_multiplier'] = result['mf_multiplier'].fillna(0)
        
        # Calcola Money Flow Volume
        result['mf_volume'] = result['mf_multiplier'] * result['volume']
        
        # Calcola Accumulation/Distribution Line
        result['ad_line'] = result['mf_volume'].cumsum()
        
        # Rimuovi colonne temporanee
        result.drop(['mf_multiplier', 'mf_volume'], axis=1, inplace=True)
        
        # Aggiungi SMA dell'AD Line
        result['ad_line_sma'] = result['ad_line'].rolling(window=21).mean()
        
        # Segnali di trading
        result['ad_increasing'] = result['ad_line'] > result['ad_line'].shift(1)
        result['ad_decreasing'] = result['ad_line'] < result['ad_line'].shift(1)
        
        # Divergenza AD Line/prezzo
        result['ad_bullish_div'] = (result['close'] < result['close'].shift(1)) & (result['ad_line'] > result['ad_line'].shift(1))
        result['ad_bearish_div'] = (result['close'] > result['close'].shift(1)) & (result['ad_line'] < result['ad_line'].shift(1))
        
        return result


class PivotIndicators:
    """Indicatori basati sui punti pivot."""
    
    @staticmethod
    def standard_pivots(df: pd.DataFrame, timeframe: str = 'daily') -> pd.DataFrame:
        """
        Calcola i punti pivot standard.
        
        Args:
            df: DataFrame con colonne high, low, close
            timeframe: Intervallo temporale ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame con punti pivot aggiunti
        """
        result = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Punti Pivot")
        
        # Inizializza colonne per punti pivot
        result['pivot_pp'] = np.nan  # Punto Pivot
        result['pivot_r1'] = np.nan  # Resistenza 1
        result['pivot_r2'] = np.nan  # Resistenza 2
        result['pivot_r3'] = np.nan  # Resistenza 3
        result['pivot_s1'] = np.nan  # Supporto 1
        result['pivot_s2'] = np.nan  # Supporto 2
        result['pivot_s3'] = np.nan  # Supporto 3
        
        # Determina come raggruppare i dati in base al timeframe
        if timeframe == 'daily':
            # Per pivot giornalieri, calcola utilizzando i dati del giorno precedente
            for i in range(1, len(result)):
                prev_high = result['high'].iloc[i-1]
                prev_low = result['low'].iloc[i-1]
                prev_close = result['close'].iloc[i-1]
                
                # Calcola Punto Pivot
                pp = (prev_high + prev_low + prev_close) / 3
                
                # Calcola livelli di supporto e resistenza
                r1 = (2 * pp) - prev_low
                r2 = pp + (prev_high - prev_low)
                r3 = r1 + (prev_high - prev_low)
                
                s1 = (2 * pp) - prev_high
                s2 = pp - (prev_high - prev_low)
                s3 = s1 - (prev_high - prev_low)
                
                # Salva i risultati
                result.loc[result.index[i], 'pivot_pp'] = pp
                result.loc[result.index[i], 'pivot_r1'] = r1
                result.loc[result.index[i], 'pivot_r2'] = r2
                result.loc[result.index[i], 'pivot_r3'] = r3
                result.loc[result.index[i], 'pivot_s1'] = s1
                result.loc[result.index[i], 'pivot_s2'] = s2
                result.loc[result.index[i], 'pivot_s3'] = s3
        
        elif timeframe in ['weekly', 'monthly']:
            # Per pivot settimanali/mensili, prima raggruppiamo i dati
            if 'timestamp' not in result.columns:
                raise ProcessingError("Colonna timestamp richiesta per pivot settimanali/mensili")
            
            grouper = pd.Grouper(key='timestamp', freq='W' if timeframe == 'weekly' else 'M')
            grouped = result.groupby(grouper).agg({
                'high': 'max',
                'low': 'min',
                'close': lambda x: x.iloc[-1] if len(x) > 0 else np.nan
            })
            
            # Per ogni periodo, calcola i pivot
            for i in range(1, len(grouped)):
                prev_high = grouped['high'].iloc[i-1]
                prev_low = grouped['low'].iloc[i-1]
                prev_close = grouped['close'].iloc[i-1]
                
                # Calcola Punto Pivot
                pp = (prev_high + prev_low + prev_close) / 3
                
                # Calcola livelli di supporto e resistenza
                r1 = (2 * pp) - prev_low
                r2 = pp + (prev_high - prev_low)
                r3 = r1 + (prev_high - prev_low)
                
                s1 = (2 * pp) - prev_high
                s2 = pp - (prev_high - prev_low)
                s3 = s1 - (prev_high - prev_low)
                
                # Ottieni i timestamp per il periodo corrente
                current_period_start = grouped.index[i]
                current_period_end = grouped.index[i] if i == len(grouped) - 1 else grouped.index[i+1]
                
                # Assegna i valori pivot a tutte le righe nel periodo corrente
                period_mask = (result['timestamp'] >= current_period_start) & (result['timestamp'] < current_period_end)
                result.loc[period_mask, 'pivot_pp'] = pp
                result.loc[period_mask, 'pivot_r1'] = r1
                result.loc[period_mask, 'pivot_r2'] = r2
                result.loc[period_mask, 'pivot_r3'] = r3
                result.loc[period_mask, 'pivot_s1'] = s1
                result.loc[period_mask, 'pivot_s2'] = s2
                result.loc[period_mask, 'pivot_s3'] = s3
        
        else:
            raise ValueError(f"Timeframe non supportato: {timeframe}. Usa 'daily', 'weekly' o 'monthly'.")
        
        # Aggiungi colonne per indicare quando il prezzo è vicino ai livelli pivot
        # Definiamo "vicino" come entro lo 0.5% del livello
        for level in ['pp', 'r1', 'r2', 'r3', 's1', 's2', 's3']:
            distance_pct = 0.005  # 0.5%
            level_col = f'pivot_{level}'
            near_col = f'near_{level}'
            
            result[near_col] = (
                (result['close'] >= result[level_col] * (1 - distance_pct)) & 
                (result['close'] <= result[level_col] * (1 + distance_pct))
            )
        
        # Determina se il prezzo è sopra o sotto il punto pivot
        result['above_pivot_pp'] = result['close'] > result['pivot_pp']
        result['below_pivot_pp'] = result['close'] < result['pivot_pp']
        
        return result
    
    @staticmethod
    def fibonacci_pivots(df: pd.DataFrame, timeframe: str = 'daily') -> pd.DataFrame:
        """
        Calcola i punti pivot con livelli di Fibonacci.
        
        Args:
            df: DataFrame con colonne high, low, close
            timeframe: Intervallo temporale ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame con punti pivot Fibonacci aggiunti
        """
        result = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Punti Pivot Fibonacci")
        
        # Inizializza colonne per punti pivot
        result['fib_pivot_pp'] = np.nan  # Punto Pivot
        result['fib_pivot_r1'] = np.nan  # Resistenza 1 (38.2%)
        result['fib_pivot_r2'] = np.nan  # Resistenza 2 (61.8%)
        result['fib_pivot_r3'] = np.nan  # Resistenza 3 (100%)
        result['fib_pivot_s1'] = np.nan  # Supporto 1 (38.2%)
        result['fib_pivot_s2'] = np.nan  # Supporto 2 (61.8%)
        result['fib_pivot_s3'] = np.nan  # Supporto 3 (100%)
        
        # Livelli di Fibonacci
        fib_38_2 = 0.382
        fib_61_8 = 0.618
        fib_100 = 1.000
        
        # Determina come raggruppare i dati in base al timeframe
        if timeframe == 'daily':
            # Per pivot giornalieri, calcola utilizzando i dati del giorno precedente
            for i in range(1, len(result)):
                prev_high = result['high'].iloc[i-1]
                prev_low = result['low'].iloc[i-1]
                prev_close = result['close'].iloc[i-1]
                
                # Calcola Punto Pivot
                pp = (prev_high + prev_low + prev_close) / 3
                
                # Range del giorno precedente
                prev_range = prev_high - prev_low
                
                # Calcola livelli di supporto e resistenza usando Fibonacci
                r1 = pp + (fib_38_2 * prev_range)
                r2 = pp + (fib_61_8 * prev_range)
                r3 = pp + (fib_100 * prev_range)
                
                s1 = pp - (fib_38_2 * prev_range)
                s2 = pp - (fib_61_8 * prev_range)
                s3 = pp - (fib_100 * prev_range)
                
                # Salva i risultati
                result.loc[result.index[i], 'fib_pivot_pp'] = pp
                result.loc[result.index[i], 'fib_pivot_r1'] = r1
                result.loc[result.index[i], 'fib_pivot_r2'] = r2
                result.loc[result.index[i], 'fib_pivot_r3'] = r3
                result.loc[result.index[i], 'fib_pivot_s1'] = s1
                result.loc[result.index[i], 'fib_pivot_s2'] = s2
                result.loc[result.index[i], 'fib_pivot_s3'] = s3
        
        elif timeframe in ['weekly', 'monthly']:
            # Per pivot settimanali/mensili, prima raggruppiamo i dati
            if 'timestamp' not in result.columns:
                raise ProcessingError("Colonna timestamp richiesta per pivot settimanali/mensili")
            
            grouper = pd.Grouper(key='timestamp', freq='W' if timeframe == 'weekly' else 'M')
            grouped = result.groupby(grouper).agg({
                'high': 'max',
                'low': 'min',
                'close': lambda x: x.iloc[-1] if len(x) > 0 else np.nan
            })
            
            # Per ogni periodo, calcola i pivot
            for i in range(1, len(grouped)):
                prev_high = grouped['high'].iloc[i-1]
                prev_low = grouped['low'].iloc[i-1]
                prev_close = grouped['close'].iloc[i-1]
                
                # Calcola Punto Pivot
                pp = (prev_high + prev_low + prev_close) / 3
                
                # Range del periodo precedente
                prev_range = prev_high - prev_low
                
                # Calcola livelli di supporto e resistenza usando Fibonacci
                r1 = pp + (fib_38_2 * prev_range)
                r2 = pp + (fib_61_8 * prev_range)
                r3 = pp + (fib_100 * prev_range)
                
                s1 = pp - (fib_38_2 * prev_range)
                s2 = pp - (fib_61_8 * prev_range)
                s3 = pp - (fib_100 * prev_range)
                
                # Ottieni i timestamp per il periodo corrente
                current_period_start = grouped.index[i]
                current_period_end = grouped.index[i] if i == len(grouped) - 1 else grouped.index[i+1]
                
                # Assegna i valori pivot a tutte le righe nel periodo corrente
                period_mask = (result['timestamp'] >= current_period_start) & (result['timestamp'] < current_period_end)
                result.loc[period_mask, 'fib_pivot_pp'] = pp
                result.loc[period_mask, 'fib_pivot_r1'] = r1
                result.loc[period_mask, 'fib_pivot_r2'] = r2
                result.loc[period_mask, 'fib_pivot_r3'] = r3
                result.loc[period_mask, 'fib_pivot_s1'] = s1
                result.loc[period_mask, 'fib_pivot_s2'] = s2
                result.loc[period_mask, 'fib_pivot_s3'] = s3
        
        else:
            raise ValueError(f"Timeframe non supportato: {timeframe}. Usa 'daily', 'weekly' o 'monthly'.")
        
        # Aggiungi colonne per indicare quando il prezzo è vicino ai livelli pivot
        # Definiamo "vicino" come entro lo 0.5% del livello
        for level in ['pp', 'r1', 'r2', 'r3', 's1', 's2', 's3']:
            distance_pct = 0.005  # 0.5%
            level_col = f'fib_pivot_{level}'
            near_col = f'near_fib_{level}'
            
            result[near_col] = (
                (result['close'] >= result[level_col] * (1 - distance_pct)) & 
                (result['close'] <= result[level_col] * (1 + distance_pct))
            )
        
        # Determina se il prezzo è sopra o sotto il punto pivot
        result['above_fib_pivot_pp'] = result['close'] > result['fib_pivot_pp']
        result['below_fib_pivot_pp'] = result['close'] < result['fib_pivot_pp']
        
        return result


class MarketStructureIndicators:
    """Indicatori per l'analisi della struttura di mercato."""
    
    @staticmethod
    def smart_money_concepts(df: pd.DataFrame, high_tf_atr_factor: float = 1.5) -> pd.DataFrame:
        """
        Implementa indicatori basati su concetti "Smart Money" (SMC).
        
        Args:
            df: DataFrame con colonne OHLC
            high_tf_atr_factor: Fattore moltiplicativo per l'ATR per identificare zone HTF
            
        Returns:
            DataFrame con indicatori SMC aggiunti
        """
        result = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne OHLC richieste per analisi Smart Money")
        
        # 1. Rileva Equal Highs (EH) e Equal Lows (EL)
        lookback = 15  # Barre per cercare eguali
        eq_threshold = 0.0005  # Soglia per considerare due livelli come eguali (0.05%)
        
        # Inizializza colonne
        result['equal_high'] = False
        result['equal_low'] = False
        
        for i in range(lookback, len(result)):
            # Cerca equal highs
            current_high = result['high'].iloc[i]
            for j in range(i-lookback, i):
                prev_high = result['high'].iloc[j]
                if abs(current_high - prev_high) / prev_high < eq_threshold:
                    result.loc[result.index[i], 'equal_high'] = True
                    break
            
            # Cerca equal lows
            current_low = result['low'].iloc[i]
            for j in range(i-lookback, i):
                prev_low = result['low'].iloc[j]
                if abs(current_low - prev_low) / prev_low < eq_threshold:
                    result.loc[result.index[i], 'equal_low'] = True
                    break
        
        # 2. Rileva Fair Value Gaps (FVG)
        # Bullish FVG: low[i] > high[i-2]
        # Bearish FVG: high[i] < low[i-2]
        result['bullish_fvg'] = False
        result['bearish_fvg'] = False
        
        for i in range(2, len(result)):
            if result['low'].iloc[i] > result['high'].iloc[i-2]:
                result.loc[result.index[i], 'bullish_fvg'] = True
            
            if result['high'].iloc[i] < result['low'].iloc[i-2]:
                result.loc[result.index[i], 'bearish_fvg'] = True
        
        # 3. Rileva Liquidity Levels (zone con potenziale accumulo di ordini)
        # Calcoliamo prima l'ATR per determinare la significatività dei livelli
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift(1))
            ),
            np.abs(result['low'] - result['close'].shift(1))
        )
        result['atr_14'] = result['tr'].rolling(window=14).mean()
        
        # Cerca swing highs e lows che possono rappresentare livelli di liquidità
        swing_lookback = 5
        result['swing_high'] = False
        result['swing_low'] = False
        
        for i in range(swing_lookback, len(result) - swing_lookback):
            if all(result['high'].iloc[i] > result['high'].iloc[i-j] for j in range(1, swing_lookback+1)) and \
               all(result['high'].iloc[i] > result['high'].iloc[i+j] for j in range(1, swing_lookback+1)):
                result.loc[result.index[i], 'swing_high'] = True
            
            if all(result['low'].iloc[i] < result['low'].iloc[i-j] for j in range(1, swing_lookback+1)) and \
               all(result['low'].iloc[i] < result['low'].iloc[i+j] for j in range(1, swing_lookback+1)):
                result.loc[result.index[i], 'swing_low'] = True
        
        # Identifica zone di "High Time Frame" (HTF) liquidità
        # Queste sono aree con swing high/low significativi (maggiori dell'ATR medio)
        result['htf_liquidity_high'] = False
        result['htf_liquidity_low'] = False
        
        for i in range(swing_lookback + 1, len(result)):
            if result['swing_high'].iloc[i-1]:
                # Calcola la dimensione dello swing rispetto all'ATR
                swing_size = result['high'].iloc[i-1] - result['close'].iloc[i-1]
                if swing_size > result['atr_14'].iloc[i-1] * high_tf_atr_factor:
                    result.loc[result.index[i], 'htf_liquidity_high'] = True
            
            if result['swing_low'].iloc[i-1]:
                # Calcola la dimensione dello swing rispetto all'ATR
                swing_size = result['close'].iloc[i-1] - result['low'].iloc[i-1]
                if swing_size > result['atr_14'].iloc[i-1] * high_tf_atr_factor:
                    result.loc[result.index[i], 'htf_liquidity_low'] = True
        
        # 4. Rileva Order Blocks (OB)
        # Bullish OB: candela ribassista prima di un movimento verso l'alto
        # Bearish OB: candela rialzista prima di un movimento verso il basso
        result['bullish_ob'] = False
        result['bearish_ob'] = False
        
        for i in range(1, len(result) - 1):
            # Bullish OB: candela ribassista (close < open) seguita da forte movimento rialzista
            if result['close'].iloc[i] < result['open'].iloc[i] and \
               result['close'].iloc[i+1] > result['high'].iloc[i] and \
               (result['close'].iloc[i+1] - result['open'].iloc[i+1]) > result['atr_14'].iloc[i]:
                result.loc[result.index[i], 'bullish_ob'] = True
            
            # Bearish OB: candela rialzista (close > open) seguita da forte movimento ribassista
            if result['close'].iloc[i] > result['open'].iloc[i] and \
               result['close'].iloc[i+1] < result['low'].iloc[i] and \
               (result['open'].iloc[i+1] - result['close'].iloc[i+1]) > result['atr_14'].iloc[i]:
                result.loc[result.index[i], 'bearish_ob'] = True
        
        # 5. Rileva Imbalances (squilibri tra la pressione dell'acquirente e del venditore)
        # Un grave squilibrio può essere indicato da una grande candela body rispetto all'ATR
        result['buyer_imbalance'] = (result['close'] - result['open']) > result['atr_14'] * 1.2
        result['seller_imbalance'] = (result['open'] - result['close']) > result['atr_14'] * 1.2
        
        # 6. Rileva cambio di struttura
        # Higher High & Higher Low (Trend rialzista) o Lower High & Lower Low (Trend ribassista)
        result['higher_high'] = False
        result['higher_low'] = False
        result['lower_high'] = False
        result['lower_low'] = False
        
        # Trova i minimi e massimi precedenti
        for i in range(swing_lookback*2, len(result)):
            # Cerca il massimo precedente
            prev_highs = [result['high'].iloc[j] for j in range(i-swing_lookback*2, i) if result['swing_high'].iloc[j]]
            if prev_highs and result['swing_high'].iloc[i]:
                if result['high'].iloc[i] > max(prev_highs):
                    result.loc[result.index[i], 'higher_high'] = True
                if result['high'].iloc[i] < max(prev_highs):
                    result.loc[result.index[i], 'lower_high'] = True
            
            # Cerca il minimo precedente
            prev_lows = [result['low'].iloc[j] for j in range(i-swing_lookback*2, i) if result['swing_low'].iloc[j]]
            if prev_lows and result['swing_low'].iloc[i]:
                if result['low'].iloc[i] > min(prev_lows):
                    result.loc[result.index[i], 'higher_low'] = True
                if result['low'].iloc[i] < min(prev_lows):
                    result.loc[result.index[i], 'lower_low'] = True
        
        # Rimuovi colonne temporanee
        result.drop(['tr'], axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def detect_demand_supply_zones(df: pd.DataFrame, atr_mult: float = 1.5) -> pd.DataFrame:
        """
        Rileva zone di domanda e offerta.
        
        Args:
            df: DataFrame con colonne OHLC
            atr_mult: Moltiplicatore ATR per determinare la forza della zona
            
        Returns:
            DataFrame con zone di domanda/offerta aggiunte
        """
        result = df.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne OHLC richieste per rilevare zone di domanda/offerta")
        
        # Calcola l'ATR per determinare le candele significative
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift(1))
            ),
            np.abs(result['low'] - result['close'].shift(1))
        )
        result['atr_14'] = result['tr'].rolling(window=14).mean()
        
        # Identifica candele significative (basate sull'ATR)
        result['significant_bullish'] = (result['close'] - result['open']) > (result['atr_14'] * atr_mult)
        result['significant_bearish'] = (result['open'] - result['close']) > (result['atr_14'] * atr_mult)
        
        # Inizializza colonne per zone di domanda/offerta
        result['demand_zone_top'] = np.nan
        result['demand_zone_bottom'] = np.nan
        result['supply_zone_top'] = np.nan
        result['supply_zone_bottom'] = np.nan
        
        # Una zona di domanda è creata dopo una forte candela rialzista
        # Una zona di offerta è creata dopo una forte candela ribassista
        
        for i in range(1, len(result) - 1):
            # Zona di domanda (base di una forte candela rialzista)
            if result['significant_bullish'].iloc[i]:
                result.loc[result.index[i+1], 'demand_zone_top'] = result['open'].iloc[i]
                result.loc[result.index[i+1], 'demand_zone_bottom'] = min(result['low'].iloc[i], result['low'].iloc[i-1])
            
            # Zona di offerta (cima di una forte candela ribassista)
            if result['significant_bearish'].iloc[i]:
                result.loc[result.index[i+1], 'supply_zone_top'] = max(result['high'].iloc[i], result['high'].iloc[i-1])
                result.loc[result.index[i+1], 'supply_zone_bottom'] = result['open'].iloc[i]
        
        # Identifica quando il prezzo è dentro una zona di domanda/offerta
        result['in_demand_zone'] = (
            ~result['demand_zone_top'].isna() & 
            (result['low'] <= result['demand_zone_top']) & 
            (result['high'] >= result['demand_zone_bottom'])
        )
        
        result['in_supply_zone'] = (
            ~result['supply_zone_top'].isna() & 
            (result['high'] >= result['supply_zone_bottom']) & 
            (result['low'] <= result['supply_zone_top'])
        )
        
        # Rimuovi colonne temporanee
        result.drop(['tr', 'significant_bullish', 'significant_bearish'], axis=1, inplace=True)
        
        return result


class CustomIndicators:
    """Indicatori personalizzati per strategie specifiche."""
    
    @staticmethod
    def hull_moving_average(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calcola Hull Moving Average (HMA), un indicatore di trend più reattivo.
        
        Args:
            df: DataFrame con colonna close
            period: Periodo per l'HMA
            
        Returns:
            DataFrame con colonna hull_ma aggiunta
        """
        result = df.copy()
        
        if 'close' not in result.columns:
            raise ProcessingError("Colonna close richiesta per Hull Moving Average")
        
        # Calcola due WMA con periodi diversi
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = result['close'].rolling(window=half_period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
        )
        
        wma_full = result['close'].rolling(window=period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
        )
        
        # Calcola la differenza tra le due WMA
        result['hull_raw'] = 2 * wma_half - wma_full
        
        # Calcola WMA della differenza con periodo sqrt(n)
        result[f'hull_ma_{period}'] = result['hull_raw'].rolling(window=sqrt_period).apply(
            lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
        )
        
        # Rimuovi colonna temporanea
        result.drop(['hull_raw'], axis=1, inplace=True)
        
        # Aggiungi segnali
        result[f'hull_ma_{period}_uptrend'] = result[f'hull_ma_{period}'] > result[f'hull_ma_{period}'].shift(1)
        result[f'hull_ma_{period}_downtrend'] = result[f'hull_ma_{period}'] < result[f'hull_ma_{period}'].shift(1)
        
        return result
    
    @staticmethod
    def mean_reversion_bands(df: pd.DataFrame, lookback: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calcola bande di mean reversion per strategie di ritorno alla media.
        
        Args:
            df: DataFrame con colonna close
            lookback: Periodo di lookback
            std_dev: Deviazione standard per le bande
            
        Returns:
            DataFrame con bande di mean reversion aggiunte
        """
        result = df.copy()
        
        if 'close' not in result.columns:
            raise ProcessingError("Colonna close richiesta per Mean Reversion Bands")
        
        # Calcola la media mobile
        result[f'mr_ma_{lookback}'] = result['close'].rolling(window=lookback).mean()
        
        # Calcola la deviazione standard
        result[f'mr_std_{lookback}'] = result['close'].rolling(window=lookback).std()
        
        # Calcola le bande
        result[f'mr_upper_{lookback}'] = result[f'mr_ma_{lookback}'] + (result[f'mr_std_{lookback}'] * std_dev)
        result[f'mr_lower_{lookback}'] = result[f'mr_ma_{lookback}'] - (result[f'mr_std_{lookback}'] * std_dev)
        
        # Calcola l'indicatore di mean reversion
        result[f'mr_indicator_{lookback}'] = (result['close'] - result[f'mr_ma_{lookback}']) / (result[f'mr_std_{lookback}'] * std_dev)
        
        # Segnali di trading
        result[f'mr_overbought_{lookback}'] = result[f'mr_indicator_{lookback}'] > 0.8
        result[f'mr_oversold_{lookback}'] = result[f'mr_indicator_{lookback}'] < -0.8
        
        # Rimuovi colonna temporanea
        result.drop([f'mr_std_{lookback}'], axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def squeeze_momentum(df: pd.DataFrame, bb_length: int = 20, kc_length: int = 20, 
                        mult: float = 1.5) -> pd.DataFrame:
        """
        Implementa l'indicatore Squeeze Momentum (Lazybear).
        
        Args:
            df: DataFrame con colonne OHLC
            bb_length: Periodo per Bollinger Bands
            kc_length: Periodo per Keltner Channel
            mult: Moltiplicatore per le bande
            
        Returns:
            DataFrame con indicatore squeeze momentum aggiunto
        """
        result = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Squeeze Momentum")
        
        # Calcola Bollinger Bands
        result['basis'] = result['close'].rolling(window=bb_length).mean()
        result['dev'] = mult * result['close'].rolling(window=bb_length).std()
        
        result['upper_bb'] = result['basis'] + result['dev']
        result['lower_bb'] = result['basis'] - result['dev']
        
        # Calcola True Range per Keltner Channel
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift(1))
            ),
            np.abs(result['low'] - result['close'].shift(1))
        )
        
        result['atr'] = result['tr'].rolling(window=kc_length).mean()
        
        # Calcola Keltner Channel
        result['upper_kc'] = result['basis'] + (result['atr'] * mult)
        result['lower_kc'] = result['basis'] - (result['atr'] * mult)
        
        # Determina se il mercato è "in squeeze"
        result['squeeze_on'] = (result['lower_bb'] > result['lower_kc']) & (result['upper_bb'] < result['upper_kc'])
        
        # Calcola il momentum
        result['mom'] = result['close'] - ((result['high'] + result['low']) / 2 + result['close']).shift(bb_length)
        
        # Determina se il momentum è in aumento o in diminuzione
        result['mom_increasing'] = result['mom'] > result['mom'].shift(1)
        result['mom_decreasing'] = result['mom'] < result['mom'].shift(1)
        
        # Rimuovi colonne temporanee
        columns_to_drop = ['basis', 'dev', 'tr', 'atr']
        result.drop(columns_to_drop, axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def elder_ray(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """
        Implementa l'indicatore Elder Ray.
        
        Args:
            df: DataFrame con colonne high, low, close
            period: Periodo per l'EMA
            
        Returns:
            DataFrame con Bull Power e Bear Power aggiunti
        """
        result = df.copy()
        
        required_cols = ['high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne high, low, close richieste per Elder Ray")
        
        # Calcola l'EMA
        result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
        
        # Calcola Bull Power e Bear Power
        result['bull_power'] = result['high'] - result[f'ema_{period}']
        result['bear_power'] = result['low'] - result[f'ema_{period}']
        
        # Identifica segnali di trading
        result['elder_ray_bullish'] = (result['bull_power'] > 0) & (result['bear_power'] < 0) & (result['bear_power'] > result['bear_power'].shift(1))
        result['elder_ray_bearish'] = (result['bull_power'] < 0) & (result['bear_power'] < 0) & (result['bull_power'] < result['bull_power'].shift(1))
        
        return result


# Factory function per accedere agli indicatori
def get_indicators():
    """
    Ottiene un'istanza di ogni classe di indicatori.
    
    Returns:
        Dizionario con istanze di classi di indicatori
    """
    return {
        'trend': TrendIndicators(),
        'momentum': MomentumIndicators(),
        'volume': VolumeIndicators(),
        'pivot': PivotIndicators(),
        'structure': MarketStructureIndicators(),
        'custom': CustomIndicators()
    }