# Pattern di mercato

"""
Riconoscimento di pattern di mercato per il progetto TradingLab.
Questo modulo fornisce funzioni per identificare pattern candlestick, harmonici e altre formazioni.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import re
from datetime import datetime, timedelta

# Importazioni dal modulo utils
from ..utils import app_logger, ProcessingError, time_it


class CandlePatterns:
    """Pattern di candele giapponesi."""
    
    @staticmethod
    def recognizer(df: pd.DataFrame, body_ratio: float = 0.3, 
                  doji_ratio: float = 0.05) -> pd.DataFrame:
        """
        Riconosce i pattern di candele più comuni.
        
        Args:
            df: DataFrame con colonne OHLC
            body_ratio: Rapporto per considerare una candela con corpo lungo/corto
            doji_ratio: Rapporto per considerare una candela doji
            
        Returns:
            DataFrame con pattern di candele riconosciuti
        """
        result = df.copy()
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in result.columns for col in required_cols):
            raise ProcessingError("Colonne OHLC richieste per riconoscere pattern di candele")
        
        # Calcola le caratteristiche delle candele
        result['body'] = np.abs(result['close'] - result['open'])
        result['range'] = result['high'] - result['low']
        result['body_perc'] = result['body'] / result['range']
        
        result['upper_wick'] = result['high'] - np.maximum(result['open'], result['close'])
        result['lower_wick'] = np.minimum(result['open'], result['close']) - result['low']
        
        result['upper_wick_perc'] = result['upper_wick'] / result['range']
        result['lower_wick_perc'] = result['lower_wick'] / result['range']
        
        result['bullish'] = result['close'] > result['open']
        result['bearish'] = result['close'] < result['open']
        
        # 1. Pattern a Candela Singola
        
        # Doji: corpo molto piccolo
        result['doji'] = result['body_perc'] <= doji_ratio
        
        # Dragonfly Doji: doji con ombra inferiore lunga
        result['dragonfly_doji'] = (
            result['doji'] & 
            (result['lower_wick_perc'] > 0.6) & 
            (result['upper_wick_perc'] < 0.1)
        )
        
        # Gravestone Doji: doji con ombra superiore lunga
        result['gravestone_doji'] = (
            result['doji'] & 
            (result['upper_wick_perc'] > 0.6) & 
            (result['lower_wick_perc'] < 0.1)
        )
        
        # Hammer: corpo piccolo in alto con lunga ombra inferiore
        result['hammer'] = (
            ~result['doji'] &
            (result['lower_wick'] > 2 * result['body']) &
            (result['upper_wick'] < 0.2 * result['body']) &
            (result['body_perc'] < 0.4)
        )
        
        # Hanging Man: simile a hammer ma in trend rialzista
        result['hanging_man'] = result['hammer']  # Stessi criteri, il contesto fa la differenza
        
        # Inverted Hammer: corpo piccolo in basso con lunga ombra superiore
        result['inverted_hammer'] = (
            ~result['doji'] &
            (result['upper_wick'] > 2 * result['body']) &
            (result['lower_wick'] < 0.2 * result['body']) &
            (result['body_perc'] < 0.4)
        )
        
        # Shooting Star: simile a inverted hammer ma in trend rialzista
        result['shooting_star'] = result['inverted_hammer']  # Stessi criteri, il contesto fa la differenza
        
        # Marubozu: candela con corpo lungo senza o con piccolissime ombre
        result['marubozu'] = (
            (result['body_perc'] > 0.8) &
            (result['upper_wick_perc'] < 0.05) &
            (result['lower_wick_perc'] < 0.05)
        )
        
        # 2. Pattern a Due Candele
        
        # Bullish Engulfing
        result['bullish_engulfing'] = False
        # Bearish Engulfing
        result['bearish_engulfing'] = False
        # Tweezer Top
        result['tweezer_top'] = False
        # Tweezer Bottom
        result['tweezer_bottom'] = False
        
        # Aggiungi pattern a due candele
        for i in range(1, len(result)):
            # Bullish Engulfing: candela ribassista seguita da candela rialzista che la ingloba
            if (result['bearish'].iloc[i-1] and 
                result['bullish'].iloc[i] and 
                result['open'].iloc[i] <= result['close'].iloc[i-1] and 
                result['close'].iloc[i] >= result['open'].iloc[i-1]):
                result.loc[result.index[i], 'bullish_engulfing'] = True
            
            # Bearish Engulfing: candela rialzista seguita da candela ribassista che la ingloba
            if (result['bullish'].iloc[i-1] and 
                result['bearish'].iloc[i] and 
                result['open'].iloc[i] >= result['close'].iloc[i-1] and 
                result['close'].iloc[i] <= result['open'].iloc[i-1]):
                result.loc[result.index[i], 'bearish_engulfing'] = True
            
            # Tweezer Top: due candele con high molto simili in zona di resistenza
            high_diff_pct = abs(result['high'].iloc[i] - result['high'].iloc[i-1]) / result['high'].iloc[i-1]
            if (high_diff_pct < 0.001 and 
                result['bullish'].iloc[i-1] and 
                result['bearish'].iloc[i]):
                result.loc[result.index[i], 'tweezer_top'] = True
            
            # Tweezer Bottom: due candele con low molto simili in zona di supporto
            low_diff_pct = abs(result['low'].iloc[i] - result['low'].iloc[i-1]) / result['low'].iloc[i-1]
            if (low_diff_pct < 0.001 and 
                result['bearish'].iloc[i-1] and 
                result['bullish'].iloc[i]):
                result.loc[result.index[i], 'tweezer_bottom'] = True
        
        # 3. Pattern a Tre Candele
        
        # Evening Star
        result['evening_star'] = False
        # Morning Star
        result['morning_star'] = False
        # Three White Soldiers
        result['three_white_soldiers'] = False
        # Three Black Crows
        result['three_black_crows'] = False
        
        # Aggiungi pattern a tre candele
        for i in range(2, len(result)):
            # Evening Star
            if (result['bullish'].iloc[i-2] and 
                result['body_perc'].iloc[i-2] > body_ratio and
                result['body_perc'].iloc[i-1] < body_ratio and
                result['bearish'].iloc[i] and 
                result['body_perc'].iloc[i] > body_ratio and
                result['open'].iloc[i-1] > result['close'].iloc[i-2] and
                result['close'].iloc[i] < result['open'].iloc[i-2]):
                result.loc[result.index[i], 'evening_star'] = True
            
            # Morning Star
            if (result['bearish'].iloc[i-2] and 
                result['body_perc'].iloc[i-2] > body_ratio and
                result['body_perc'].iloc[i-1] < body_ratio and
                result['bullish'].iloc[i] and 
                result['body_perc'].iloc[i] > body_ratio and
                result['open'].iloc[i-1] < result['close'].iloc[i-2] and
                result['close'].iloc[i] > result['open'].iloc[i-2]):
                result.loc[result.index[i], 'morning_star'] = True
            
            # Three White Soldiers
            if (result['bullish'].iloc[i-2] and 
                result['bullish'].iloc[i-1] and 
                result['bullish'].iloc[i] and
                result['body_perc'].iloc[i-2] > body_ratio and
                result['body_perc'].iloc[i-1] > body_ratio and
                result['body_perc'].iloc[i] > body_ratio and
                result['open'].iloc[i-1] > result['open'].iloc[i-2] and
                result['open'].iloc[i] > result['open'].iloc[i-1] and
                result['close'].iloc[i-1] > result['close'].iloc[i-2] and
                result['close'].iloc[i] > result['close'].iloc[i-1]):
                result.loc[result.index[i], 'three_white_soldiers'] = True
            
            # Three Black Crows
            if (result['bearish'].iloc[i-2] and 
                result['bearish'].iloc[i-1] and 
                result['bearish'].iloc[i] and
                result['body_perc'].iloc[i-2] > body_ratio and
                result['body_perc'].iloc[i-1] > body_ratio and
                result['body_perc'].iloc[i] > body_ratio and
                result['open'].iloc[i-1] < result['open'].iloc[i-2] and
                result['open'].iloc[i] < result['open'].iloc[i-1] and
                result['close'].iloc[i-1] < result['close'].iloc[i-2] and
                result['close'].iloc[i] < result['close'].iloc[i-1]):
                result.loc[result.index[i], 'three_black_crows'] = True
        
        # Rimuovi colonne temporanee
        cols_to_remove = ['body', 'range', 'body_perc', 'upper_wick', 'lower_wick', 
                         'upper_wick_perc', 'lower_wick_perc']
        result.drop(cols_to_remove, axis=1, inplace=True)
        
        # Aggiungi colonne di riepilogo
        result['bullish_pattern'] = (
            result['hammer'] | result['bullish_engulfing'] | 
            result['morning_star'] | result['three_white_soldiers'] |
            result['tweezer_bottom'] | result['dragonfly_doji']
        )
        
        result['bearish_pattern'] = (
            result['hanging_man'] | result['shooting_star'] | 
            result['bearish_engulfing'] | result['evening_star'] | 
            result['three_black_crows'] | result['tweezer_top'] |
            result['gravestone_doji']
        )
        
        return result
    
    @staticmethod
    def detect_inside_bars(df: pd.DataFrame, consecutive: int = 1) -> pd.DataFrame:
        """
        Rileva Inside Bars (barre completamente contenute nella precedente).
        
        Args:
            df: DataFrame con colonne high e low
            consecutive: Numero minimo di inside bars consecutive
            
        Returns:
            DataFrame con colonna inside_bar aggiunta
        """
        result = df.copy()
        
        if 'high' not in result.columns or 'low' not in result.columns:
            raise ProcessingError("Colonne high e low richieste per rilevare Inside Bars")
        
        # Inizializza la colonna
        result['inside_bar'] = False
        
        # Rileva inside bars
        for i in range(1, len(result)):
            if (result['high'].iloc[i] <= result['high'].iloc[i-1] and 
                result['low'].iloc[i] >= result['low'].iloc[i-1]):
                result.loc[result.index[i], 'inside_bar'] = True
        
        """
        Rileva Inside Bars (barre completamente contenute nella precedente).
        
        Args:
            df: DataFrame con colonne high e low
            consecutive: Numero minimo di inside bars consecutive
            
        Returns:
            DataFrame con colonna inside_bar aggiunta
        """
        result = df.copy()
        
        if 'high' not in result.columns or 'low' not in result.columns:
            raise ProcessingError("Colonne high e low richieste per rilevare Inside Bars")
        
        # Inizializza la colonna
        result['inside_bar'] = False
        result['inside_bar_count'] = 0
        
        # Rileva inside bars
        for i in range(1, len(result)):
            if (result['high'].iloc[i] <= result['high'].iloc[i-1] and 
                result['low'].iloc[i] >= result['low'].iloc[i-1]):
                result.loc[result.index[i], 'inside_bar'] = True
                result.loc[result.index[i], 'inside_bar_count'] = result['inside_bar_count'].iloc[i-1] + 1
            else:
                result.loc[result.index[i], 'inside_bar_count'] = 0
        
        # Identifica inside bars consecutive
        result[f'consecutive_inside_bar_{consecutive}'] = result['inside_bar_count'] >= consecutive
        
        return result
    
    @staticmethod
    def detect_outside_bars(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rileva Outside Bars (barre che inglobano completamente la precedente).
        
        Args:
            df: DataFrame con colonne high e low
            
        Returns:
            DataFrame con colonna outside_bar aggiunta
        """
        result = df.copy()
        
        if 'high' not in result.columns or 'low' not in result.columns:
            raise ProcessingError("Colonne high e low richieste per rilevare Outside Bars")
        
        # Inizializza la colonna
        result['outside_bar'] = False
        
        # Rileva outside bars
        for i in range(1, len(result)):
            if (result['high'].iloc[i] > result['high'].iloc[i-1] and 
                result['low'].iloc[i] < result['low'].iloc[i-1]):
                result.loc[result.index[i], 'outside_bar'] = True
        
        # Classifica outside bars come bullish o bearish in base alla chiusura
        result['bullish_outside'] = result['outside_bar'] & (result['close'] > result['open'])
        result['bearish_outside'] = result['outside_bar'] & (result['close'] < result['open'])
        
        return result


class HarmonicPatterns:
    """Pattern armonici come Gartley, Butterfly, ecc."""
    
    @staticmethod
    def is_valid_harmonic_move(move: float, target: float, tolerance: float = 0.05) -> bool:
        """
        Verifica se un movimento (ratio) corrisponde a un target Fibonacci.
        
        Args:
            move: Movimento misurato
            target: Valore target Fibonacci
            tolerance: Tolleranza permessa
            
        Returns:
            True se il movimento è valido
        """
        return abs(move - target) <= tolerance
    
    @staticmethod
    def find_pivots(df: pd.DataFrame, n_bars: int = 5) -> pd.DataFrame:
        """
        Trova i punti pivot (swing high/low) per l'analisi armonica.
        
        Args:
            df: DataFrame con colonne high e low
            n_bars: Numero di barre per definire uno swing
            
        Returns:
            DataFrame con colonne pivot_high e pivot_low aggiunte
        """
        result = df.copy()
        
        result['pivot_high'] = False
        result['pivot_low'] = False
        
        # Ottieni gli indici per n_bars da sinistra e da destra
        for i in range(n_bars, len(result) - n_bars):
            # Definisci intervalli
            left_bars = result.iloc[i-n_bars:i]
            right_bars = result.iloc[i+1:i+n_bars+1]
            current = result.iloc[i]
            
            # Verifica pivot high
            if (current['high'] > left_bars['high'].max() and 
                current['high'] > right_bars['high'].max()):
                result.loc[result.index[i], 'pivot_high'] = True
            
            # Verifica pivot low
            if (current['low'] < left_bars['low'].min() and 
                current['low'] < right_bars['low'].min()):
                result.loc[result.index[i], 'pivot_low'] = True
        
        return result
    
    @staticmethod
    def detect_gartley_pattern(df: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
        """
        Rileva pattern Gartley bullish e bearish.
        
        Args:
            df: DataFrame con colonne OHLC, pivot_high e pivot_low
            tolerance: Tolleranza per i ratio Fibonacci
            
        Returns:
            DataFrame con pattern Gartley aggiunti
        """
        result = df.copy()
        
        # Verifica che le colonne dei pivot siano presenti
        if 'pivot_high' not in result.columns or 'pivot_low' not in result.columns:
            result = HarmonicPatterns.find_pivots(result)
        
        # Inizializza colonne per i pattern
        result['bullish_gartley'] = False
        result['bearish_gartley'] = False
        
        # Ottieni indici di pivot high e low
        pivot_highs = result[result['pivot_high']].index
        pivot_lows = result[result['pivot_low']].index
        
        # Per identificare un pattern Gartley, servono 5 punti (X, A, B, C, D)
        # Dobbiamo trovare le sequenze di pivot high/low che soddisfano i ratio Fibonacci
        
        # Bullish Gartley (inizia con pivot low a X)
        for i in range(len(pivot_lows) - 4):
            x_idx = result.index.get_loc(pivot_lows[i])
            
            # Cerca i punti corrispondenti A, B, C, D
            # A è un pivot high dopo X
            a_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > x_idx]
            if not a_candidates:
                continue
                
            a_idx = result.index.get_loc(a_candidates[0])
            
            # B è un pivot low dopo A
            b_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > a_idx]
            if not b_candidates:
                continue
                
            b_idx = result.index.get_loc(b_candidates[0])
            
            # C è un pivot high dopo B
            c_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > b_idx]
            if not c_candidates:
                continue
                
            c_idx = result.index.get_loc(c_candidates[0])
            
            # D è un pivot low dopo C
            d_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > c_idx]
            if not d_candidates:
                continue
                
            d_idx = result.index.get_loc(d_candidates[0])
            
            # Calcola le diverse "gambe" o movimenti
            xa = abs(result['high'].iloc[a_idx] - result['low'].iloc[x_idx])
            ab = abs(result['low'].iloc[b_idx] - result['high'].iloc[a_idx])
            bc = abs(result['high'].iloc[c_idx] - result['low'].iloc[b_idx])
            cd = abs(result['low'].iloc[d_idx] - result['high'].iloc[c_idx])
            xd = abs(result['low'].iloc[d_idx] - result['low'].iloc[x_idx])
            
            # Calcola i ratio
            ab_xa_ratio = ab / xa if xa != 0 else 0
            bc_ab_ratio = bc / ab if ab != 0 else 0
            cd_bc_ratio = cd / bc if bc != 0 else 0
            xd_xa_ratio = xd / xa if xa != 0 else 0
            
            # Verifica i criteri per il pattern Gartley
            # AB = 0.618 di XA
            # BC = 0.382-0.886 di AB
            # CD = 1.272-1.618 di BC
            # AD = 0.786 di XA
            is_gartley = (
                HarmonicPatterns.is_valid_harmonic_move(ab_xa_ratio, 0.618, tolerance) and
                (HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.382, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.886, tolerance)) and
                (HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 1.272, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 1.618, tolerance)) and
                HarmonicPatterns.is_valid_harmonic_move(xd_xa_ratio, 0.786, tolerance)
            )
            
            if is_gartley:
                result.loc[result.index[d_idx], 'bullish_gartley'] = True
        
        # Bearish Gartley (inizia con pivot high a X)
        for i in range(len(pivot_highs) - 4):
            x_idx = result.index.get_loc(pivot_highs[i])
            
            # A è un pivot low dopo X
            a_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > x_idx]
            if not a_candidates:
                continue
                
            a_idx = result.index.get_loc(a_candidates[0])
            
            # B è un pivot high dopo A
            b_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > a_idx]
            if not b_candidates:
                continue
                
            b_idx = result.index.get_loc(b_candidates[0])
            
            # C è un pivot low dopo B
            c_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > b_idx]
            if not c_candidates:
                continue
                
            c_idx = result.index.get_loc(c_candidates[0])
            
            # D è un pivot high dopo C
            d_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > c_idx]
            if not d_candidates:
                continue
                
            d_idx = result.index.get_loc(d_candidates[0])
            
            # Calcola le diverse "gambe" o movimenti
            xa = abs(result['low'].iloc[a_idx] - result['high'].iloc[x_idx])
            ab = abs(result['high'].iloc[b_idx] - result['low'].iloc[a_idx])
            bc = abs(result['low'].iloc[c_idx] - result['high'].iloc[b_idx])
            cd = abs(result['high'].iloc[d_idx] - result['low'].iloc[c_idx])
            xd = abs(result['high'].iloc[d_idx] - result['high'].iloc[x_idx])
            
            # Calcola i ratio
            ab_xa_ratio = ab / xa if xa != 0 else 0
            bc_ab_ratio = bc / ab if ab != 0 else 0
            cd_bc_ratio = cd / bc if bc != 0 else 0
            xd_xa_ratio = xd / xa if xa != 0 else 0
            
            # Verifica i criteri
            is_gartley = (
                HarmonicPatterns.is_valid_harmonic_move(ab_xa_ratio, 0.618, tolerance) and
                (HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.382, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.886, tolerance)) and
                (HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 1.272, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 1.618, tolerance)) and
                HarmonicPatterns.is_valid_harmonic_move(xd_xa_ratio, 0.786, tolerance)
            )
            
            if is_gartley:
                result.loc[result.index[d_idx], 'bearish_gartley'] = True
        
        return result
    
    @staticmethod
    def detect_butterfly_pattern(df: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
        """
        Rileva pattern Butterfly bullish e bearish.
        
        Args:
            df: DataFrame con colonne OHLC, pivot_high e pivot_low
            tolerance: Tolleranza per i ratio Fibonacci
            
        Returns:
            DataFrame con pattern Butterfly aggiunti
        """
        result = df.copy()
        
        # Verifica che le colonne dei pivot siano presenti
        if 'pivot_high' not in result.columns or 'pivot_low' not in result.columns:
            result = HarmonicPatterns.find_pivots(result)
        
        # Inizializza colonne per i pattern
        result['bullish_butterfly'] = False
        result['bearish_butterfly'] = False
        
        # Ottieni indici di pivot high e low
        pivot_highs = result[result['pivot_high']].index
        pivot_lows = result[result['pivot_low']].index
        
        # Bullish Butterfly (inizia con pivot low a X)
        for i in range(len(pivot_lows) - 4):
            x_idx = result.index.get_loc(pivot_lows[i])
            
            # Cerca i punti corrispondenti come prima
            a_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > x_idx]
            if not a_candidates:
                continue
                
            a_idx = result.index.get_loc(a_candidates[0])
            
            b_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > a_idx]
            if not b_candidates:
                continue
                
            b_idx = result.index.get_loc(b_candidates[0])
            
            c_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > b_idx]
            if not c_candidates:
                continue
                
            c_idx = result.index.get_loc(c_candidates[0])
            
            d_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > c_idx]
            if not d_candidates:
                continue
                
            d_idx = result.index.get_loc(d_candidates[0])
            
            # Calcola le diverse "gambe" come prima
            xa = abs(result['high'].iloc[a_idx] - result['low'].iloc[x_idx])
            ab = abs(result['low'].iloc[b_idx] - result['high'].iloc[a_idx])
            bc = abs(result['high'].iloc[c_idx] - result['low'].iloc[b_idx])
            cd = abs(result['low'].iloc[d_idx] - result['high'].iloc[c_idx])
            xd = abs(result['low'].iloc[d_idx] - result['low'].iloc[x_idx])
            
            # Calcola i ratio
            ab_xa_ratio = ab / xa if xa != 0 else 0
            bc_ab_ratio = bc / ab if ab != 0 else 0
            cd_bc_ratio = cd / bc if bc != 0 else 0
            xd_xa_ratio = xd / xa if xa != 0 else 0
            
            # Verifica i criteri per il pattern Butterfly:
            # AB = 0.786 di XA
            # BC = 0.382-0.886 di AB
            # CD = 1.618-2.618 di BC
            # AD = 1.27-1.618 di XA
            is_butterfly = (
                HarmonicPatterns.is_valid_harmonic_move(ab_xa_ratio, 0.786, tolerance) and
                (HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.382, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.886, tolerance)) and
                (HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 1.618, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 2.618, tolerance)) and
                (HarmonicPatterns.is_valid_harmonic_move(xd_xa_ratio, 1.27, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(xd_xa_ratio, 1.618, tolerance))
            )
            
            if is_butterfly:
                result.loc[result.index[d_idx], 'bullish_butterfly'] = True
        
        # Bearish Butterfly (inizia con pivot high a X)
        for i in range(len(pivot_highs) - 4):
            x_idx = result.index.get_loc(pivot_highs[i])
            
            # Cerca i punti corrispondenti come prima
            a_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > x_idx]
            if not a_candidates:
                continue
                
            a_idx = result.index.get_loc(a_candidates[0])
            
            b_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > a_idx]
            if not b_candidates:
                continue
                
            b_idx = result.index.get_loc(b_candidates[0])
            
            c_candidates = [pl for pl in pivot_lows if result.index.get_loc(pl) > b_idx]
            if not c_candidates:
                continue
                
            c_idx = result.index.get_loc(c_candidates[0])
            
            d_candidates = [ph for ph in pivot_highs if result.index.get_loc(ph) > c_idx]
            if not d_candidates:
                continue
                
            d_idx = result.index.get_loc(d_candidates[0])
            
            # Calcola le diverse "gambe" come prima
            xa = abs(result['low'].iloc[a_idx] - result['high'].iloc[x_idx])
            ab = abs(result['high'].iloc[b_idx] - result['low'].iloc[a_idx])
            bc = abs(result['low'].iloc[c_idx] - result['high'].iloc[b_idx])
            cd = abs(result['high'].iloc[d_idx] - result['low'].iloc[c_idx])
            xd = abs(result['high'].iloc[d_idx] - result['high'].iloc[x_idx])
            
            # Calcola i ratio
            ab_xa_ratio = ab / xa if xa != 0 else 0
            bc_ab_ratio = bc / ab if ab != 0 else 0
            cd_bc_ratio = cd / bc if bc != 0 else 0
            xd_xa_ratio = xd / xa if xa != 0 else 0
            
            # Verifica i criteri
            is_butterfly = (
                HarmonicPatterns.is_valid_harmonic_move(ab_xa_ratio, 0.786, tolerance) and
                (HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.382, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(bc_ab_ratio, 0.886, tolerance)) and
                (HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 1.618, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(cd_bc_ratio, 2.618, tolerance)) and
                (HarmonicPatterns.is_valid_harmonic_move(xd_xa_ratio, 1.27, tolerance) or
                 HarmonicPatterns.is_valid_harmonic_move(xd_xa_ratio, 1.618, tolerance))
            )
            
            if is_butterfly:
                result.loc[result.index[d_idx], 'bearish_butterfly'] = True
        
        return result


class ChartPatterns:
    """Pattern di grafici classici come Testa e Spalle, Doppi Massimi/Minimi, ecc."""
    
    @staticmethod
    def detect_head_and_shoulders(df: pd.DataFrame, n_bars: int = 5) -> pd.DataFrame:
        """
        Rileva pattern Testa e Spalle (regolare e invertito).
        
        Args:
            df: DataFrame con colonne OHLC
            n_bars: Numero di barre per definire uno swing
            
        Returns:
            DataFrame con pattern Testa e Spalle aggiunti
        """
        result = df.copy()
        
        # Verifica che le colonne dei pivot siano presenti
        if 'pivot_high' not in result.columns or 'pivot_low' not in result.columns:
            result = HarmonicPatterns.find_pivots(result, n_bars)
        
        # Inizializza colonne per i pattern
        result['head_and_shoulders'] = False  # Pattern ribassista
        result['inverse_head_and_shoulders'] = False  # Pattern rialzista
        
        # Ottieni indici di pivot high e low
        pivot_highs = result[result['pivot_high']].index
        pivot_lows = result[result['pivot_low']].index
        
        # Head and Shoulders (3 pivot high con quello centrale più alto)
        for i in range(len(pivot_highs) - 2):
            # Ottieni indici dei tre pivot high consecutivi
            h1_idx = result.index.get_loc(pivot_highs[i])
            h2_idx = result.index.get_loc(pivot_highs[i+1])
            h3_idx = result.index.get_loc(pivot_highs[i+2])
            
            # Verifica che h2 (la testa) sia più alto di h1 e h3 (le spalle)
            h1_price = result['high'].iloc[h1_idx]
            h2_price = result['high'].iloc[h2_idx]
            h3_price = result['high'].iloc[h3_idx]
            
            if h2_price > h1_price and h2_price > h3_price:
                # Verifica che le due spalle abbiano altezze simili (±10%)
                shoulder_height_diff = abs(h1_price - h3_price) / h1_price
                if shoulder_height_diff <= 0.1:
                    # Cerca i colli (neckline) - pivot low tra le spalle
                    neck1_candidates = [pl for pl in pivot_lows 
                                       if h1_idx < result.index.get_loc(pl) < h2_idx]
                    neck2_candidates = [pl for pl in pivot_lows 
                                       if h2_idx < result.index.get_loc(pl) < h3_idx]
                    
                    if neck1_candidates and neck2_candidates:
                        neck1_idx = result.index.get_loc(neck1_candidates[0])
                        neck2_idx = result.index.get_loc(neck2_candidates[0])
                        
                        neck1_price = result['low'].iloc[neck1_idx]
                        neck2_price = result['low'].iloc[neck2_idx]
                        
                        # Verifica che i colli abbiano altezze simili (±10%)
                        neck_height_diff = abs(neck1_price - neck2_price) / neck1_price
                        if neck_height_diff <= 0.1:
                            # Cerca conferma di rottura della neckline dopo la formazione
                            if h3_idx + 1 < len(result) and result['low'].iloc[h3_idx+1] < min(neck1_price, neck2_price):
                                result.loc[result.index[h3_idx], 'head_and_shoulders'] = True
        
        # Inverse Head and Shoulders (3 pivot low con quello centrale più basso)
        for i in range(len(pivot_lows) - 2):
            # Ottieni indici dei tre pivot low consecutivi
            l1_idx = result.index.get_loc(pivot_lows[i])
            l2_idx = result.index.get_loc(pivot_lows[i+1])
            l3_idx = result.index.get_loc(pivot_lows[i+2])
            
            # Verifica che l2 (la testa) sia più basso di l1 e l3 (le spalle)
            l1_price = result['low'].iloc[l1_idx]
            l2_price = result['low'].iloc[l2_idx]
            l3_price = result['low'].iloc[l3_idx]
            
            if l2_price < l1_price and l2_price < l3_price:
                # Verifica che le due spalle abbiano altezze simili (±10%)
                shoulder_height_diff = abs(l1_price - l3_price) / l1_price
                if shoulder_height_diff <= 0.1:
                    # Cerca i colli (neckline) - pivot high tra le spalle
                    neck1_candidates = [ph for ph in pivot_highs 
                                       if l1_idx < result.index.get_loc(ph) < l2_idx]
                    neck2_candidates = [ph for ph in pivot_highs 
                                       if l2_idx < result.index.get_loc(ph) < l3_idx]
                    
                    if neck1_candidates and neck2_candidates:
                        neck1_idx = result.index.get_loc(neck1_candidates[0])
                        neck2_idx = result.index.get_loc(neck2_candidates[0])
                        
                        neck1_price = result['high'].iloc[neck1_idx]
                        neck2_price = result['high'].iloc[neck2_idx]
                        
                        # Verifica che i colli abbiano altezze simili (±10%)
                        neck_height_diff = abs(neck1_price - neck2_price) / neck1_price
                        if neck_height_diff <= 0.1:
                            # Cerca conferma di rottura della neckline dopo la formazione
                            if l3_idx + 1 < len(result) and result['high'].iloc[l3_idx+1] > max(neck1_price, neck2_price):
                                result.loc[result.index[l3_idx], 'inverse_head_and_shoulders'] = True
        
        return result
    
    @staticmethod
    def detect_double_top_bottom(df: pd.DataFrame, n_bars: int = 5, 
                               price_threshold: float = 0.03) -> pd.DataFrame:
        """
        Rileva pattern Doppio Massimo e Doppio Minimo.
        
        Args:
            df: DataFrame con colonne OHLC
            n_bars: Numero di barre per definire uno swing
            price_threshold: Soglia di prezzo per considerare i massimi/minimi uguali
            
        Returns:
            DataFrame con pattern Doppio Massimo/Minimo aggiunti
        """
        result = df.copy()
        
        # Verifica che le colonne dei pivot siano presenti
        if 'pivot_high' not in result.columns or 'pivot_low' not in result.columns:
            result = HarmonicPatterns.find_pivots(result, n_bars)
        
        # Inizializza colonne per i pattern
        result['double_top'] = False
        result['double_bottom'] = False
        
        # Ottieni indici di pivot high e low
        pivot_highs = result[result['pivot_high']].index
        pivot_lows = result[result['pivot_low']].index
        
        # Double Top (due pivot high con prezzo simile)
        for i in range(len(pivot_highs) - 1):
            h1_idx = result.index.get_loc(pivot_highs[i])
            h2_idx = result.index.get_loc(pivot_highs[i+1])
            
            # Verifica che ci sia una distanza minima tra i due picchi
            if h2_idx - h1_idx >= n_bars * 2:
                h1_price = result['high'].iloc[h1_idx]
                h2_price = result['high'].iloc[h2_idx]
                
                # Verifica che i due massimi abbiano prezzi simili
                price_diff = abs(h1_price - h2_price) / h1_price
                if price_diff <= price_threshold:
                    # Cerca il minimo tra i due picchi
                    between_idx = slice(h1_idx + 1, h2_idx)
                    if len(result.iloc[between_idx]) > 0:
                        between_low = result.iloc[between_idx]['low'].min()
                        between_low_idx = result.iloc[between_idx]['low'].idxmin()
                        
                        # Verifica la rottura dopo il secondo picco
                        if h2_idx + 1 < len(result) and result['low'].iloc[h2_idx + 1] < between_low:
                            result.loc[result.index[h2_idx], 'double_top'] = True
        
        # Double Bottom (due pivot low con prezzo simile)
        for i in range(len(pivot_lows) - 1):
            l1_idx = result.index.get_loc(pivot_lows[i])
            l2_idx = result.index.get_loc(pivot_lows[i+1])
            
            # Verifica che ci sia una distanza minima tra i due minimi
            if l2_idx - l1_idx >= n_bars * 2:
                l1_price = result['low'].iloc[l1_idx]
                l2_price = result['low'].iloc[l2_idx]
                
                # Verifica che i due minimi abbiano prezzi simili
                price_diff = abs(l1_price - l2_price) / l1_price
                if price_diff <= price_threshold:
                    # Cerca il massimo tra i due minimi
                    between_idx = slice(l1_idx + 1, l2_idx)
                    if len(result.iloc[between_idx]) > 0:
                        between_high = result.iloc[between_idx]['high'].max()
                        between_high_idx = result.iloc[between_idx]['high'].idxmax()
                        
                        # Verifica la rottura dopo il secondo minimo
                        if l2_idx + 1 < len(result) and result['high'].iloc[l2_idx + 1] > between_high:
                            result.loc[result.index[l2_idx], 'double_bottom'] = True
        
        return result
    
    @staticmethod
    def detect_triangles(df: pd.DataFrame, min_points: int = 5) -> pd.DataFrame:
        """
        Rileva pattern di triangoli (simmetrici, ascendenti, discendenti).
        
        Args:
            df: DataFrame con colonne OHLC
            min_points: Numero minimo di punti di contatto per identificare un triangolo
            
        Returns:
            DataFrame con pattern di triangoli aggiunti
        """
        result = df.copy()
        
        # Verifica che le colonne dei pivot siano presenti
        if 'pivot_high' not in result.columns or 'pivot_low' not in result.columns:
            result = HarmonicPatterns.find_pivots(result)
        
        # Inizializza colonne per i pattern
        result['symmetric_triangle'] = False
        result['ascending_triangle'] = False
        result['descending_triangle'] = False
        
        # Ottieni indici di pivot high e low
        pivot_highs = result[result['pivot_high']].index
        pivot_lows = result[result['pivot_low']].index
        
        # Cerca almeno 3 pivot highs e 3 pivot lows per formare un triangolo
        if len(pivot_highs) < 3 or len(pivot_lows) < 3:
            return result
        
        # Calcola la pendenza della linea di trend superiore
        top_points = [(result.index.get_loc(idx), result['high'].loc[idx]) 
                     for idx in pivot_highs]
        
        # Calcola la pendenza della linea di trend inferiore
        bottom_points = [(result.index.get_loc(idx), result['low'].loc[idx]) 
                        for idx in pivot_lows]
        
        # Analizza possibili triangoli per ogni finestra di tempo
        for start_idx in range(len(result) - min_points * 3):
            end_idx = start_idx + min_points * 3
            
            # Seleziona i punti in questa finestra
            window_top_points = [(x, y) for x, y in top_points 
                               if start_idx <= x < end_idx]
            window_bottom_points = [(x, y) for x, y in bottom_points 
                                  if start_idx <= x < end_idx]
            
            if len(window_top_points) < 3 or len(window_bottom_points) < 3:
                continue
            
            # Calcola linee di tendenza usando regressione lineare
            top_x = [x for x, y in window_top_points]
            top_y = [y for x, y in window_top_points]
            bottom_x = [x for x, y in window_bottom_points]
            bottom_y = [y for x, y in window_bottom_points]
            
            if len(top_x) < 2 or len(bottom_x) < 2:
                continue
            
            # Calcola coefficienti di regressione lineare (pendenza, intercetta)
            top_slope, top_intercept = np.polyfit(top_x, top_y, 1)
            bottom_slope, bottom_intercept = np.polyfit(bottom_x, bottom_y, 1)
            
            # Classifica il triangolo in base alle pendenze
            if top_slope < -0.0001 and bottom_slope > 0.0001:
                # Triangolo simmetrico: linea superiore discendente, linea inferiore ascendente
                result.loc[result.index[end_idx-1], 'symmetric_triangle'] = True
            elif abs(top_slope) < 0.0001 and bottom_slope > 0.0001:
                # Triangolo ascendente: linea superiore orizzontale, linea inferiore ascendente
                result.loc[result.index[end_idx-1], 'ascending_triangle'] = True
            elif top_slope < -0.0001 and abs(bottom_slope) < 0.0001:
                # Triangolo discendente: linea superiore discendente, linea inferiore orizzontale
                result.loc[result.index[end_idx-1], 'descending_triangle'] = True
        
        return result


class PricePatterns:
    """Pattern di prezzi come Breccia, Flag, Cup and Handle, ecc."""
    
    @staticmethod
    def detect_flags_pennants(df: pd.DataFrame, min_pole_len: int = 5, 
                            max_flag_len: int = 15) -> pd.DataFrame:
        """
        Rileva pattern Flag e Pennant.
        
        Args:
            df: DataFrame con colonne OHLC
            min_pole_len: Lunghezza minima dell'asta della bandiera
            max_flag_len: Lunghezza massima della bandiera/pennant
            
        Returns:
            DataFrame con pattern Flag/Pennant aggiunti
        """
        result = df.copy()
        
        # Inizializza colonne per i pattern
        result['bull_flag'] = False
        result['bear_flag'] = False
        result['bull_pennant'] = False
        result['bear_pennant'] = False
        
        # Calcola i rendimenti per identificare forti movimenti (aste)
        result['returns'] = result['close'].pct_change() * 100
        
        # Identifica possibili aste per bandiere rialziste (strong upward move)
        for i in range(min_pole_len, len(result) - max_flag_len):
            # Controlla se c'è un forte movimento al rialzo per l'asta
            pole_returns = result['returns'].iloc[i-min_pole_len:i].sum()
            if pole_returns >= 5.0:  # Movimento al rialzo di almeno 5%
                # Cerca una flag o pennant dopo l'asta
                for flag_len in range(3, max_flag_len + 1):
                    if i + flag_len >= len(result):
                        break
                    
                    flag_section = result.iloc[i:i+flag_len]
                    
                    # Calcola l'intervallo della flag
                    flag_high = flag_section['high'].max()
                    flag_low = flag_section['low'].min()
                    flag_range = flag_high - flag_low
                    
                    # Calcola le linee di tendenza per la flag
                    flag_highs = flag_section['high'].values
                    flag_lows = flag_section['low'].values
                    flag_indices = np.array(range(flag_len))
                    
                    # Calcola pendenze delle linee di tendenza (se possibile)
                    if flag_len >= 3:
                        high_slope, _ = np.polyfit(flag_indices, flag_highs, 1)
                        low_slope, _ = np.polyfit(flag_indices, flag_lows, 1)
                        
                        # Flag: canale discendente
                        if -0.001 > high_slope and -0.001 > low_slope:
                            # Verifica se c'è una rottura rialzista alla fine della flag
                            if (i + flag_len + 1 < len(result) and 
                                result['close'].iloc[i + flag_len] > flag_high):
                                result.loc[result.index[i + flag_len], 'bull_flag'] = True
                                break
                        
                        # Pennant: linee convergenti
                        elif -0.001 > high_slope and low_slope > 0.001:
                            # Verifica se c'è una rottura rialzista alla fine del pennant
                            if (i + flag_len + 1 < len(result) and 
                                result['close'].iloc[i + flag_len] > flag_high):
                                result.loc[result.index[i + flag_len], 'bull_pennant'] = True
                                break
        
        # Identifica possibili aste per bandiere ribassiste (strong downward move)
        for i in range(min_pole_len, len(result) - max_flag_len):
            # Controlla se c'è un forte movimento al ribasso per l'asta
            pole_returns = result['returns'].iloc[i-min_pole_len:i].sum()
            if pole_returns <= -5.0:  # Movimento al ribasso di almeno 5%
                # Cerca una flag o pennant dopo l'asta
                for flag_len in range(3, max_flag_len + 1):
                    if i + flag_len >= len(result):
                        break
                    
                    flag_section = result.iloc[i:i+flag_len]
                    
                    # Calcola l'intervallo della flag
                    flag_high = flag_section['high'].max()
                    flag_low = flag_section['low'].min()
                    flag_range = flag_high - flag_low
                    
                    # Calcola le linee di tendenza per la flag
                    flag_highs = flag_section['high'].values
                    flag_lows = flag_section['low'].values
                    flag_indices = np.array(range(flag_len))
                    
                    # Calcola pendenze delle linee di tendenza (se possibile)
                    if flag_len >= 3:
                        high_slope, _ = np.polyfit(flag_indices, flag_highs, 1)
                        low_slope, _ = np.polyfit(flag_indices, flag_lows, 1)
                        
                        # Flag: canale ascendente
                        if high_slope > 0.001 and low_slope > 0.001:
                            # Verifica se c'è una rottura ribassista alla fine della flag
                            if (i + flag_len + 1 < len(result) and 
                                result['close'].iloc[i + flag_len] < flag_low):
                                result.loc[result.index[i + flag_len], 'bear_flag'] = True
                                break
                        
                        # Pennant: linee convergenti
                        elif high_slope < -0.001 and low_slope > 0.001:
                            # Verifica se c'è una rottura ribassista alla fine del pennant
                            if (i + flag_len + 1 < len(result) and 
                                result['close'].iloc[i + flag_len] < flag_low):
                                result.loc[result.index[i + flag_len], 'bear_pennant'] = True
                                break
        
        # Rimuovi colonna temporanea
        result.drop(['returns'], axis=1, inplace=True)
        
        return result
    
    @staticmethod
    def detect_cup_and_handle(df: pd.DataFrame, max_cup_len: int = 60, 
                            max_handle_len: int = 15) -> pd.DataFrame:
        """
        Rileva pattern Cup and Handle.
        
        Args:
            df: DataFrame con colonne OHLC
            max_cup_len: Lunghezza massima della coppa
            max_handle_len: Lunghezza massima del manico
            
        Returns:
            DataFrame con pattern Cup and Handle aggiunti
        """
        result = df.copy()
        
        # Inizializza colonna per il pattern
        result['cup_and_handle'] = False
        
        # Trova pivot highs per identificare potenziali bordi della coppa
        if 'pivot_high' not in result.columns:
            result = HarmonicPatterns.find_pivots(result)
        
        pivot_highs = result[result['pivot_high']].index
        
        # Esamina ogni pivot high come potenziale bordo sinistro della coppa
        for i in range(len(pivot_highs) - 1):
            left_idx = result.index.get_loc(pivot_highs[i])
            
            # Cerca un altro pivot high che potrebbe formare il bordo destro della coppa
            for j in range(i + 1, len(pivot_highs)):
                right_idx = result.index.get_loc(pivot_highs[j])
                
                # Verifica che la distanza tra i bordi rientri nel limite
                cup_len = right_idx - left_idx
                if cup_len > max_cup_len or cup_len < 10:  # Almeno 10 barre per una coppa
                    continue
                
                # Verifica che i due bordi abbiano prezzi simili (±5%)
                left_price = result['high'].iloc[left_idx]
                right_price = result['high'].iloc[right_idx]
                
                price_diff = abs(left_price - right_price) / left_price
                if price_diff > 0.05:
                    continue
                
                # Cerca il minimo della coppa
                cup_section = result.iloc[left_idx:right_idx + 1]
                cup_low = cup_section['low'].min()
                cup_low_idx = cup_section['low'].idxmin()
                
                # Verifica che il minimo sia circa a metà della coppa e sufficientemente profondo
                cup_low_pos = result.index.get_loc(cup_low_idx) - left_idx
                if not (0.25 * cup_len <= cup_low_pos <= 0.75 * cup_len):
                    continue
                
                # Verifica che la profondità della coppa sia adeguata (>10% dal bordo)
                cup_depth = (left_price - cup_low) / left_price
                if cup_depth < 0.1:
                    continue
                
                # Cerca un manico dopo il bordo destro della coppa
                if right_idx + max_handle_len >= len(result):
                    continue
                
                handle_section = result.iloc[right_idx:right_idx + max_handle_len + 1]
                
                # Il manico dovrebbe essere un piccolo pullback (più piccolo della coppa)
                handle_low = handle_section['low'].min()
                handle_low_idx = handle_section['low'].idxmin()
                
                # Verifica che il manico non sia troppo profondo
                handle_depth = (right_price - handle_low) / right_price
                if handle_depth > cup_depth * 0.5 or handle_depth < 0.03:
                    continue
                
                # Verifica che ci sia una rottura rialzista dopo il manico
                handle_end_idx = result.index.get_loc(handle_low_idx) + 5
                if handle_end_idx >= len(result):
                    continue
                
                if result['close'].iloc[handle_end_idx] > right_price:
                    result.loc[result.index[handle_end_idx], 'cup_and_handle'] = True
        
        return result


# Factory function per accedere ai pattern recognizers
def get_pattern_recognizers():
    """
    Ottiene un'istanza di ogni classe di pattern recognizer.
    
    Returns:
        Dizionario con istanze di classi di pattern recognizer
    """
    return {
        'candle': CandlePatterns(),
        'harmonic': HarmonicPatterns(),
        'chart': ChartPatterns(),
        'price': PricePatterns()
    }