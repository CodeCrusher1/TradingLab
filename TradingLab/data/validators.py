# Validazione dei dati

"""
Funzioni e classi per la validazione dei dati nel progetto TradingLab.
Questo modulo fornisce strumenti per verificare la correttezza, l'integrità e la qualità dei dati finanziari.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import re
from datetime import datetime, timedelta

# Importazione dal modulo config
from ..config import SYMBOLS, TIMEFRAMES, get_symbol, get_timeframe
from ..utils import app_logger, data_logger, ValidationError, InvalidSymbolError, InvalidTimeframeError


class DataValidator:
    """
    Classe base per i validatori di dati.
    Fornisce metodi comuni per verificare la validità dei dati.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str], 
                          index_col: Optional[str] = None) -> pd.DataFrame:
        """
        Verifica che un DataFrame abbia le colonne richieste.
        
        Args:
            df: DataFrame da validare
            required_columns: Lista di colonne che devono essere presenti
            index_col: Colonna da usare come indice (opzionale)
            
        Returns:
            Il DataFrame validato
            
        Raises:
            ValidationError: Se il DataFrame è vuoto o mancano colonne richieste
        """
        if df is None or df.empty:
            raise ValidationError("Il DataFrame è vuoto o None")
        
        # Verifica le colonne richieste
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValidationError(f"Colonne mancanti nel DataFrame: {missing_columns}")
        
        # Imposta la colonna indice se specificata
        if index_col and index_col in df.columns and not df.index.name == index_col:
            df = df.set_index(index_col)
        
        return df
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """
        Verifica che un simbolo sia valido.
        
        Args:
            symbol: Nome del simbolo da validare
            
        Returns:
            Il simbolo validato
            
        Raises:
            InvalidSymbolError: Se il simbolo non è valido
        """
        symbol_obj = get_symbol(symbol)
        if symbol_obj is None:
            raise InvalidSymbolError(f"Simbolo non valido: {symbol}")
        return symbol
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """
        Verifica che un timeframe sia valido.
        
        Args:
            timeframe: Nome del timeframe da validare
            
        Returns:
            Il timeframe validato
            
        Raises:
            InvalidTimeframeError: Se il timeframe non è valido
        """
        timeframe_obj = get_timeframe(timeframe)
        if timeframe_obj is None:
            raise InvalidTimeframeError(f"Timeframe non valido: {timeframe}")
        return timeframe
    
    @staticmethod
    def validate_date_range(start_date: Union[str, datetime], 
                           end_date: Union[str, datetime, None] = None) -> Tuple[datetime, datetime]:
        """
        Verifica e converte un intervallo di date.
        
        Args:
            start_date: Data di inizio
            end_date: Data di fine (opzionale, default=oggi)
            
        Returns:
            Tuple di (start_date, end_date) come oggetti datetime
            
        Raises:
            ValidationError: Se le date non sono valide
        """
        # Converti stringhe in datetime
        if isinstance(start_date, str):
            try:
                start_date = pd.to_datetime(start_date)
            except Exception as e:
                raise ValidationError(f"Data di inizio non valida: {start_date}", details={"error": str(e)})
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            try:
                end_date = pd.to_datetime(end_date)
            except Exception as e:
                raise ValidationError(f"Data di fine non valida: {end_date}", details={"error": str(e)})
        
        # Verifica che start_date sia prima di end_date
        if start_date > end_date:
            raise ValidationError("La data di inizio deve essere precedente alla data di fine",
                                 details={"start_date": start_date, "end_date": end_date})
        
        return start_date, end_date


class MarketDataValidator(DataValidator):
    """
    Validatore specifico per dati di mercato (OHLCV).
    """
    
    # Colonne richieste per dati OHLCV
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    @classmethod
    def validate_ohlcv(cls, df: pd.DataFrame, set_index: bool = True) -> pd.DataFrame:
        """
        Valida un DataFrame OHLCV.
        
        Args:
            df: DataFrame OHLCV da validare
            set_index: Se impostare 'timestamp' come indice
            
        Returns:
            DataFrame OHLCV validato
        """
        df = cls.validate_dataframe(df, cls.REQUIRED_COLUMNS, 
                                   index_col='timestamp' if set_index else None)
        
        # Verifica che 'timestamp' sia di tipo datetime
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    raise ValidationError("Impossibile convertire 'timestamp' in datetime", 
                                         details={"error": str(e)})
        
        # Verifica che i valori numerici siano effettivamente numeri
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception as e:
                    raise ValidationError(f"Impossibile convertire '{col}' in numerico", 
                                         details={"error": str(e)})
        
        return df
    
    @classmethod
    def validate_ohlcv_consistency(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Verifica la consistenza logica dei dati OHLCV.
        
        Args:
            df: DataFrame OHLCV da validare
            
        Returns:
            DataFrame pulito e validato
            
        Raises:
            ValidationError: Se i dati contengono incongruenze gravi
        """
        df = cls.validate_ohlcv(df, set_index=False)
        
        # Verifica che high >= low
        invalid_hl = df[df['high'] < df['low']]
        if not invalid_hl.empty:
            data_logger.warning(f"Trovate {len(invalid_hl)} righe con high < low")
            # Correggi scambiando high e low
            idx = invalid_hl.index
            df.loc[idx, ['high', 'low']] = df.loc[idx, ['low', 'high']].values
        
        # Verifica che high >= open e high >= close
        invalid_ho = df[df['high'] < df['open']]
        if not invalid_ho.empty:
            data_logger.warning(f"Trovate {len(invalid_ho)} righe con high < open")
            # Correggi impostando high = max(high, open)
            df.loc[invalid_ho.index, 'high'] = np.maximum(
                df.loc[invalid_ho.index, 'high'], 
                df.loc[invalid_ho.index, 'open']
            )
        
        invalid_hc = df[df['high'] < df['close']]
        if not invalid_hc.empty:
            data_logger.warning(f"Trovate {len(invalid_hc)} righe con high < close")
            # Correggi impostando high = max(high, close)
            df.loc[invalid_hc.index, 'high'] = np.maximum(
                df.loc[invalid_hc.index, 'high'], 
                df.loc[invalid_hc.index, 'close']
            )
        
        # Verifica che low <= open e low <= close
        invalid_lo = df[df['low'] > df['open']]
        if not invalid_lo.empty:
            data_logger.warning(f"Trovate {len(invalid_lo)} righe con low > open")
            # Correggi impostando low = min(low, open)
            df.loc[invalid_lo.index, 'low'] = np.minimum(
                df.loc[invalid_lo.index, 'low'], 
                df.loc[invalid_lo.index, 'open']
            )
        
        invalid_lc = df[df['low'] > df['close']]
        if not invalid_lc.empty:
            data_logger.warning(f"Trovate {len(invalid_lc)} righe con low > close")
            # Correggi impostando low = min(low, close)
            df.loc[invalid_lc.index, 'low'] = np.minimum(
                df.loc[invalid_lc.index, 'low'], 
                df.loc[invalid_lc.index, 'close']
            )
        
        # Verifica valori negativi
        for col in ['open', 'high', 'low', 'close']:
            negative_vals = df[df[col] < 0]
            if not negative_vals.empty:
                data_logger.warning(f"Trovate {len(negative_vals)} righe con {col} < 0")
                # Per i prezzi, i valori negativi sono certamente un errore
                # Sostituisci con valore assoluto
                df.loc[negative_vals.index, col] = np.abs(df.loc[negative_vals.index, col])
        
        # Per il volume, i valori negativi potrebbero indicare vendite, ma per semplicità
        # usiamo sempre valori assoluti
        negative_volume = df[df['volume'] < 0]
        if not negative_volume.empty:
            data_logger.warning(f"Trovate {len(negative_volume)} righe con volume < 0")
            df.loc[negative_volume.index, 'volume'] = np.abs(df.loc[negative_volume.index, 'volume'])
        
        # Verifica timestamp duplicati
        duplicates = df[df.duplicated(subset=['timestamp'], keep=False)]
        if not duplicates.empty:
            data_logger.warning(f"Trovati {len(duplicates)} timestamp duplicati")
            # Mantieni solo la prima occorrenza di ogni timestamp duplicato
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Ordina per timestamp
        df = df.sort_values('timestamp')
        
        return df
    
    @classmethod
    def detect_outliers(cls, df: pd.DataFrame, columns: List[str] = None, 
                       z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Rileva e gestisce outlier nei dati.
        
        Args:
            df: DataFrame da analizzare
            columns: Colonne da controllare (default: tutte le colonne numeriche)
            z_threshold: Soglia per z-score (default: 3.0)
            
        Returns:
            DataFrame con maschera booleana degli outlier
        """
        if columns is None:
            # Usa tutte le colonne numeriche tranne timestamp
            columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col != 'timestamp']
        
        # DataFrame per memorizzare i risultati
        outliers_df = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Calcola z-score
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:  # Evita divisione per zero
                outliers_df[f'{col}_outlier'] = False
                continue
            
            z_scores = np.abs((df[col] - mean) / std)
            outliers_df[f'{col}_outlier'] = z_scores > z_threshold
            
            # Log del numero di outlier
            n_outliers = outliers_df[f'{col}_outlier'].sum()
            if n_outliers > 0:
                data_logger.info(f"Rilevati {n_outliers} outlier nella colonna '{col}'")
        
        return outliers_df
    
    @classmethod
    def detect_gaps(cls, df: pd.DataFrame, max_gap: Optional[timedelta] = None) -> pd.DataFrame:
        """
        Rileva gap nei dati timestamp.
        
        Args:
            df: DataFrame da analizzare
            max_gap: Gap massimo consentito (default: basato sul timeframe)
            
        Returns:
            DataFrame con informazioni sui gap
        """
        # Verifica che timestamp sia presente
        if 'timestamp' not in df.columns:
            raise ValidationError("Colonna 'timestamp' non trovata nel DataFrame")
        
        # Ordina per timestamp
        df = df.sort_values('timestamp')
        
        # Calcola la differenza tra timestamp consecutivi
        time_diffs = df['timestamp'].diff()
        
        # Se max_gap non è specificato, calcola in base ai dati
        if max_gap is None:
            # Euristica: usa il 95° percentile delle differenze moltiplicato per 3
            normal_diff = np.percentile(time_diffs.dropna(), 95)
            max_gap = normal_diff * 3
        
        # Identifica i gap
        gaps = df[time_diffs > max_gap].copy()
        
        if not gaps.empty:
            # Aggiungi informazioni sul gap
            gaps['gap_size'] = time_diffs[gaps.index]
            gaps['prev_timestamp'] = df['timestamp'].shift(1)[gaps.index]
            
            data_logger.info(f"Rilevati {len(gaps)} gap nei dati")
        
        return gaps
    
    @classmethod
    def validate_processed_data(cls, df: pd.DataFrame, 
                               indicator_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Valida dati elaborati con indicatori tecnici.
        
        Args:
            df: DataFrame con dati elaborati
            indicator_columns: Colonne degli indicatori da validare
            
        Returns:
            DataFrame validato
        """
        # Prima valida la struttura OHLCV di base
        df = cls.validate_ohlcv(df, set_index=False)
        
        if indicator_columns:
            # Verifica che le colonne degli indicatori siano presenti
            missing_indicators = [col for col in indicator_columns if col not in df.columns]
            if missing_indicators:
                data_logger.warning(f"Indicatori mancanti: {missing_indicators}")
            
            # Verifica che gli indicatori siano tipi numerici
            for col in indicator_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except Exception:
                        data_logger.warning(f"Impossibile convertire '{col}' in numerico")
        
        return df


class DataFrameValidator:
    """
    Validatore generico per DataFrame.
    """
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Verifica che un DataFrame abbia le colonne richieste.
        
        Args:
            df: DataFrame da validare
            required_columns: Colonne richieste
            
        Returns:
            True se tutte le colonne richieste sono presenti
        """
        return all(col in df.columns for col in required_columns)
    
    @staticmethod
    def validate_types(df: pd.DataFrame, column_types: Dict[str, type]) -> bool:
        """
        Verifica che le colonne abbiano i tipi specificati.
        
        Args:
            df: DataFrame da validare
            column_types: Dizionario {nome_colonna: tipo}
            
        Returns:
            True se tutte le colonne hanno i tipi corretti
        """
        for col, dtype in column_types.items():
            if col not in df.columns:
                return False
            
            if dtype == int:
                if not pd.api.types.is_integer_dtype(df[col]):
                    return False
            elif dtype == float:
                if not pd.api.types.is_float_dtype(df[col]):
                    return False
            elif dtype == str:
                if not pd.api.types.is_string_dtype(df[col]):
                    return False
            elif dtype == bool:
                if not pd.api.types.is_bool_dtype(df[col]):
                    return False
            elif dtype == datetime:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    return False
        
        return True
    
    @staticmethod
    def validate_range(df: pd.DataFrame, column: str, min_val: Optional[Any] = None, 
                      max_val: Optional[Any] = None) -> bool:
        """
        Verifica che i valori in una colonna siano entro un intervallo.
        
        Args:
            df: DataFrame da validare
            column: Nome della colonna
            min_val: Valore minimo ammesso (opzionale)
            max_val: Valore massimo ammesso (opzionale)
            
        Returns:
            True se tutti i valori sono nell'intervallo
        """
        if column not in df.columns:
            return False
        
        if min_val is not None and (df[column] < min_val).any():
            return False
        
        if max_val is not None and (df[column] > max_val).any():
            return False
        
        return True
    
    @staticmethod
    def validate_no_null(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
        """
        Verifica che non ci siano valori null.
        
        Args:
            df: DataFrame da validare
            columns: Colonne da verificare (opzionale, default: tutte)
            
        Returns:
            True se non ci sono valori null
        """
        if columns is None:
            return not df.isnull().any().any()
        
        return not df[columns].isnull().any().any()
    
    @staticmethod
    def validate_uniqueness(df: pd.DataFrame, columns: Union[str, List[str]]) -> bool:
        """
        Verifica che i valori nelle colonne siano unici.
        
        Args:
            df: DataFrame da validare
            columns: Colonna o colonne da verificare
            
        Returns:
            True se i valori sono unici
        """
        if isinstance(columns, str):
            columns = [columns]
        
        return not df.duplicated(subset=columns).any()


class YahooFinanceValidator:
    """
    Validatore specifico per dati provenienti da Yahoo Finance.
    """
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """
        Verifica che un ticker Yahoo Finance sia in formato valido.
        
        Args:
            ticker: Ticker da validare
            
        Returns:
            True se il formato è valido
        """
        # Verifica formato generale dei ticker Yahoo Finance
        valid_patterns = [
            r'^[A-Z0-9\.-]{1,10}$',                  # Ticker US standard
            r'^[A-Z0-9\.-]{1,10}\.[A-Z]{1,2}$',       # Ticker con suffisso di borsa
            r'^[A-Z0-9\.-]{1,10}=F$',                # Futures
            r'^[A-Z0-9\.-]{1,10}-[A-Z]{3}$',         # Cryptocurrency
            r'^[A-Z]{3}[A-Z]{3}=X$'                  # Forex
        ]
        
        return any(re.match(pattern, ticker) for pattern in valid_patterns)
    
    @staticmethod
    def convert_yahoo_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte i dati Yahoo Finance nel formato OHLCV standard.
        
        Args:
            df: DataFrame di Yahoo Finance
            
        Returns:
            DataFrame nel formato OHLCV standard
        """
        # Mappatura delle colonne Yahoo Finance
        column_mapping = {
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Rinomina le colonne
        renamed_df = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in renamed_df.columns:
                renamed_df = renamed_df.rename(columns={old_col: new_col})
        
        # Verifica che le colonne necessarie siano presenti
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in renamed_df.columns]
        
        if missing_cols:
            data_logger.warning(f"Colonne mancanti nei dati Yahoo Finance: {missing_cols}")
            # Aggiungi colonne mancanti con valori NaN
            for col in missing_cols:
                renamed_df[col] = np.nan
        
        return renamed_df


# Funzioni di utilità per la validazione
def is_valid_symbol(symbol: str) -> bool:
    """
    Verifica se un simbolo è valido.
    
    Args:
        symbol: Nome del simbolo
        
    Returns:
        True se il simbolo è valido
    """
    return get_symbol(symbol) is not None


def is_valid_timeframe(timeframe: str) -> bool:
    """
    Verifica se un timeframe è valido.
    
    Args:
        timeframe: Nome del timeframe
        
    Returns:
        True se il timeframe è valido
    """
    return get_timeframe(timeframe) is not None


def validate_date_str(date_str: str) -> bool:
    """
    Verifica se una stringa rappresenta una data valida.
    
    Args:
        date_str: Stringa data da validare
        
    Returns:
        True se la data è valida
    """
    try:
        pd.to_datetime(date_str)
        return True
    except:
        return False