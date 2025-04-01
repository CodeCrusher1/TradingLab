# Gestione storage basato su file

"""
Modulo per la gestione dello storage basato su file per il progetto TradingLab.
Fornisce funzionalità per leggere e scrivere dati da/a file in vari formati.
"""
import os
import csv
import json
import pickle
import gzip
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta

# Importazioni dal modulo config
from ..config import (
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    get_data_path, SYMBOLS, TIMEFRAMES
)

# Importazioni dal modulo utils
from ..utils import (
    app_logger, data_logger, FileError, 
    ensure_directory, safe_file_write, safe_file_read,
    save_json, load_json, save_pickle, load_pickle,
    save_csv, load_csv
)

# Importazioni dal modulo validators
from .validators import MarketDataValidator


class MarketDataStorage:
    """
    Classe per la gestione dello storage di dati di mercato su file.
    """
    
    def __init__(self):
        """Inizializza lo storage di dati di mercato."""
        # Assicurati che le directory esistano
        ensure_directory(RAW_DATA_DIR)
        ensure_directory(PROCESSED_DATA_DIR)
        self.validator = MarketDataValidator()
    
    def save_raw_data(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                     compress: bool = False) -> str:
        """
        Salva dati grezzi su file.
        
        Args:
            df: DataFrame con dati OHLCV
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            compress: Se comprimere i dati
            
        Returns:
            Percorso del file salvato
            
        Raises:
            FileError: Se il salvataggio fallisce
        """
        try:
            # Valida i dati
            df = self.validator.validate_ohlcv_consistency(df)
            
            # Crea il percorso del file
            file_path = get_data_path(symbol, timeframe, processed=False)
            
            # Assicurati che la directory esista
            ensure_directory(file_path.parent)
            
            # Salva in CSV
            if compress:
                # Comprimi con gzip
                with gzip.open(str(file_path) + '.gz', 'wt', encoding='utf-8') as f:
                    df.to_csv(f, index=False)
                file_path = Path(str(file_path) + '.gz')
            else:
                df.to_csv(file_path, index=False)
            
            data_logger.info(f"Dati grezzi salvati in {file_path}")
            return str(file_path)
        
        except Exception as e:
            error_msg = f"Errore durante il salvataggio dei dati grezzi per {symbol} con timeframe {timeframe}: {e}"
            data_logger.error(error_msg)
            raise FileError(error_msg)
    
    def load_raw_data(self, symbol: str, timeframe: str, 
                     compressed: bool = False) -> pd.DataFrame:
        """
        Carica dati grezzi da file.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            compressed: Se i dati sono compressi
            
        Returns:
            DataFrame con dati OHLCV
            
        Raises:
            FileError: Se il caricamento fallisce
        """
        try:
            # Crea il percorso del file
            file_path = get_data_path(symbol, timeframe, processed=False)
            
            if compressed:
                file_path = Path(str(file_path) + '.gz')
            
            # Verifica che il file esista
            if not file_path.exists():
                data_logger.warning(f"File non trovato: {file_path}")
                return pd.DataFrame()
            
            # Carica da CSV
            if compressed:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)
            
            # Converti timestamp in datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Valida i dati
            df = self.validator.validate_ohlcv(df)
            
            data_logger.info(f"Dati grezzi caricati da {file_path}: {len(df)} record")
            return df
        
        except Exception as e:
            error_msg = f"Errore durante il caricamento dei dati grezzi per {symbol} con timeframe {timeframe}: {e}"
            data_logger.error(error_msg)
            raise FileError(error_msg)
    
    def save_processed_data(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                           compress: bool = False) -> str:
        """
        Salva dati elaborati su file.
        
        Args:
            df: DataFrame con dati elaborati
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            compress: Se comprimere i dati
            
        Returns:
            Percorso del file salvato
            
        Raises:
            FileError: Se il salvataggio fallisce
        """
        try:
            # Valida i dati
            df = self.validator.validate_processed_data(df)
            
            # Crea il percorso del file
            file_path = get_data_path(symbol, timeframe, processed=True)
            
            # Assicurati che la directory esista
            ensure_directory(file_path.parent)
            
            # Converte booleani in int per CSV
            bool_columns = ['fair_value_gap', 'order_block_bullish', 'order_block_bearish', 'breaker_block']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(int)
            
            # Salva in CSV
            if compress:
                # Comprimi con gzip
                with gzip.open(str(file_path) + '.gz', 'wt', encoding='utf-8') as f:
                    df.to_csv(f, index=False)
                file_path = Path(str(file_path) + '.gz')
            else:
                df.to_csv(file_path, index=False)
            
            data_logger.info(f"Dati elaborati salvati in {file_path}")
            return str(file_path)
        
        except Exception as e:
            error_msg = f"Errore durante il salvataggio dei dati elaborati per {symbol} con timeframe {timeframe}: {e}"
            data_logger.error(error_msg)
            raise FileError(error_msg)
    
    def load_processed_data(self, symbol: str, timeframe: str, 
                           compressed: bool = False) -> pd.DataFrame:
        """
        Carica dati elaborati da file.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            compressed: Se i dati sono compressi
            
        Returns:
            DataFrame con dati elaborati
            
        Raises:
            FileError: Se il caricamento fallisce
        """
        try:
            # Crea il percorso del file
            file_path = get_data_path(symbol, timeframe, processed=True)
            
            if compressed:
                file_path = Path(str(file_path) + '.gz')
            
            # Verifica che il file esista
            if not file_path.exists():
                data_logger.warning(f"File non trovato: {file_path}")
                return pd.DataFrame()
            
            # Carica da CSV
            if compressed:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)
            
            # Converti timestamp in datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Converti colonne booleane
            bool_columns = ['fair_value_gap', 'order_block_bullish', 'order_block_bearish', 'breaker_block']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
            
            data_logger.info(f"Dati elaborati caricati da {file_path}: {len(df)} record")
            return df
        
        except Exception as e:
            error_msg = f"Errore durante il caricamento dei dati elaborati per {symbol} con timeframe {timeframe}: {e}"
            data_logger.error(error_msg)
            raise FileError(error_msg)
    
    def append_raw_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        Aggiunge nuovi dati grezzi a un file esistente.
        
        Args:
            df: DataFrame con nuovi dati
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            
        Returns:
            Numero di record aggiunti
            
        Raises:
            FileError: Se l'operazione fallisce
        """
        try:
            # Se il DataFrame è vuoto, non c'è nulla da fare
            if df.empty:
                return 0
            
            # Carica i dati esistenti
            existing_df = self.load_raw_data(symbol, timeframe)
            
            # Se non ci sono dati esistenti, salva i nuovi dati
            if existing_df.empty:
                self.save_raw_data(df, symbol, timeframe)
                return len(df)
            
            # Concatena i DataFrames
            combined_df = pd.concat([existing_df, df])
            
            # Rimuovi duplicati
            combined_df = combined_df.drop_duplicates(subset=['timestamp'])
            
            # Ordina per timestamp
            combined_df = combined_df.sort_values('timestamp')
            
            # Salva il risultato
            self.save_raw_data(combined_df, symbol, timeframe)
            
            # Calcola quanti record sono stati effettivamente aggiunti
            records_added = len(combined_df) - len(existing_df)
            
            data_logger.info(f"Aggiunti {records_added} nuovi record per {symbol} con timeframe {timeframe}")
            return records_added
        
        except Exception as e:
            error_msg = f"Errore durante l'aggiunta di dati per {symbol} con timeframe {timeframe}: {e}"
            data_logger.error(error_msg)
            raise FileError(error_msg)
    
    def export_data(self, df: pd.DataFrame, file_path: Union[str, Path], 
                   file_format: str = 'csv') -> str:
        """
        Esporta dati in vari formati.
        
        Args:
            df: DataFrame da esportare
            file_path: Percorso del file di output
            file_format: Formato del file ('csv', 'json', 'excel', 'pickle')
            
        Returns:
            Percorso del file esportato
            
        Raises:
            FileError: Se l'esportazione fallisce
            ValueError: Se il formato non è supportato
        """
        try:
            file_path = Path(file_path)
            
            # Assicurati che la directory esista
            ensure_directory(file_path.parent)
            
            # Esporta nel formato richiesto
            if file_format.lower() == 'csv':
                df.to_csv(file_path, index=False)
            elif file_format.lower() == 'json':
                # Converte date in stringhe per JSON
                df_copy = df.copy()
                for col in df_copy.select_dtypes(include=['datetime64']).columns:
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_copy.to_json(file_path, orient='records', date_format='iso')
            elif file_format.lower() == 'excel':
                df.to_excel(file_path, index=False)
            elif file_format.lower() == 'pickle':
                df.to_pickle(file_path)
            else:
                raise ValueError(f"Formato file non supportato: {file_format}")
            
            data_logger.info(f"Dati esportati in {file_path}")
            return str(file_path)
        
        except Exception as e:
            error_msg = f"Errore durante l'esportazione dei dati in {file_path}: {e}"
            data_logger.error(error_msg)
            raise FileError(error_msg)
    
    def get_available_data_info(self) -> pd.DataFrame:
        """
        Ottiene informazioni sui dati disponibili nei file.
        
        Returns:
            DataFrame con informazioni sui dati disponibili
        """
        try:
            # Elenco dei dati disponibili
            data_info = []
            
            # Controlla dati grezzi
            for symbol in SYMBOLS.keys():
                for timeframe in TIMEFRAMES.keys():
                    raw_path = get_data_path(symbol, timeframe, processed=False)
                    processed_path = get_data_path(symbol, timeframe, processed=True)
                    
                    # Informazioni sui dati grezzi
                    raw_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'type': 'raw',
                        'available': False,
                        'record_count': 0,
                        'start_date': None,
                        'end_date': None,
                        'file_size_mb': 0
                    }
                    
                    if raw_path.exists():
                        raw_info['available'] = True
                        raw_info['file_size_mb'] = round(raw_path.stat().st_size / (1024 * 1024), 2)
                        
                        # Carica dati per ottenere info
                        try:
                            df = pd.read_csv(raw_path)
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                raw_info['record_count'] = len(df)
                                raw_info['start_date'] = df['timestamp'].min()
                                raw_info['end_date'] = df['timestamp'].max()
                        except:
                            # Se non è possibile caricare, mantieni i valori predefiniti
                            pass
                    
                    data_info.append(raw_info)
                    
                    # Informazioni sui dati elaborati
                    processed_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'type': 'processed',
                        'available': False,
                        'record_count': 0,
                        'start_date': None,
                        'end_date': None,
                        'file_size_mb': 0
                    }
                    
                    if processed_path.exists():
                        processed_info['available'] = True
                        processed_info['file_size_mb'] = round(processed_path.stat().st_size / (1024 * 1024), 2)
                        
                        # Carica dati per ottenere info
                        try:
                            df = pd.read_csv(processed_path)
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                processed_info['record_count'] = len(df)
                                processed_info['start_date'] = df['timestamp'].min()
                                processed_info['end_date'] = df['timestamp'].max()
                        except:
                            # Se non è possibile caricare, mantieni i valori predefiniti
                            pass
                    
                    data_info.append(processed_info)
            
            # Crea DataFrame con le informazioni
            return pd.DataFrame(data_info)
        
        except Exception as e:
            data_logger.error(f"Errore durante il recupero delle informazioni sui dati disponibili: {e}")
            return pd.DataFrame()


class DataPersistenceManager:
    """
    Gestore per il salvataggio e caricamento di dati persistenti.
    Supporta diverse modalità di storage (file, database).
    """
    
    def __init__(self, db_repository=None, file_storage=None):
        """
        Inizializza il gestore.
        
        Args:
            db_repository: Repository del database (opzionale)
            file_storage: Storage basato su file (opzionale)
        """
        # Importa qui per evitare dipendenze circolari
        from .database import get_default_repository
        
        self.db_repository = db_repository or get_default_repository()
        self.file_storage = file_storage or MarketDataStorage()
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                 storage_type: str = 'both', processed: bool = False) -> bool:
        """
        Salva dati.
        
        Args:
            df: DataFrame con dati
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            storage_type: Tipo di storage ('db', 'file', 'both')
            processed: Se sono dati elaborati
            
        Returns:
            True se il salvataggio ha successo
        """
        success = True
        
        try:
            if storage_type in ['db', 'both']:
                # Salva nel database
                if processed:
                    self.db_repository.store_processed_data(df, symbol, timeframe)
                else:
                    self.db_repository.store_raw_data(df, symbol, timeframe)
            
            if storage_type in ['file', 'both']:
                # Salva su file
                if processed:
                    self.file_storage.save_processed_data(df, symbol, timeframe)
                else:
                    self.file_storage.save_raw_data(df, symbol, timeframe)
            
            return success
        
        except Exception as e:
            data_logger.error(f"Errore durante il salvataggio dei dati per {symbol} con timeframe {timeframe}: {e}")
            return False
    
    def load_data(self, symbol: str, timeframe: str, 
                 storage_type: str = 'db', processed: bool = False, 
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 limit: int = 5000) -> pd.DataFrame:
        """
        Carica dati.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            storage_type: Tipo di storage ('db', 'file')
            processed: Se sono dati elaborati
            start_date: Data di inizio (opzionale)
            end_date: Data di fine (opzionale)
            limit: Limite di record da recuperare
            
        Returns:
            DataFrame con dati
        """
        try:
            if storage_type == 'db':
                # Carica dal database
                if processed:
                    return self.db_repository.fetch_processed_data(
                        symbol, timeframe, start_date, end_date, limit=limit
                    )
                else:
                    return self.db_repository.fetch_raw_data(
                        symbol, timeframe, start_date, end_date, limit=limit
                    )
            
            elif storage_type == 'file':
                # Carica da file
                if processed:
                    df = self.file_storage.load_processed_data(symbol, timeframe)
                else:
                    df = self.file_storage.load_raw_data(symbol, timeframe)
                
                # Applica filtri dopo il caricamento
                if not df.empty:
                    if start_date:
                        df = df[df['timestamp'] >= start_date]
                    
                    if end_date:
                        df = df[df['timestamp'] <= end_date]
                    
                    # Limita numero di record
                    df = df.sort_values('timestamp').iloc[-limit:]
                
                return df
            
            else:
                raise ValueError(f"Tipo di storage non supportato: {storage_type}")
        
        except Exception as e:
            data_logger.error(f"Errore durante il caricamento dei dati per {symbol} con timeframe {timeframe}: {e}")
            return pd.DataFrame()
    
    def sync_data(self, symbol: str, timeframe: str, 
                 direction: str = 'db_to_file', processed: bool = False) -> bool:
        """
        Sincronizza dati tra database e file.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            direction: Direzione della sincronizzazione ('db_to_file', 'file_to_db')
            processed: Se sono dati elaborati
            
        Returns:
            True se la sincronizzazione ha successo
        """
        try:
            if direction == 'db_to_file':
                # Sincronizza da database a file
                df = self.load_data(symbol, timeframe, 'db', processed)
                if not df.empty:
                    if processed:
                        self.file_storage.save_processed_data(df, symbol, timeframe)
                    else:
                        self.file_storage.save_raw_data(df, symbol, timeframe)
                    
                    data_logger.info(f"Sincronizzati {len(df)} record da database a file per {symbol} con timeframe {timeframe}")
                    return True
                else:
                    data_logger.warning(f"Nessun dato da sincronizzare per {symbol} con timeframe {timeframe}")
                    return False
            
            elif direction == 'file_to_db':
                # Sincronizza da file a database
                df = self.load_data(symbol, timeframe, 'file', processed)
                if not df.empty:
                    if processed:
                        self.db_repository.store_processed_data(df, symbol, timeframe)
                    else:
                        self.db_repository.store_raw_data(df, symbol, timeframe)
                    
                    data_logger.info(f"Sincronizzati {len(df)} record da file a database per {symbol} con timeframe {timeframe}")
                    return True
                else:
                    data_logger.warning(f"Nessun dato da sincronizzare per {symbol} con timeframe {timeframe}")
                    return False
            
            else:
                raise ValueError(f"Direzione di sincronizzazione non supportata: {direction}")
        
        except Exception as e:
            data_logger.error(f"Errore durante la sincronizzazione dei dati per {symbol} con timeframe {timeframe}: {e}")
            return False
    
    def export_to_csv(self, symbol: str, timeframe: str, 
                     export_path: Optional[Union[str, Path]] = None,
                     processed: bool = False,
                     storage_type: str = 'db') -> str:
        """
        Esporta dati in CSV.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            export_path: Percorso di esportazione (opzionale)
            processed: Se sono dati elaborati
            storage_type: Tipo di storage da cui recuperare i dati
            
        Returns:
            Percorso del file esportato
        """
        try:
            # Carica i dati
            df = self.load_data(symbol, timeframe, storage_type, processed)
            
            if df.empty:
                data_logger.warning(f"Nessun dato da esportare per {symbol} con timeframe {timeframe}")
                return ""
            
            # Determina il percorso di esportazione
            if export_path is None:
                from ..config import EXPORTS_DIR
                processed_str = "processed" if processed else "raw"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{symbol}_{timeframe}_{processed_str}_{timestamp}.csv"
                export_path = EXPORTS_DIR / filename
            
            # Esporta i dati
            path = self.file_storage.export_data(df, export_path, 'csv')
            
            data_logger.info(f"Dati esportati in {path}")
            return path
        
        except Exception as e:
            data_logger.error(f"Errore durante l'esportazione dei dati per {symbol} con timeframe {timeframe}: {e}")
            return ""


# Funzioni factory

def get_file_storage() -> MarketDataStorage:
    """
    Ottiene uno storage basato su file per i dati di mercato.
    
    Returns:
        Istanza del MarketDataStorage
    """
    return MarketDataStorage()


def get_persistence_manager(db_repository=None, file_storage=None) -> DataPersistenceManager:
    """
    Ottiene un gestore di persistenza per i dati.
    
    Args:
        db_repository: Repository del database (opzionale)
        file_storage: Storage basato su file (opzionale)
        
    Returns:
        Istanza del DataPersistenceManager
    """
    return DataPersistenceManager(db_repository, file_storage)