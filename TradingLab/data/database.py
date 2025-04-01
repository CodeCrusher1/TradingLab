# Gestione database locale (SQLite)

"""
Modulo per la gestione del database nel progetto TradingLab.
Fornisce classi e funzioni per interagire con database MySQL e SQLite.
"""
import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from contextlib import contextmanager
from pathlib import Path
import datetime

# Importazioni dal modulo config
from ..config import (
    DB_CONFIG, DEFAULT_DB_TYPE, SQLITE_DB_PATH, 
    SYMBOLS, TIMEFRAMES
)

# Importazioni dal modulo utils
from ..utils import (
    app_logger, data_logger, DatabaseError, 
    ensure_directory, time_it, handle_exception
)

# Importazioni dal modulo validators
from .validators import MarketDataValidator


# Definizione delle istruzioni SQL per la creazione delle tabelle
CREATE_RAW_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS raw_market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
)
"""

CREATE_PROCESSED_DATA_TABLE = """
CREATE TABLE IF NOT EXISTS processed_market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_data_id INTEGER,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    ema_21 REAL,
    ema_50 REAL,
    ema_200 REAL,
    rsi_14 REAL,
    atr_14 REAL,
    premium_discount REAL,
    fair_value_gap BOOLEAN,
    order_block_bullish BOOLEAN,
    order_block_bearish BOOLEAN,
    liquidity_level_above REAL,
    liquidity_level_below REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (raw_data_id) REFERENCES raw_market_data(id),
    UNIQUE(symbol, timeframe, timestamp)
)
"""

CREATE_MODEL_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    direction INTEGER,
    confidence REAL,
    entry_price REAL,
    tp_price REAL,
    sl_price REAL,
    risk_reward_ratio REAL,
    recommended_action VARCHAR(20),
    model_version VARCHAR(50),
    UNIQUE(symbol, timeframe, timestamp)
)
"""

CREATE_ADVANCED_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS advanced_model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    direction INTEGER,
    confidence REAL,
    ta_lstm_prediction INTEGER,
    ta_lstm_confidence REAL,
    transformer_prediction INTEGER,
    transformer_confidence REAL,
    entry_price REAL,
    tp_price REAL,
    sl_price REAL,
    risk_reward_ratio REAL,
    recommended_action VARCHAR(20),
    order_type VARCHAR(20),
    probability_down REAL,
    probability_neutral REAL,
    probability_up REAL,
    model_version VARCHAR(50),
    UNIQUE(symbol, timeframe, timestamp)
)
"""

CREATE_BACKTEST_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    start_date DATETIME NOT NULL,
    end_date DATETIME NOT NULL,
    initial_capital REAL NOT NULL,
    final_capital REAL NOT NULL,
    total_return REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    win_rate REAL NOT NULL,
    profit_factor REAL NOT NULL,
    total_trades INTEGER NOT NULL,
    model_name VARCHAR(50),
    parameters TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""


class DatabaseManager:
    """
    Classe base per la gestione di database.
    Definisce l'interfaccia comune per diversi tipi di database.
    """
    
    def __init__(self):
        """Inizializza il gestore database."""
        pass
    
    def connect(self) -> Any:
        """
        Crea una connessione al database.
        
        Returns:
            Oggetto connessione
        """
        raise NotImplementedError("Il metodo connect deve essere implementato dalle sottoclassi")
    
    @contextmanager
    def get_connection(self) -> Any:
        """
        Context manager per gestire la connessione al database.
        
        Yields:
            Oggetto connessione
        """
        conn = None
        try:
            conn = self.connect()
            yield conn
        finally:
            if conn is not None:
                conn.close()
    
    def initialize_database(self) -> None:
        """Inizializza il database creando tabelle se non esistono."""
        raise NotImplementedError("Il metodo initialize_database deve essere implementato dalle sottoclassi")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Esegue una query SQL.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            Numero di righe interessate
        """
        raise NotImplementedError("Il metodo execute_query deve essere implementato dalle sottoclassi")
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Esegue una query SQL con più set di parametri.
        
        Args:
            query: Query SQL
            params_list: Lista di parametri per la query
            
        Returns:
            Numero di righe interessate
        """
        raise NotImplementedError("Il metodo execute_many deve essere implementato dalle sottoclassi")
    
    def fetch_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        Esegue una query SQL e restituisce i risultati.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            Lista di dizionari con i risultati
        """
        raise NotImplementedError("Il metodo fetch_query deve essere implementato dalle sottoclassi")
    
    def fetch_df(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Esegue una query SQL e restituisce i risultati come DataFrame.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            DataFrame con i risultati
        """
        raise NotImplementedError("Il metodo fetch_df deve essere implementato dalle sottoclassi")


class SQLiteManager(DatabaseManager):
    """
    Gestore per database SQLite.
    """
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Inizializza il gestore SQLite.
        
        Args:
            db_path: Percorso del file database (opzionale)
        """
        super().__init__()
        self.db_path = Path(db_path) if db_path else SQLITE_DB_PATH
        
        # Assicurati che la directory esista
        ensure_directory(self.db_path.parent)
    
    def connect(self) -> sqlite3.Connection:
        """
        Crea una connessione al database SQLite.
        
        Returns:
            Connessione SQLite
            
        Raises:
            DatabaseError: Se non è possibile connettersi
        """
        try:
            # Abilita il supporto per chiavi esterne
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Configurazione per ottenere i risultati come dizionari
            conn.row_factory = sqlite3.Row
            
            return conn
        except Exception as e:
            raise DatabaseError(f"Impossibile connettersi al database SQLite {self.db_path}", 
                               details={"error": str(e)})
    
    def initialize_database(self) -> None:
        """
        Inizializza il database SQLite creando le tabelle se non esistono.
        
        Raises:
            DatabaseError: Se non è possibile creare le tabelle
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Crea le tabelle
                cursor.execute(CREATE_RAW_DATA_TABLE)
                cursor.execute(CREATE_PROCESSED_DATA_TABLE)
                cursor.execute(CREATE_MODEL_PREDICTIONS_TABLE)
                cursor.execute(CREATE_ADVANCED_PREDICTIONS_TABLE)
                cursor.execute(CREATE_BACKTEST_RESULTS_TABLE)
                
                conn.commit()
                
                data_logger.info("Database SQLite inizializzato correttamente")
        except Exception as e:
            raise DatabaseError("Impossibile inizializzare il database SQLite", 
                               details={"error": str(e)})
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Esegue una query SQL.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            Numero di righe interessate
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query", 
                               details={"error": str(e), "query": query})
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Esegue una query SQL con più set di parametri.
        
        Args:
            query: Query SQL
            params_list: Lista di parametri per la query
            
        Returns:
            Numero di righe interessate
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        if not params_list:
            return 0
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query con molti parametri", 
                               details={"error": str(e), "query": query})
    
    def fetch_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        Esegue una query SQL e restituisce i risultati.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            Lista di dizionari con i risultati
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Converti i risultati in lista di dizionari
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query di fetch", 
                               details={"error": str(e), "query": query})
    
    def fetch_df(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Esegue una query SQL e restituisce i risultati come DataFrame.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            DataFrame con i risultati
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        try:
            with self.get_connection() as conn:
                
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
                
                return df
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query di fetch DataFrame", 
                               details={"error": str(e), "query": query})


class MySQLManager(DatabaseManager):
    """
    Gestore per database MySQL.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inizializza il gestore MySQL.
        
        Args:
            config: Configurazione del database (opzionale)
        """
        super().__init__()
        self.config = config or DB_CONFIG["mysql"]
    
    def connect(self) -> Any:
        """
        Crea una connessione al database MySQL.
        
        Returns:
            Connessione MySQL
            
        Raises:
            DatabaseError: Se non è possibile connettersi o se manca il modulo mysql.connector
        """
        try:
            import mysql.connector
            from mysql.connector import Error
            
            conn = mysql.connector.connect(**self.config)
            if conn.is_connected():
                return conn
            else:
                raise DatabaseError("Impossibile stabilire la connessione MySQL")
        except ImportError:
            raise DatabaseError("Modulo mysql.connector non trovato. Installalo con 'pip install mysql-connector-python'")
        except Exception as e:
            raise DatabaseError("Errore nella connessione MySQL", 
                               details={"error": str(e)})
    
    def initialize_database(self) -> None:
        """
        Inizializza il database MySQL creando le tabelle se non esistono.
        
        Raises:
            DatabaseError: Se non è possibile creare le tabelle
        """
        # Aggiorna la sintassi SQL per MySQL
        mysql_raw_data = CREATE_RAW_DATA_TABLE.replace("AUTOINCREMENT", "AUTO_INCREMENT")
        mysql_processed_data = CREATE_PROCESSED_DATA_TABLE.replace("AUTOINCREMENT", "AUTO_INCREMENT")
        mysql_model_predictions = CREATE_MODEL_PREDICTIONS_TABLE.replace("AUTOINCREMENT", "AUTO_INCREMENT")
        mysql_advanced_predictions = CREATE_ADVANCED_PREDICTIONS_TABLE.replace("AUTOINCREMENT", "AUTO_INCREMENT")
        mysql_backtest_results = CREATE_BACKTEST_RESULTS_TABLE.replace("AUTOINCREMENT", "AUTO_INCREMENT")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Crea le tabelle
                cursor.execute(mysql_raw_data)
                cursor.execute(mysql_processed_data)
                cursor.execute(mysql_model_predictions)
                cursor.execute(mysql_advanced_predictions)
                cursor.execute(mysql_backtest_results)
                
                conn.commit()
                
                data_logger.info("Database MySQL inizializzato correttamente")
        except Exception as e:
            raise DatabaseError("Impossibile inizializzare il database MySQL", 
                               details={"error": str(e)})
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> int:
        """
        Esegue una query SQL.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            Numero di righe interessate
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query MySQL", 
                               details={"error": str(e), "query": query})
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Esegue una query SQL con più set di parametri.
        
        Args:
            query: Query SQL
            params_list: Lista di parametri per la query
            
        Returns:
            Numero di righe interessate
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        if not params_list:
            return 0
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query MySQL con molti parametri", 
                               details={"error": str(e), "query": query})
    
    def fetch_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict]:
        """
        Esegue una query SQL e restituisce i risultati.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            Lista di dizionari con i risultati
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                return cursor.fetchall()
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query MySQL di fetch", 
                               details={"error": str(e), "query": query})
    
    def fetch_df(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Esegue una query SQL e restituisce i risultati come DataFrame.
        
        Args:
            query: Query SQL
            params: Parametri della query (opzionale)
            
        Returns:
            DataFrame con i risultati
            
        Raises:
            DatabaseError: Se non è possibile eseguire la query
        """
        try:
            import mysql.connector
            from mysql.connector import Error
            
            with self.get_connection() as conn:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
                
                return df
        except ImportError:
            raise DatabaseError("Modulo mysql.connector non trovato")
        except Exception as e:
            raise DatabaseError("Errore nell'esecuzione della query MySQL di fetch DataFrame", 
                               details={"error": str(e), "query": query})


class MarketDataRepository:
    """
    Repository per accedere e manipolare dati di mercato nel database.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, db_type: str = DEFAULT_DB_TYPE):
        """
        Inizializza il repository dei dati di mercato.
        
        Args:
            db_manager: Gestore database (opzionale)
            db_type: Tipo di database (default: dal config)
        """
        self.db_type = db_type
        
        if db_manager:
            self.db_manager = db_manager
        else:
            # Crea il gestore database appropriato
            if db_type == "sqlite":
                self.db_manager = SQLiteManager()
            elif db_type == "mysql":
                self.db_manager = MySQLManager()
            else:
                raise ValueError(f"Tipo di database non supportato: {db_type}")
            
            # Inizializza il database
            self.db_manager.initialize_database()
    
    @time_it
    def store_raw_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        Salva dati grezzi nel database.
        
        Args:
            df: DataFrame con dati OHLCV
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            
        Returns:
            Numero di righe inserite
            
        Raises:
            DatabaseError: Se non è possibile salvare i dati
        """
        if df.empty:
            data_logger.warning(f"Nessun dato da salvare per {symbol} con timeframe {timeframe}")
            return 0
        
        # Valida i dati
        validator = MarketDataValidator()
        df = validator.validate_ohlcv_consistency(df)
        
        # Prepara i parametri per la query
        insert_query = """
        INSERT OR REPLACE INTO raw_market_data 
        (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Adatta per MySQL se necessario
        if self.db_type == "mysql":
            insert_query = insert_query.replace("INSERT OR REPLACE", "REPLACE")
            insert_query = insert_query.replace("?", "%s")
        
        try:
            params_list = []
            for _, row in df.iterrows():
                # Converti timestamp se necessario
                timestamp = row['timestamp']
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
                
                params = (
                    symbol,
                    timeframe,
                    timestamp,
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']) if not pd.isna(row['volume']) else 0
                )
                params_list.append(params)
            
            # Esegui inserimento batch
            rows_affected = self.db_manager.execute_many(insert_query, params_list)
            data_logger.info(f"Inseriti/aggiornati {rows_affected} record grezzi per {symbol} con timeframe {timeframe}")
            
            return rows_affected
        
        except Exception as e:
            raise DatabaseError(f"Impossibile salvare i dati grezzi per {symbol} con timeframe {timeframe}", 
                               details={"error": str(e)})
    
    @time_it
    def store_processed_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        Salva dati elaborati nel database.
        
        Args:
            df: DataFrame con dati elaborati
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            
        Returns:
            Numero di righe inserite
            
        Raises:
            DatabaseError: Se non è possibile salvare i dati
        """
        if df.empty:
            data_logger.warning(f"Nessun dato elaborato da salvare per {symbol} con timeframe {timeframe}")
            return 0
        
        # Ottieni gli ID dei dati grezzi corrispondenti
        df_with_ids = self._get_raw_data_ids(df, symbol, timeframe)
        
        # Prepara i parametri per la query
        insert_query = """
        INSERT OR REPLACE INTO processed_market_data 
        (raw_data_id, symbol, timeframe, timestamp, ema_21, ema_50, ema_200, 
         rsi_14, atr_14, premium_discount, fair_value_gap, 
         order_block_bullish, order_block_bearish, 
         liquidity_level_above, liquidity_level_below)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Adatta per MySQL se necessario
        if self.db_type == "mysql":
            insert_query = insert_query.replace("INSERT OR REPLACE", "REPLACE")
            insert_query = insert_query.replace("?", "%s")
        
        try:
            params_list = []
            for _, row in df_with_ids.iterrows():
                # Converti timestamp se necessario
                timestamp = row['timestamp']
                if isinstance(timestamp, pd.Timestamp):
                    timestamp = timestamp.to_pydatetime()
                
                # Converte valori booleani in interi per il database
                fair_value_gap = int(row.get('fair_value_gap', False))
                order_block_bullish = int(row.get('order_block_bullish', False))
                order_block_bearish = int(row.get('order_block_bearish', False))
                
                params = (
                    int(row['raw_data_id']) if 'raw_data_id' in row else None,
                    symbol,
                    timeframe,
                    timestamp,
                    float(row.get('ema_21', 0)) if not pd.isna(row.get('ema_21')) else None,
                    float(row.get('ema_50', 0)) if not pd.isna(row.get('ema_50')) else None,
                    float(row.get('ema_200', 0)) if not pd.isna(row.get('ema_200')) else None,
                    float(row.get('rsi_14', 0)) if not pd.isna(row.get('rsi_14')) else None,
                    float(row.get('atr_14', 0)) if not pd.isna(row.get('atr_14')) else None,
                    float(row.get('premium_discount', 0)) if not pd.isna(row.get('premium_discount')) else None,
                    fair_value_gap,
                    order_block_bullish,
                    order_block_bearish,
                    float(row.get('liquidity_level_above', 0)) if not pd.isna(row.get('liquidity_level_above')) else None,
                    float(row.get('liquidity_level_below', 0)) if not pd.isna(row.get('liquidity_level_below')) else None
                )
                params_list.append(params)
            
            # Esegui inserimento batch
            rows_affected = self.db_manager.execute_many(insert_query, params_list)
            data_logger.info(f"Inseriti/aggiornati {rows_affected} record elaborati per {symbol} con timeframe {timeframe}")
            
            return rows_affected
        
        except Exception as e:
            raise DatabaseError(f"Impossibile salvare i dati elaborati per {symbol} con timeframe {timeframe}", 
                               details={"error": str(e)})
    
    def _get_raw_data_ids(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Recupera gli ID dei dati grezzi corrispondenti ai timestamp.
        
        Args:
            df: DataFrame con dati elaborati
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            
        Returns:
            DataFrame con colonna 'raw_data_id' aggiunta
        """
        try:
            # Estrai i timestamp dal DataFrame
            timestamps = df['timestamp'].tolist()
            
            # Formato per la query parametrica IN
            placeholders = ', '.join(['?'] * len(timestamps))
            if self.db_type == "mysql":
                placeholders = ', '.join(['%s'] * len(timestamps))
            
            # Query per ottenere gli ID
            query = f"""
            SELECT id as raw_data_id, timestamp
            FROM raw_market_data
            WHERE symbol = ? AND timeframe = ? AND timestamp IN ({placeholders})
            """
            
            # Adatta per MySQL se necessario
            if self.db_type == "mysql":
                query = query.replace("?", "%s")
            
            # Prepara parametri
            params = [symbol, timeframe] + timestamps
            
            # Esegui query
            id_df = self.db_manager.fetch_df(query, tuple(params))
            
            # Unisci con DataFrame originale su timestamp
            if not id_df.empty:
                result_df = pd.merge(df, id_df, on='timestamp', how='left')
                return result_df
            
            return df  # Restituisci DataFrame originale se non ci sono corrispondenze
        
        except Exception as e:
            data_logger.error(f"Errore nel recupero degli ID raw_data: {e}")
            return df  # Restituisci DataFrame originale in caso di errore
    
    @time_it
    def fetch_raw_data(self, symbol: str, timeframe: str, 
                      start_date: Optional[datetime.datetime] = None,
                      end_date: Optional[datetime.datetime] = None,
                      limit: int = 1000) -> pd.DataFrame:
        """
        Recupera dati grezzi dal database.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            start_date: Data di inizio (opzionale)
            end_date: Data di fine (opzionale)
            limit: Limite di righe da recuperare
            
        Returns:
            DataFrame con dati grezzi
        """
        try:
            # Base della query
            query = """
            SELECT id, timestamp, open, high, low, close, volume
            FROM raw_market_data
            WHERE symbol = ? AND timeframe = ?
            """
            
            # Filtri opzionali per date
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            # Ordinamento e limite
            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)
            
            # Adatta per MySQL se necessario
            if self.db_type == "mysql":
                query = query.replace("?", "%s")
            
            # Esegui query
            df = self.db_manager.fetch_df(query, tuple(params))
            
            # Converti timestamp in datetime
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            data_logger.info(f"Recuperati {len(df)} record grezzi per {symbol} con timeframe {timeframe}")
            return df
        
        except Exception as e:
            data_logger.error(f"Errore nel recupero dei dati grezzi: {e}")
            return pd.DataFrame()
    
    @time_it
    def fetch_processed_data(self, symbol: str, timeframe: str, 
                            start_date: Optional[datetime.datetime] = None,
                            end_date: Optional[datetime.datetime] = None,
                            include_raw: bool = True,
                            limit: int = 1000) -> pd.DataFrame:
        """
        Recupera dati elaborati dal database.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            start_date: Data di inizio (opzionale)
            end_date: Data di fine (opzionale)
            include_raw: Se includere anche i dati OHLCV grezzi
            limit: Limite di righe da recuperare
            
        Returns:
            DataFrame con dati elaborati
        """
        try:
            # Base della query
            if include_raw:
                query = """
                SELECT p.*, r.open, r.high, r.low, r.close, r.volume
                FROM processed_market_data p
                JOIN raw_market_data r ON p.raw_data_id = r.id
                WHERE p.symbol = ? AND p.timeframe = ?
                """
            else:
                query = """
                SELECT *
                FROM processed_market_data
                WHERE symbol = ? AND timeframe = ?
                """
            
            # Filtri opzionali per date
            params = [symbol, timeframe]
            
            if start_date:
                query += " AND p.timestamp >= ?" if include_raw else " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND p.timestamp <= ?" if include_raw else " AND timestamp <= ?"
                params.append(end_date)
            
            # Ordinamento e limite
            query += " ORDER BY " + ("p.timestamp" if include_raw else "timestamp") + " ASC LIMIT ?"
            params.append(limit)
            
            # Adatta per MySQL se necessario
            if self.db_type == "mysql":
                query = query.replace("?", "%s")
            
            # Esegui query
            df = self.db_manager.fetch_df(query, tuple(params))
            
            # Converti timestamp in datetime
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Converti colonne booleane
            bool_columns = ['fair_value_gap', 'order_block_bullish', 'order_block_bearish']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
            
            data_logger.info(f"Recuperati {len(df)} record elaborati per {symbol} con timeframe {timeframe}")
            return df
        
        except Exception as e:
            data_logger.error(f"Errore nel recupero dei dati elaborati: {e}")
            return pd.DataFrame()
    
    @time_it
    def store_model_prediction(self, prediction: Dict[str, Any], advanced: bool = False) -> bool:
        """
        Salva una previsione del modello nel database.
        
        Args:
            prediction: Dizionario con i dati della previsione
            advanced: Se è una previsione avanzata
            
        Returns:
            True se l'operazione ha successo
        """
        try:
            if advanced:
                # Query per previsioni avanzate
                query = """
                INSERT OR REPLACE INTO advanced_model_predictions
                (symbol, timeframe, timestamp, direction, confidence,
                ta_lstm_prediction, ta_lstm_confidence, transformer_prediction, transformer_confidence,
                entry_price, tp_price, sl_price, risk_reward_ratio,
                recommended_action, order_type,
                probability_down, probability_neutral, probability_up,
                model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Estrai parametri da prediction
                params = (
                    prediction.get('symbol'),
                    prediction.get('timeframe'),
                    prediction.get('timestamp'),
                    prediction.get('direction'),
                    prediction.get('confidence'),
                    prediction.get('ta_lstm_prediction'),
                    prediction.get('ta_lstm_confidence'),
                    prediction.get('transformer_prediction'),
                    prediction.get('transformer_confidence'),
                    prediction.get('entry_price'),
                    prediction.get('tp_price'),
                    prediction.get('sl_price'),
                    prediction.get('risk_reward_ratio', 0),
                    prediction.get('action'),
                    prediction.get('order_type', 'wait'),
                    prediction.get('probability_distribution', {}).get('down', 0),
                    prediction.get('probability_distribution', {}).get('neutral', 0),
                    prediction.get('probability_distribution', {}).get('up', 0),
                    prediction.get('model_version', 'unknown')
                )
            else:
                # Query per previsioni standard
                query = """
                INSERT OR REPLACE INTO model_predictions
                (symbol, timeframe, timestamp, direction, confidence,
                entry_price, tp_price, sl_price, risk_reward_ratio,
                recommended_action, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                # Estrai parametri da prediction
                params = (
                    prediction.get('symbol'),
                    prediction.get('timeframe'),
                    prediction.get('timestamp'),
                    prediction.get('direction'),
                    prediction.get('confidence'),
                    prediction.get('entry_price'),
                    prediction.get('tp_price'),
                    prediction.get('sl_price'),
                    prediction.get('risk_reward_ratio', 0),
                    prediction.get('action'),
                    prediction.get('model_version', 'unknown')
                )
            
            # Adatta per MySQL se necessario
            if self.db_type == "mysql":
                query = query.replace("INSERT OR REPLACE", "REPLACE")
                query = query.replace("?", "%s")
            
            # Esegui query
            rows_affected = self.db_manager.execute_query(query, params)
            
            model_type = "avanzato" if advanced else "standard"
            data_logger.info(f"Previsione {model_type} salvata per {prediction.get('symbol')} "
                            f"con timeframe {prediction.get('timeframe')}")
            
            return rows_affected > 0
        
        except Exception as e:
            data_logger.error(f"Errore nel salvataggio della previsione: {e}")
            return False
    
    @time_it
    def fetch_model_predictions(self, symbol: str, timeframe: str, 
                               advanced: bool = False, limit: int = 100) -> pd.DataFrame:
        """
        Recupera previsioni dei modelli dal database.
        
        Args:
            symbol: Nome del simbolo
            timeframe: Nome del timeframe
            advanced: Se recuperare previsioni avanzate
            limit: Limite di righe da recuperare
            
        Returns:
            DataFrame con previsioni
        """
        try:
            # Seleziona la tabella appropriata
            table = "advanced_model_predictions" if advanced else "model_predictions"
            
            # Query base
            query = f"""
            SELECT *
            FROM {table}
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """
            
            # Parametri
            params = (symbol, timeframe, limit)
            
            # Adatta per MySQL se necessario
            if self.db_type == "mysql":
                query = query.replace("?", "%s")
            
            # Esegui query
            df = self.db_manager.fetch_df(query, params)
            
            # Converti timestamp in datetime
            if not df.empty:
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                if 'prediction_timestamp' in df.columns:
                    df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
            
            model_type = "avanzate" if advanced else "standard"
            data_logger.info(f"Recuperate {len(df)} previsioni {model_type} per {symbol} "
                            f"con timeframe {timeframe}")
            
            return df
        
        except Exception as e:
            data_logger.error(f"Errore nel recupero delle previsioni: {e}")
            return pd.DataFrame()
    
    @time_it
    def get_available_data_summary(self) -> pd.DataFrame:
        """
        Ottiene un riepilogo dei dati disponibili.
        
        Returns:
            DataFrame con riepilogo dei dati
        """
        try:
            # Query per ottenere il riepilogo
            query = """
            SELECT 
                symbol, 
                timeframe, 
                COUNT(*) as record_count,
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM raw_market_data
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
            """
            
            # Esegui query
            df = self.db_manager.fetch_df(query)
            
            # Converti timestamp in datetime
            if not df.empty:
                if 'start_date' in df.columns:
                    df['start_date'] = pd.to_datetime(df['start_date'])
                
                if 'end_date' in df.columns:
                    df['end_date'] = pd.to_datetime(df['end_date'])
            
            data_logger.info(f"Riepilogo dati disponibili recuperato: {len(df)} combinazioni")
            return df
        
        except Exception as e:
            data_logger.error(f"Errore nel recupero del riepilogo dati: {e}")
            return pd.DataFrame()


class DatabaseFactory:
    """
    Factory per creare gestori database.
    """
    
    @staticmethod
    def get_db_manager(db_type: str = DEFAULT_DB_TYPE) -> DatabaseManager:
        """
        Ottiene un gestore database del tipo specificato.
        
        Args:
            db_type: Tipo di database ('sqlite' o 'mysql')
            
        Returns:
            Gestore database appropriato
            
        Raises:
            ValueError: Se il tipo di database non è supportato
        """
        if db_type == "sqlite":
            return SQLiteManager()
        elif db_type == "mysql":
            return MySQLManager()
        else:
            raise ValueError(f"Tipo di database non supportato: {db_type}")
    
    @staticmethod
    def get_repository(db_type: str = DEFAULT_DB_TYPE) -> MarketDataRepository:
        """
        Ottiene un repository per i dati di mercato.
        
        Args:
            db_type: Tipo di database ('sqlite' o 'mysql')
            
        Returns:
            Repository per i dati di mercato
        """
        db_manager = DatabaseFactory.get_db_manager(db_type)
        return MarketDataRepository(db_manager, db_type)


# Inizializza e ottieni il repository predefinito
def get_default_repository() -> MarketDataRepository:
    """
    Ottiene il repository predefinito per i dati di mercato.
    
    Returns:
        Repository per i dati di mercato
    """
    return DatabaseFactory.get_repository()