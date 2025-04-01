# Sistema di download dati da Yahoo Finance

"""
Modulo per il download dei dati finanziari per il progetto TradingLab.
Fornisce classi e funzioni per scaricare dati storici e in tempo reale da varie fonti.
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import threading
import queue

# Importazioni dal modulo config
from ..config import (
    SYMBOLS, TIMEFRAMES, API_RATE_LIMIT, API_TIMEOUT,
    get_symbol, get_timeframe
)

# Importazioni dal modulo utils
from ..utils import (
    app_logger, data_logger, DownloadError, RateLimitError, TimeoutError,
    time_it, AsyncResult, Task, ThreadPool, RateLimiter, retry
)

# Importazioni dal modulo validators
from .validators import (
    MarketDataValidator, YahooFinanceValidator, validate_date_str,
    is_valid_symbol, is_valid_timeframe
)


class DataDownloader:
    """
    Classe base per i downloader di dati.
    Definisce l'interfaccia comune per diversi tipi di downloader.
    """
    
    def __init__(self):
        """Inizializza il downloader."""
        pass
    
    def download_historical(self, symbol: str, timeframe: str, 
                           start_date: Optional[Union[str, datetime]] = None,
                           end_date: Optional[Union[str, datetime]] = None,
                           period: Optional[str] = None) -> pd.DataFrame:
        """
        Scarica dati storici.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe per i dati
            start_date: Data di inizio (opzionale)
            end_date: Data di fine (opzionale)
            period: Periodo alternativo (es. '1y', '2mo')
            
        Returns:
            DataFrame con dati storici
        """
        raise NotImplementedError("Il metodo download_historical deve essere implementato dalle sottoclassi")
    
    def download_realtime(self, symbol: str) -> Dict[str, Any]:
        """
        Scarica dati in tempo reale.
        
        Args:
            symbol: Simbolo dell'asset
            
        Returns:
            Dizionario con dati in tempo reale
        """
        raise NotImplementedError("Il metodo download_realtime deve essere implementato dalle sottoclassi")


class YahooFinanceDownloader(DataDownloader):
    """
    Downloader per dati da Yahoo Finance.
    """
    
    def __init__(self):
        """Inizializza il downloader Yahoo Finance."""
        super().__init__()
        self.validator = YahooFinanceValidator()
        self.data_validator = MarketDataValidator()
        self._rate_limiter = threading.Semaphore(5)  # Limita a 5 richieste contemporanee
    
    @retry(max_attempts=3, delay=2.0, backoff=2.0)
    def download_historical(self, symbol: str, timeframe: str, 
                           start_date: Optional[Union[str, datetime]] = None,
                           end_date: Optional[Union[str, datetime]] = None,
                           period: Optional[str] = "max") -> pd.DataFrame:
        """
        Scarica dati storici da Yahoo Finance.
        
        Args:
            symbol: Simbolo dell'asset (nome interno)
            timeframe: Timeframe per i dati (nome interno)
            start_date: Data di inizio (opzionale)
            end_date: Data di fine (opzionale)
            period: Periodo alternativo (es. '1y', '2mo')
            
        Returns:
            DataFrame con dati storici OHLCV
            
        Raises:
            DownloadError: Se il download fallisce
            ValueError: Se il simbolo o il timeframe non è valido
        """
        try:
            import yfinance as yf
        except ImportError:
            raise DownloadError("Modulo yfinance non trovato. Installalo con 'pip install yfinance'")
        
        # Valida e ottieni il ticker Yahoo Finance
        symbol_obj = get_symbol(symbol)
        if symbol_obj is None:
            raise ValueError(f"Simbolo non valido: {symbol}")
        
        ticker_code = symbol_obj.ticker
        
        # Valida e ottieni il codice timeframe Yahoo Finance
        timeframe_obj = get_timeframe(timeframe)
        if timeframe_obj is None:
            raise ValueError(f"Timeframe non valido: {timeframe}")
        
        interval_code = timeframe_obj.code
        
        # Valida le date se specificate
        if start_date and end_date:
            # Usa date esplicite
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            period = None  # Ignora period se date specificate
        elif period:
            # Usa period se non sono specificate le date
            start_date = None
            end_date = None
        else:
            # Default a 'max' se non sono specificate date o period
            period = "max"
            start_date = None
            end_date = None
        
        # Log dell'operazione
        data_logger.info(f"Downloading historical data for {symbol} ({ticker_code}) with timeframe {timeframe} ({interval_code})")
        if start_date and end_date:
            data_logger.info(f"Date range: {start_date} to {end_date}")
        elif period:
            data_logger.info(f"Period: {period}")
        
        try:
            # Usa semaforo per limitare le richieste concorrenti
            with self._rate_limiter:
                # Scarica i dati
                ticker = yf.Ticker(ticker_code)
                
                # Diverse chiamate in base ai parametri forniti
                if start_date and end_date:
                    data = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval_code,
                        auto_adjust=True
                    )
                else:
                    data = ticker.history(
                        period=period,
                        interval=interval_code,
                        auto_adjust=True
                    )
            
            # Gestisci DataFrame vuoto
            if data.empty:
                data_logger.warning(f"Nessun dato trovato per {symbol} con timeframe {timeframe}")
                return pd.DataFrame()
            
            # Converti i dati nel formato standard
            data = self.validator.convert_yahoo_ohlcv(data)
            
            # Aggiungi datetime index come colonna
            if data.index.name in ['Date', 'Datetime']:
                data = data.reset_index()
            
            # Valida i dati
            data = self.data_validator.validate_ohlcv_consistency(data)
            
            data_logger.info(f"Scaricati {len(data)} record per {symbol} con timeframe {timeframe}")
            return data
        
        except Exception as e:
            error_msg = f"Errore durante il download dei dati per {symbol} con timeframe {timeframe}: {str(e)}"
            data_logger.error(error_msg)
            raise DownloadError(error_msg)
    
    @retry(max_attempts=3, delay=1.0, backoff=2.0)
    def download_realtime(self, symbol: str) -> Dict[str, Any]:
        """
        Scarica dati in tempo reale da Yahoo Finance.
        
        Args:
            symbol: Simbolo dell'asset (nome interno)
            
        Returns:
            Dizionario con dati in tempo reale
            
        Raises:
            DownloadError: Se il download fallisce
            ValueError: Se il simbolo non è valido
        """
        try:
            import yfinance as yf
        except ImportError:
            raise DownloadError("Modulo yfinance non trovato. Installalo con 'pip install yfinance'")
        
        # Valida e ottieni il ticker Yahoo Finance
        symbol_obj = get_symbol(symbol)
        if symbol_obj is None:
            raise ValueError(f"Simbolo non valido: {symbol}")
        
        ticker_code = symbol_obj.ticker
        
        data_logger.info(f"Downloading realtime data for {symbol} ({ticker_code})")
        
        try:
            # Usa semaforo per limitare le richieste concorrenti
            with self._rate_limiter:
                # Ottieni i dati in tempo reale
                ticker = yf.Ticker(ticker_code)
                
                # Richiedi dati intraday più recenti
                data = ticker.history(period="1d", interval="1m", prepost=True)
                
                if not data.empty:
                    # Utilizza l'ultimo punto dati disponibile
                    last_row = data.iloc[-1]
                    # Calcola la variazione rispetto all'apertura giornaliera
                    day_open = ticker.history(period="1d").iloc[0]['Open']
                    
                    result = {
                        'price': float(last_row['Close']),
                        'change': float((last_row['Close'] - day_open) / day_open * 100),
                        'volume': int(data['Volume'].sum()),
                        'high': float(data['High'].max()),
                        'low': float(data['Low'].min()),
                        'timestamp': datetime.now()
                    }
                else:
                    # Fallback a dati giornalieri se i dati intraday non sono disponibili
                    data = ticker.history(period="2d")
                    if not data.empty and len(data) >= 2:
                        today = data.iloc[-1]
                        yesterday = data.iloc[-2]
                        
                        result = {
                            'price': float(today['Close']),
                            'change': float((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100),
                            'volume': int(today['Volume']),
                            'high': float(today['High']),
                            'low': float(today['Low']),
                            'timestamp': datetime.now()
                        }
                    elif not data.empty:
                        today = data.iloc[-1]
                        result = {
                            'price': float(today['Close']),
                            'change': float((today['Close'] - today['Open']) / today['Open'] * 100),
                            'volume': int(today['Volume']),
                            'high': float(today['High']),
                            'low': float(today['Low']),
                            'timestamp': datetime.now()
                        }
                    else:
                        raise DownloadError(f"Nessun dato disponibile per {symbol}")
            
            data_logger.info(f"Dati in tempo reale ottenuti per {symbol}: {result['price']}")
            return result
        
        except Exception as e:
            error_msg = f"Errore durante il download dei dati in tempo reale per {symbol}: {str(e)}"
            data_logger.error(error_msg)
            raise DownloadError(error_msg)


class BatchDownloader:
    """
    Classe per il download batch di dati per più simboli e timeframe.
    """
    
    def __init__(self, downloader: Optional[DataDownloader] = None):
        """
        Inizializza il downloader batch.
        
        Args:
            downloader: Downloader specifico da utilizzare (opzionale)
        """
        self.downloader = downloader or YahooFinanceDownloader()
        self.thread_pool = ThreadPool(num_workers=5)
        self.results: Dict[str, AsyncResult] = {}
    
    def start_batch_download(self, symbols: List[str], timeframes: List[str], 
                            start_date: Optional[Union[str, datetime]] = None,
                            end_date: Optional[Union[str, datetime]] = None,
                            period: Optional[str] = None) -> Dict[str, AsyncResult]:
        """
        Avvia il download batch per una lista di simboli e timeframe.
        
        Args:
            symbols: Lista di simboli
            timeframes: Lista di timeframe
            start_date: Data di inizio (opzionale)
            end_date: Data di fine (opzionale)
            period: Periodo alternativo (opzionale)
            
        Returns:
            Dizionario con AsyncResult per ogni combinazione simbolo/timeframe
        """
        self.results = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Crea chiave unica per il risultato
                key = f"{symbol}_{timeframe}"
                
                # Crea task per il download
                download_task = Task(
                    func=self._download_task,
                    args=(symbol, timeframe, start_date, end_date, period),
                    callback=self._download_callback,
                    error_callback=self._download_error_callback
                )
                
                # Invia task al thread pool
                self.results[key] = self.thread_pool.submit(download_task)
                
                data_logger.info(f"Avviato download per {symbol} con timeframe {timeframe}")
                
                # Piccola pausa per evitare sovraccarico
                time.sleep(0.5)
        
        return self.results
    
    def _download_task(self, args: Tuple) -> Tuple[str, str, pd.DataFrame]:
        """
        Task di download per un singolo simbolo/timeframe.
        
        Args:
            args: Tuple con (symbol, timeframe, start_date, end_date, period)
            
        Returns:
            Tuple di (symbol, timeframe, dataframe)
        """
        symbol, timeframe, start_date, end_date, period = args
        
        df = self.downloader.download_historical(
            symbol, timeframe, start_date, end_date, period
        )
        
        return symbol, timeframe, df
    
    def _download_callback(self, result: Tuple[str, str, pd.DataFrame]) -> None:
        """
        Callback quando un download è completato con successo.
        
        Args:
            result: Risultato del download (symbol, timeframe, dataframe)
        """
        symbol, timeframe, df = result
        data_logger.info(f"Download completato per {symbol} con timeframe {timeframe}: {len(df)} record")
    
    def _download_error_callback(self, error: Exception) -> None:
        """
        Callback quando un download fallisce.
        
        Args:
            error: Eccezione di errore
        """
        data_logger.error(f"Errore durante il download batch: {error}")
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Attende il completamento di tutti i download.
        
        Args:
            timeout: Timeout in secondi
            
        Returns:
            Dizionario con DataFrame per ogni combinazione simbolo/timeframe
        """
        results_dict = {}
        
        for key, async_result in self.results.items():
            try:
                symbol, timeframe, df = async_result.get(timeout)
                results_dict[key] = df
            except Exception as e:
                data_logger.error(f"Errore recuperando il risultato per {key}: {e}")
                results_dict[key] = pd.DataFrame()
        
        return results_dict
    
    def get_result(self, symbol: str, timeframe: str, 
                  timeout: Optional[float] = None) -> Optional[pd.DataFrame]:
        """
        Ottiene il risultato per una specifica combinazione simbolo/timeframe.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe
            timeout: Timeout in secondi
            
        Returns:
            DataFrame con i dati o None se non disponibile
        """
        key = f"{symbol}_{timeframe}"
        
        if key not in self.results:
            data_logger.warning(f"Nessun download avviato per {key}")
            return None
        
        try:
            symbol, timeframe, df = self.results[key].get(timeout)
            return df
        except Exception as e:
            data_logger.error(f"Errore recuperando il risultato per {key}: {e}")
            return None


class HistoricalDataUpdater:
    """
    Classe per aggiornare i dati storici nel database.
    """
    
    def __init__(self, db_repository=None, downloader=None):
        """
        Inizializza l'updater.
        
        Args:
            db_repository: Repository del database
            downloader: Downloader per i dati
        """
        # Importa qui per evitare dipendenze circolari
        from .database import get_default_repository
        
        self.db_repository = db_repository or get_default_repository()
        self.downloader = downloader or YahooFinanceDownloader()
        self.validator = MarketDataValidator()
    
    @time_it
    def update_symbol_timeframe(self, symbol: str, timeframe: str, 
                               full_refresh: bool = False) -> int:
        """
        Aggiorna i dati per un simbolo e timeframe specifico.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe
            full_refresh: Se ricaricare tutti i dati
            
        Returns:
            Numero di record aggiornati
        """
        try:
            data_logger.info(f"Aggiornamento dati per {symbol} con timeframe {timeframe}")
            
            if full_refresh:
                # Scarica tutti i dati storici
                df = self.downloader.download_historical(symbol, timeframe)
            else:
                # Ottieni la data più recente dal database
                df_existing = self.db_repository.fetch_raw_data(symbol, timeframe, limit=1)
                
                if df_existing.empty:
                    # Nessun dato esistente, scarica tutti i dati storici
                    data_logger.info(f"Nessun dato esistente per {symbol} con timeframe {timeframe}, scarico l'intero storico")
                    df = self.downloader.download_historical(symbol, timeframe)
                else:
                    # Calcola la data di inizio per l'aggiornamento
                    # Sottraiamo qualche giorno per sicurezza e potenziali correzioni retrospettive
                    latest_date = df_existing['timestamp'].max()
                    start_date = latest_date - timedelta(days=7)
                    
                    data_logger.info(f"Ultimo dato per {symbol} con timeframe {timeframe}: {latest_date}")
                    data_logger.info(f"Scarico dati dal {start_date}")
                    
                    # Scarica i dati mancanti
                    df = self.downloader.download_historical(
                        symbol, timeframe, start_date=start_date
                    )
            
            if df.empty:
                data_logger.warning(f"Nessun nuovo dato trovato per {symbol} con timeframe {timeframe}")
                return 0
            
            # Salva i dati nel database
            records_affected = self.db_repository.store_raw_data(df, symbol, timeframe)
            
            data_logger.info(f"Aggiornati {records_affected} record per {symbol} con timeframe {timeframe}")
            return records_affected
        
        except Exception as e:
            data_logger.error(f"Errore durante l'aggiornamento di {symbol} con timeframe {timeframe}: {e}")
            return 0
    
    @time_it
    def update_all(self, symbols: Optional[List[str]] = None, 
                  timeframes: Optional[List[str]] = None,
                  full_refresh: bool = False) -> Dict[str, int]:
        """
        Aggiorna i dati per tutti i simboli e timeframe specificati.
        
        Args:
            symbols: Lista di simboli (default: tutti)
            timeframes: Lista di timeframe (default: tutti)
            full_refresh: Se ricaricare tutti i dati
            
        Returns:
            Dizionario con numero di record aggiornati per ogni combinazione
        """
        # Usa tutti i simboli e timeframe disponibili se non specificati
        if symbols is None:
            symbols = [s for s in SYMBOLS.keys()]
        
        if timeframes is None:
            timeframes = [t for t in TIMEFRAMES.keys()]
        
        results = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                try:
                    records = self.update_symbol_timeframe(symbol, timeframe, full_refresh)
                    results[key] = records
                except Exception as e:
                    data_logger.error(f"Errore aggiornando {key}: {e}")
                    results[key] = -1
                
                # Piccola pausa per evitare sovraccarico
                time.sleep(1)
        
        return results


class RealtimeDataService:
    """
    Servizio per ottenere dati in tempo reale.
    """
    
    def __init__(self, downloader=None):
        """
        Inizializza il servizio dati in tempo reale.
        
        Args:
            downloader: Downloader per i dati
        """
        self.downloader = downloader or YahooFinanceDownloader()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_time: Dict[str, datetime] = {}
        self.cache_ttl = 60  # Tempo di vita della cache in secondi
        self._lock = threading.Lock()
    
    def get_realtime_data(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:

        """
        Ottiene dati in tempo reale per un simbolo.
        
        Args:
            symbol: Simbolo dell'asset
            use_cache: Se usare la cache
            
        Returns:
            Dizionario con dati in tempo reale
            
        Raises:
            DownloadError: Se il download fallisce
        """
        
        # Verifica se i dati sono in cache e non scaduti
        current_time = datetime.now()
        
        if use_cache and symbol in self.cache:
            with self._lock:
                cache_time = self.cache_time.get(symbol)
                if cache_time and (current_time - cache_time).total_seconds() < self.cache_ttl:
                    return self.cache[symbol]
        
        # Scarica nuovi dati
        try:
            data = self.downloader.download_realtime(symbol)
            
            # Aggiorna la cache
            with self._lock:
                self.cache[symbol] = data
                self.cache_time[symbol] = current_time
            
            return data
        
        except Exception as e:
            # Se il download fallisce e abbiamo dati in cache, restituisci quelli
            if symbol in self.cache:
                data_logger.warning(f"Download in tempo reale fallito per {symbol}, uso dati in cache: {e}")
                return self.cache[symbol]
            else:
                raise DownloadError(f"Impossibile ottenere dati in tempo reale per {symbol}: {e}")
    
    def get_multiple_realtime(self, symbols: List[str], use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Ottiene dati in tempo reale per più simboli.
        
        Args:
            symbols: Lista di simboli
            use_cache: Se usare la cache
            
        Returns:
            Dizionario con dati in tempo reale per ogni simbolo
        """
        results = {}
        errors = []
        
        # Crea una coda per i risultati
        result_queue = queue.Queue()
        
        # Funzione worker per il download
        def download_worker(symbol):
            try:
                data = self.get_realtime_data(symbol, use_cache)
                result_queue.put((symbol, data, None))
            except Exception as e:
                result_queue.put((symbol, None, e))
        
        # Avvia thread separati per ogni simbolo
        threads = []
        for symbol in symbols:
            thread = threading.Thread(target=download_worker, args=(symbol,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Attendi il completamento di tutti i thread
        for thread in threads:
            thread.join(timeout=10)  # Timeout di 10 secondi per thread
        
        # Raccogli i risultati
        while not result_queue.empty():
            symbol, data, error = result_queue.get()
            if error:
                errors.append(f"{symbol}: {error}")
                continue
            
            results[symbol] = data
        
        # Log degli errori
        if errors:
            data_logger.warning(f"Errori nel download multiplo in tempo reale: {errors}")
        
        return results
    
    def clear_cache(self) -> None:
        """Pulisce la cache dei dati in tempo reale."""
        with self._lock:
            self.cache.clear()
            self.cache_time.clear()
        
        data_logger.info("Cache dati in tempo reale pulita")
    
    def set_cache_ttl(self, seconds: int) -> None:
        """
        Imposta il tempo di vita della cache.
        
        Args:
            seconds: Tempo di vita in secondi
        """
        with self._lock:
            self.cache_ttl = seconds
        
        data_logger.info(f"TTL cache impostato a {seconds} secondi")


class DataScheduler:
    """
    Pianificatore per aggiornamenti dati periodici.
    """
    
    def __init__(self, updater=None):
        """
        Inizializza il pianificatore.
        
        Args:
            updater: Updater per i dati
        """
        self.updater = updater or HistoricalDataUpdater()
        self.scheduler_thread = None
        self.stop_event = threading.Event()
        self.schedule: Dict[str, Dict[str, Any]] = {}
    
    def add_schedule(self, symbol: str, timeframe: str, interval_hours: float) -> None:
        """
        Aggiunge una pianificazione per l'aggiornamento di un simbolo/timeframe.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe
            interval_hours: Intervallo di aggiornamento in ore
        """
        key = f"{symbol}_{timeframe}"
        self.schedule[key] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'interval_hours': interval_hours,
            'last_update': None,
            'next_update': datetime.now()
        }
        
        data_logger.info(f"Pianificazione aggiunta per {key} ogni {interval_hours} ore")
    
    def remove_schedule(self, symbol: str, timeframe: str) -> bool:
        """
        Rimuove una pianificazione.
        
        Args:
            symbol: Simbolo dell'asset
            timeframe: Timeframe
            
        Returns:
            True se la pianificazione è stata rimossa
        """
        key = f"{symbol}_{timeframe}"
        if key in self.schedule:
            del self.schedule[key]
            data_logger.info(f"Pianificazione rimossa per {key}")
            return True
        return False
    
    def start(self) -> None:
        """Avvia il pianificatore."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            data_logger.warning("Il pianificatore è già in esecuzione")
            return
        
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        data_logger.info("Pianificatore aggiornamenti dati avviato")
    
    def stop(self) -> None:
        """Ferma il pianificatore."""
        self.stop_event.set()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=30)
        
        data_logger.info("Pianificatore aggiornamenti dati fermato")
    
    def _scheduler_loop(self) -> None:
        """Loop principale del pianificatore."""
        while not self.stop_event.is_set():
            current_time = datetime.now()
            
            for key, schedule_info in list(self.schedule.items()):
                next_update = schedule_info['next_update']
                
                if current_time >= next_update:
                    symbol = schedule_info['symbol']
                    timeframe = schedule_info['timeframe']
                    
                    try:
                        # Aggiorna i dati
                        data_logger.info(f"Esecuzione aggiornamento pianificato per {key}")
                        records = self.updater.update_symbol_timeframe(symbol, timeframe)
                        
                        # Aggiorna la pianificazione
                        interval_hours = schedule_info['interval_hours']
                        self.schedule[key]['last_update'] = current_time
                        self.schedule[key]['next_update'] = current_time + timedelta(hours=interval_hours)
                        
                        data_logger.info(f"Aggiornamento completato per {key}: {records} record. "
                                         f"Prossimo aggiornamento a {self.schedule[key]['next_update']}")
                    
                    except Exception as e:
                        data_logger.error(f"Errore nell'aggiornamento pianificato per {key}: {e}")
                        # Riprova tra un'ora in caso di errore
                        self.schedule[key]['next_update'] = current_time + timedelta(hours=1)
            
            # Attendi 1 minuto prima di controllare nuovamente
            self.stop_event.wait(60)


# Funzioni factory per i downloader

def create_downloader(type_name: str = "yahoo") -> DataDownloader:
    """
    Crea un downloader del tipo specificato.
    
    Args:
        type_name: Tipo di downloader
        
    Returns:
        Istanza del downloader
        
    Raises:
        ValueError: Se il tipo di downloader non è supportato
    """
    if type_name.lower() == "yahoo":
        return YahooFinanceDownloader()
    else:
        raise ValueError(f"Tipo di downloader non supportato: {type_name}")


def get_batch_downloader() -> BatchDownloader:
    """
    Ottiene un downloader batch.
    
    Returns:
        Istanza del downloader batch
    """
    return BatchDownloader()


def get_updater() -> HistoricalDataUpdater:
    """
    Ottiene un updater per dati storici.
    
    Returns:
        Istanza dell'updater
    """
    return HistoricalDataUpdater()


def get_realtime_service() -> RealtimeDataService:
    """
    Ottiene un servizio per dati in tempo reale.
    
    Returns:
        Istanza del servizio dati in tempo reale
    """
    return RealtimeDataService()


def get_scheduler() -> DataScheduler:
    """
    Ottiene un pianificatore per aggiornamenti dati.
    
    Returns:
        Istanza del pianificatore
    """
    return DataScheduler()