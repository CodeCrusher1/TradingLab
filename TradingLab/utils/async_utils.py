# Utilit� per operazioni asincrone

"""
Utilità per operazioni asincrone nel progetto TradingLab.
Fornisce strumenti per gestire thread, processi e operazioni asincrone.
"""
import threading
import multiprocessing
import queue
import time
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union
from functools import wraps

from .logger import app_logger
from .exceptions import AsyncError, TimeoutError

# Definizione di tipi generici
T = TypeVar('T')  # Tipo di input
R = TypeVar('R')  # Tipo di output


class AsyncResult(Generic[R]):
    """Contenitore per risultati asincroni."""
    
    def __init__(self):
        """Inizializza un nuovo risultato asincrono."""
        self._result: Optional[R] = None
        self._exception: Optional[Exception] = None
        self._completed: threading.Event = threading.Event()
    
    def set_result(self, result: R) -> None:
        """Imposta il risultato e segna come completato."""
        self._result = result
        self._completed.set()
    
    def set_exception(self, exception: Exception) -> None:
        """Imposta un'eccezione e segna come completato."""
        self._exception = exception
        self._completed.set()
    
    def get(self, timeout: Optional[float] = None) -> R:
        """
        Ottiene il risultato, attendendo se necessario.
        
        Args:
            timeout: Timeout in secondi, None per attendere indefinitamente
            
        Returns:
            Il risultato
            
        Raises:
            TimeoutError: Se il timeout scade
            Exception: L'eccezione catturata durante l'esecuzione
        """
        if not self._completed.wait(timeout):
            raise TimeoutError("Timeout durante l'attesa del risultato asincrono")
        
        if self._exception:
            raise self._exception
        
        return self._result


class Task(Generic[T, R]):
    """Rappresenta un'attività asincrona con input, funzione e risultato."""
    
    def __init__(self, func: Callable[[T], R], args: T, 
                 callback: Optional[Callable[[R], None]] = None,
                 error_callback: Optional[Callable[[Exception], None]] = None):
        """
        Inizializza un nuovo task.
        
        Args:
            func: Funzione da eseguire
            args: Argomenti per la funzione
            callback: Funzione di callback per il risultato (opzionale)
            error_callback: Funzione di callback per le eccezioni (opzionale)
        """
        self.func = func
        self.args = args
        self.callback = callback
        self.error_callback = error_callback
        self.result = AsyncResult[R]()
    
    def execute(self) -> None:
        """Esegue il task e gestisce risultato/eccezioni."""
        try:
            result = self.func(self.args)
            self.result.set_result(result)
            
            if self.callback:
                try:
                    self.callback(result)
                except Exception as e:
                    app_logger.error(f"Errore nella callback: {e}")
        
        except Exception as e:
            self.result.set_exception(e)
            
            if self.error_callback:
                try:
                    self.error_callback(e)
                except Exception as callback_e:
                    app_logger.error(f"Errore nella error_callback: {callback_e}")


class ThreadPool:
    """Pool di thread per eseguire task in parallelo."""
    
    def __init__(self, num_workers: int = 4):
        """
        Inizializza un nuovo pool di thread.
        
        Args:
            num_workers: Numero di worker nel pool
        """
        self._queue: queue.Queue = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._running: bool = True
        self._lock: threading.Lock = threading.Lock()
        
        # Avvia i worker
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self) -> None:
        """Funzione principale del worker che processa task dalla coda."""
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
                try:
                    task.execute()
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                app_logger.error(f"Errore nel worker thread: {e}")
    
    def submit(self, task: Task) -> AsyncResult:
        """
        Invia un task al pool.
        
        Args:
            task: Task da eseguire
            
        Returns:
            Oggetto AsyncResult per ottenere il risultato
            
        Raises:
            AsyncError: Se il pool è stato chiuso
        """
        with self._lock:
            if not self._running:
                raise AsyncError("Il ThreadPool è stato chiuso")
            
            self._queue.put(task)
            return task.result
    
    def submit_func(self, func: Callable[[T], R], args: T, 
                   callback: Optional[Callable[[R], None]] = None,
                   error_callback: Optional[Callable[[Exception], None]] = None) -> AsyncResult:
        """
        Invia una funzione al pool.
        
        Args:
            func: Funzione da eseguire
            args: Argomenti per la funzione
            callback: Funzione di callback per il risultato (opzionale)
            error_callback: Funzione di callback per le eccezioni (opzionale)
            
        Returns:
            Oggetto AsyncResult per ottenere il risultato
        """
        task = Task(func, args, callback, error_callback)
        return self.submit(task)
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Attende il completamento di tutti i task.
        
        Args:
            timeout: Timeout in secondi, None per attendere indefinitamente
            
        Returns:
            True se tutti i task sono completati, False se è scaduto il timeout
        """
        try:
            self._queue.join()
            return True
        except Exception:
            return False
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Chiude il pool di thread.
        
        Args:
            wait: Se attendere il completamento di task in sospeso
        """
        with self._lock:
            self._running = False
        
        if wait:
            self.wait()


# Pool di thread globale predefinito
default_thread_pool = ThreadPool()


def run_in_thread(func: Callable) -> Callable:
    """
    Decoratore per eseguire una funzione in un thread separato.
    
    Args:
        func: Funzione da decorare
        
    Returns:
        Funzione decorata che restituisce un AsyncResult
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> AsyncResult:
        # Crea una funzione che prende un singolo argomento (tuple di args e kwargs)
        def task_func(inputs: Tuple[Tuple, Dict]) -> Any:
            task_args, task_kwargs = inputs
            return func(*task_args, **task_kwargs)
        
        # Invia al pool di thread
        return default_thread_pool.submit_func(task_func, (args, kwargs))
    
    return wrapper


class ProcessPool:
    """Pool di processi per eseguire task in parallelo."""
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Inizializza un nuovo pool di processi.
        
        Args:
            num_workers: Numero di worker nel pool (default: numero di CPU)
        """
        self._num_workers = num_workers or multiprocessing.cpu_count()
        self._pool = concurrent.futures.ProcessPoolExecutor(max_workers=self._num_workers)
        app_logger.debug(f"ProcessPool inizializzato con {self._num_workers} worker")
    
    def submit(self, func: Callable[..., R], *args, **kwargs) -> concurrent.futures.Future:
        """
        Invia una funzione al pool.
        
        Args:
            func: Funzione da eseguire
            *args: Argomenti posizionali
            **kwargs: Argomenti per nome
            
        Returns:
            Future per recuperare il risultato
        """
        return self._pool.submit(func, *args, **kwargs)
    
    def map(self, func: Callable[[T], R], iterable: List[T], 
           timeout: Optional[float] = None) -> List[R]:
        """
        Applica una funzione a ogni elemento dell'iterabile.
        
        Args:
            func: Funzione da applicare
            iterable: Iterabile di input
            timeout: Timeout in secondi per l'intera operazione
            
        Returns:
            Lista dei risultati
            
        Raises:
            TimeoutError: Se il timeout scade
            Exception: Altre eccezioni durante l'esecuzione
        """
        try:
            return list(self._pool.map(func, iterable, timeout=timeout))
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout durante la map dopo {timeout} secondi")
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Chiude il pool di processi.
        
        Args:
            wait: Se attendere il completamento di task in sospeso
        """
        self._pool.shutdown(wait=wait)
        app_logger.debug("ProcessPool chiuso")


# Pool di processi globale predefinito
default_process_pool = None


def get_process_pool() -> ProcessPool:
    """
    Ottiene o crea il pool di processi globale.
    
    Returns:
        Pool di processi globale
    """
    global default_process_pool
    if default_process_pool is None:
        default_process_pool = ProcessPool()
    return default_process_pool


def run_in_process(func: Callable) -> Callable:
    """
    Decoratore per eseguire una funzione in un processo separato.
    
    Args:
        func: Funzione da decorare
        
    Returns:
        Funzione decorata che restituisce un Future
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> concurrent.futures.Future:
        pool = get_process_pool()
        return pool.submit(func, *args, **kwargs)
    
    return wrapper


class Timer:
    """Classe per misurare il tempo di esecuzione."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Inizializza un nuovo timer.
        
        Args:
            name: Nome del timer per l'identificazione (opzionale)
        """
        self.name = name or "Timer"
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> 'Timer':
        """Avvia il timer quando usato con with."""
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        """Ferma il timer quando esce dal blocco with."""
        self.stop()
        self.log()
    
    def start(self) -> None:
        """Avvia il timer."""
        self.start_time = time.time()
    
    def stop(self) -> None:
        """Ferma il timer."""
        self.end_time = time.time()
    
    def reset(self) -> None:
        """Resetta il timer."""
        self.start_time = None
        self.end_time = None
    
    def get_elapsed(self) -> float:
        """
        Calcola il tempo trascorso in secondi.
        
        Returns:
            Tempo trascorso o 0 se non avviato
        """
        if self.start_time is None:
            return 0
        
        end = self.end_time or time.time()
        return end - self.start_time
    
    def log(self, level: str = 'info') -> None:
        """
        Registra il tempo trascorso nel log.
        
        Args:
            level: Livello di log ('debug', 'info', etc.)
        """
        elapsed = self.get_elapsed()
        message = f"{self.name}: {elapsed:.4f} secondi"
        
        if level == 'debug':
            app_logger.debug(message)
        elif level == 'info':
            app_logger.info(message)
        elif level == 'warning':
            app_logger.warning(message)
        else:
            app_logger.info(message)
    
    def __str__(self) -> str:
        """Rappresentazione del timer come stringa."""
        elapsed = self.get_elapsed()
        return f"{self.name}: {elapsed:.4f} secondi"


def time_it(func: Callable) -> Callable:
    """
    Decoratore per misurare il tempo di esecuzione di una funzione.
    
    Args:
        func: Funzione da decorare
        
    Returns:
        Funzione decorata
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        timer_name = f"{func.__module__}.{func.__name__}"
        with Timer(timer_name) as timer:
            result = func(*args, **kwargs)
        return result
    
    return wrapper


class RateLimiter:
    """Classe per limitare la frequenza di chiamate a una funzione."""
    
    def __init__(self, calls: int, period: float):
        """
        Inizializza un nuovo rate limiter.
        
        Args:
            calls: Numero massimo di chiamate
            period: Periodo in secondi
        """
        self.calls = calls
        self.period = period
        self.timestamps: List[float] = []
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decoratore per limitare la frequenza di chiamate."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with self._lock:
                now = time.time()
                
                # Rimuovi timestamp più vecchi del periodo
                self.timestamps = [t for t in self.timestamps if now - t <= self.period]
                
                # Verifica se abbiamo superato il limite
                if len(self.timestamps) >= self.calls:
                    oldest = self.timestamps[0]
                    sleep_time = self.period - (now - oldest)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        now = time.time()
                
                # Aggiungi il timestamp corrente e chiama la funzione
                self.timestamps.append(now)
                return func(*args, **kwargs)
        
        return wrapper


class PeriodicTask:
    """Classe per eseguire un task periodicamente in un thread separato."""
    
    def __init__(self, func: Callable, interval: float, 
                args: Optional[Tuple] = None, kwargs: Optional[Dict] = None,
                start_immediately: bool = True):
        """
        Inizializza un nuovo task periodico.
        
        Args:
            func: Funzione da eseguire
            interval: Intervallo in secondi
            args: Argomenti posizionali (opzionale)
            kwargs: Argomenti per nome (opzionale)
            start_immediately: Se avviare immediatamente
        """
        self.func = func
        self.interval = interval
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        if start_immediately:
            self.start()
    
    def _run(self) -> None:
        """Esegue il task periodicamente finché non viene fermato."""
        while not self._stop_event.is_set():
            try:
                self.func(*self.args, **self.kwargs)
            except Exception as e:
                app_logger.error(f"Errore nell'esecuzione del task periodico: {e}")
            
            # Attendi l'intervallo o fino a quando viene settato l'evento di stop
            self._stop_event.wait(self.interval)
    
    def start(self) -> None:
        """Avvia il task periodico in un thread separato."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        app_logger.debug(f"Task periodico avviato con intervallo di {self.interval} secondi")
    
    def stop(self) -> None:
        """Ferma il task periodico."""
        if not self.running:
            return
        
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 1)
        
        self.running = False
        app_logger.debug("Task periodico fermato")
    
    def is_running(self) -> bool:
        """Verifica se il task è in esecuzione."""
        return self.running


def retry(max_attempts: int = 3, delay: float = 1.0, 
         backoff: float = 2.0, exceptions: Tuple = (Exception,)):
    """
    Decoratore per riprovare una funzione in caso di eccezione.
    
    Args:
        max_attempts: Numero massimo di tentativi
        delay: Ritardo iniziale tra i tentativi (secondi)
        backoff: Fattore di backoff per aumentare il ritardo
        exceptions: Tuple di eccezioni da catturare
    
    Returns:
        Decoratore
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    if attempt == max_attempts:
                        app_logger.error(f"Tentativi esauriti ({max_attempts}) per {func.__name__}: {e}")
                        raise
                    
                    app_logger.warning(f"Tentativo {attempt}/{max_attempts} fallito per {func.__name__}: {e}")
                    app_logger.warning(f"Riprovo tra {current_delay:.2f} secondi...")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
        
        return wrapper
    
    return decorator