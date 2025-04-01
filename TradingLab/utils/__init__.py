"""
Modulo di utilità per il progetto TradingLab.
Questo modulo fornisce funzionalità di supporto come logging, gestione file e operazioni asincrone.
"""

# Importa le utilità di logging
from .logger import (
    LoggerFactory, LogColors, ColorFormatter,
    app_logger, data_logger, model_logger, error_logger,
    log_exception, configure_root_logger, silence_loggers
)

# Importa le eccezioni personalizzate
from .exceptions import (
    TradingLabException, ConfigError, DataError, ModelError, BacktestError, APIError, GUIError, AsyncError,
    PathNotFoundError, DownloadError, ProcessingError, ValidationError, DatabaseError,
    InvalidSymbolError, InvalidTimeframeError, ModelNotFoundError, TrainingError,
    InferenceError, EvaluationError, SimulationError, StrategyError,
    RateLimitError, AuthenticationError, ChartError, TimeoutError,
    handle_exception
)

# Importa le utilità per i file
from .file_manager import (
    FileError, ensure_directory, safe_file_write, safe_file_read,
    save_json, load_json, save_pickle, load_pickle, save_csv, load_csv,
    get_temp_path, get_cache_path, clean_temp_files
)

# Importa le utilità asincrone
from .async_utils import (
    AsyncResult, Task, ThreadPool, ProcessPool,
    default_thread_pool, get_process_pool,
    run_in_thread, run_in_process, Timer, time_it,
    RateLimiter, PeriodicTask, retry
)

__all__ = [
    # Da logger.py
    'LoggerFactory', 'LogColors', 'ColorFormatter',
    'app_logger', 'data_logger', 'model_logger', 'error_logger',
    'log_exception', 'configure_root_logger', 'silence_loggers',
    
    # Da exceptions.py
    'TradingLabException', 'ConfigError', 'DataError', 'ModelError', 'BacktestError', 'APIError', 'GUIError', 'AsyncError',
    'PathNotFoundError', 'DownloadError', 'ProcessingError', 'ValidationError', 'DatabaseError',
    'InvalidSymbolError', 'InvalidTimeframeError', 'ModelNotFoundError', 'TrainingError',
    'InferenceError', 'EvaluationError', 'SimulationError', 'StrategyError',
    'RateLimitError', 'AuthenticationError', 'ChartError', 'TimeoutError',
    'handle_exception',
    
    # Da file_manager.py
    'FileError', 'ensure_directory', 'safe_file_write', 'safe_file_read',
    'save_json', 'load_json', 'save_pickle', 'load_pickle', 'save_csv', 'load_csv',
    'get_temp_path', 'get_cache_path', 'clean_temp_files',
    
    # Da async_utils.py
    'AsyncResult', 'Task', 'ThreadPool', 'ProcessPool',
    'default_thread_pool', 'get_process_pool',
    'run_in_thread', 'run_in_process', 'Timer', 'time_it',
    'RateLimiter', 'PeriodicTask', 'retry'
]