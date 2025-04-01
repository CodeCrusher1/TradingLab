# Sistema di logging

"""
Sistema di logging per il progetto TradingLab.
Questo modulo fornisce funzionalità di logging configurabili per diversi componenti dell'applicazione.
"""
import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Union, Dict, List

# Importa le configurazioni
from ..config import (
    APP_NAME, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT,
    MAIN_LOG_PATH, ERROR_LOG_PATH, DATA_LOG_PATH, MODEL_LOG_PATH
)

# Definizione di colori per il logging nella console
class LogColors:
    """Colori ANSI per il logging nella console."""
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ColorFormatter(logging.Formatter):
    """Formatter personalizzato che aggiunge colori ai log nella console."""
    COLORS = {
        logging.DEBUG: LogColors.BLUE,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD + LogColors.RED,
    }

    def format(self, record):
        """Formatta il record di log con colori per la console."""
        log_message = super().format(record)
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelno, LogColors.RESET)
            return f"{color}{log_message}{LogColors.RESET}"
        return log_message


class LoggerFactory:
    """
    Factory per creare e configurare logger per diversi componenti dell'applicazione.
    """
    _loggers: Dict[str, logging.Logger] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Inizializza il sistema di logging creando le directory necessarie."""
        if cls._initialized:
            return
        
        # Crea directory per i log se non esistono
        for log_path in [MAIN_LOG_PATH, ERROR_LOG_PATH, DATA_LOG_PATH, MODEL_LOG_PATH]:
            log_dir = Path(log_path).parent
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
        
        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str, log_file: Optional[Union[str, Path]] = None,
                  level: int = LOG_LEVEL, rotating: bool = True,
                  max_bytes: int = 10*1024*1024, backup_count: int = 5) -> logging.Logger:
        """
        Ottiene o crea un logger con il nome specificato.
        
        Args:
            name: Nome del logger (es. 'data', 'models', etc.)
            log_file: Percorso opzionale del file di log
            level: Livello di logging (default LOG_LEVEL da config)
            rotating: Se usare RotatingFileHandler (True) o TimedRotatingFileHandler (False)
            max_bytes: Dimensione massima del file prima della rotazione (per RotatingFileHandler)
            backup_count: Numero di backup da mantenere
            
        Returns:
            Logger configurato
        """
        # Se il logger esiste già, restituiscilo
        logger_key = f"{name}_{str(log_file)}"
        if logger_key in cls._loggers:
            return cls._loggers[logger_key]

        # Assicurati che il sistema sia inizializzato
        cls.initialize()
        
        # Crea un nuovo logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Rimuovi handler esistenti per evitare duplicati
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Aggiungi handler per console con colori
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = ColorFormatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Aggiungi file handler se specificato
        if log_file:
            file_path = Path(log_file)
            file_dir = file_path.parent
            
            # Crea la directory se non esiste
            if not file_dir.exists():
                file_dir.mkdir(parents=True, exist_ok=True)
            
            if rotating:
                file_handler = RotatingFileHandler(
                    file_path, maxBytes=max_bytes, backupCount=backup_count
                )
            else:
                file_handler = TimedRotatingFileHandler(
                    file_path, when='midnight', interval=1, backupCount=backup_count
                )
            
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Memorizza il logger
        cls._loggers[logger_key] = logger
        return logger

    @classmethod
    def get_app_logger(cls) -> logging.Logger:
        """Ottiene il logger principale dell'applicazione."""
        return cls.get_logger(APP_NAME, MAIN_LOG_PATH)
    
    @classmethod
    def get_data_logger(cls) -> logging.Logger:
        """Ottiene il logger per operazioni sui dati."""
        return cls.get_logger("data", DATA_LOG_PATH)
    
    @classmethod
    def get_model_logger(cls) -> logging.Logger:
        """Ottiene il logger per operazioni sui modelli."""
        return cls.get_logger("model", MODEL_LOG_PATH)
    
    @classmethod
    def get_error_logger(cls) -> logging.Logger:
        """Ottiene il logger per errori."""
        return cls.get_logger("error", ERROR_LOG_PATH, level=logging.ERROR)


# Inizializza il sistema di logging
LoggerFactory.initialize()

# Esporta i logger principali
app_logger = LoggerFactory.get_app_logger()
data_logger = LoggerFactory.get_data_logger()
model_logger = LoggerFactory.get_model_logger()
error_logger = LoggerFactory.get_error_logger()


def log_exception(logger: logging.Logger, exception: Exception, message: str = "Si è verificato un errore:") -> None:
    """
    Registra un'eccezione con messaggio personalizzato e traceback completo.
    
    Args:
        logger: Logger da utilizzare
        exception: Eccezione da registrare
        message: Messaggio personalizzato
    """
    import traceback
    error_details = ''.join(traceback.format_exception(
        type(exception), exception, exception.__traceback__))
    logger.error(f"{message} {str(exception)}\n{error_details}")


def configure_root_logger(level: int = LOG_LEVEL) -> None:
    """
    Configura il logger root per catturare tutti i messaggi di logging.
    
    Args:
        level: Livello di logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Rimuovi handler esistenti
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Aggiungi handler per console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = ColorFormatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Aggiungi handler per file di errore
    error_handler = RotatingFileHandler(
        ERROR_LOG_PATH, maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)


def silence_loggers(modules: List[str], level: int = logging.WARNING) -> None:
    """
    Silenzia logger di moduli esterni impostando un livello più alto.
    
    Args:
        modules: Lista di nomi di moduli da silenziare
        level: Livello di logging minimo da mostrare
    """
    for module in modules:
        logging.getLogger(module).setLevel(level)


# Silenzia alcuni logger esterni comuni
external_modules = [
    'matplotlib', 'urllib3', 'tensorflow', 'keras', 
    'PIL', 'requests', 'yfinance', 'pandas', 'numpy'
]
silence_loggers(external_modules)