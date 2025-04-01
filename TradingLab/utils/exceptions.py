# Gestione errori

"""
Gestione personalizzata delle eccezioni per il progetto TradingLab.
Questo modulo definisce eccezioni specifiche per diversi componenti dell'applicazione.
"""
from typing import Optional, Any, Dict


class TradingLabException(Exception):
    """Classe base per tutte le eccezioni di TradingLab."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Inizializza l'eccezione con messaggio e dettagli opzionali.
        
        Args:
            message: Messaggio di errore
            details: Dettagli aggiuntivi sull'errore (opzionale)
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def __str__(self) -> str:
        """Formatta il messaggio di errore con dettagli."""
        if not self.details:
            return self.message
        
        details_str = ', '.join(f"{k}={v}" for k, v in self.details.items())
        return f"{self.message} | Dettagli: {details_str}"


# Eccezioni relative alla configurazione
class ConfigError(TradingLabException):
    """Errore relativo alla configurazione."""
    pass


class PathNotFoundError(ConfigError):
    """Errore quando un percorso necessario non è trovato."""
    pass


# Eccezioni relative ai dati
class DataError(TradingLabException):
    """Errore relativo ai dati."""
    pass


class DownloadError(DataError):
    """Errore durante il download dei dati."""
    pass


class ProcessingError(DataError):
    """Errore durante l'elaborazione dei dati."""
    pass


class ValidationError(DataError):
    """Errore durante la validazione dei dati."""
    pass


class DatabaseError(DataError):
    """Errore relativo al database."""
    pass


class InvalidSymbolError(DataError):
    """Errore per simbolo non valido."""
    pass


class InvalidTimeframeError(DataError):
    """Errore per timeframe non valido."""
    pass


# Eccezioni relative ai modelli
class ModelError(TradingLabException):
    """Errore relativo ai modelli."""
    pass


class ModelNotFoundError(ModelError):
    """Errore quando un modello non è trovato."""
    pass


class TrainingError(ModelError):
    """Errore durante l'addestramento del modello."""
    pass


class InferenceError(ModelError):
    """Errore durante l'inferenza del modello."""
    pass


class EvaluationError(ModelError):
    """Errore durante la valutazione del modello."""
    pass


# Eccezioni relative al backtest
class BacktestError(TradingLabException):
    """Errore relativo al backtest."""
    pass


class SimulationError(BacktestError):
    """Errore durante la simulazione."""
    pass


class StrategyError(BacktestError):
    """Errore relativo alla strategia di trading."""
    pass


# Eccezioni relative all'API
class APIError(TradingLabException):
    """Errore relativo all'API."""
    pass


class RateLimitError(APIError):
    """Errore per limite di richieste API superato."""
    pass


class AuthenticationError(APIError):
    """Errore di autenticazione API."""
    pass


# Eccezioni relative all'interfaccia utente
class GUIError(TradingLabException):
    """Errore relativo all'interfaccia utente."""
    pass


class ChartError(GUIError):
    """Errore durante la creazione dei grafici."""
    pass


# Eccezioni relative alle operazioni asincrone
class AsyncError(TradingLabException):
    """Errore durante operazioni asincrone."""
    pass


class TimeoutError(AsyncError):
    """Errore di timeout durante operazioni asincrone."""
    pass


# Funzioni di utilità per le eccezioni
def handle_exception(exc: Exception, raise_exception: bool = True, 
                    default_return: Any = None, logger=None) -> Any:
    """
    Gestisce un'eccezione registrandola e, opzionalmente, rilanciandola.
    
    Args:
        exc: L'eccezione da gestire
        raise_exception: Se True, rilancia l'eccezione
        default_return: Valore da restituire se non rilancia l'eccezione
        logger: Logger da utilizzare, se None usa error_logger
        
    Returns:
        default_return se non rilancia l'eccezione
        
    Raises:
        exc: Se raise_exception è True
    """
    # Importa qui per evitare import circolari
    from .logger import error_logger, log_exception
    
    # Usa il logger fornito o quello predefinito
    log = logger or error_logger
    
    # Registra l'eccezione
    log_exception(log, exc)
    
    # Rilancia l'eccezione se richiesto
    if raise_exception:
        raise exc
    
    return default_return