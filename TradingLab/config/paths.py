# Percorsi di file e directory

"""
File and directory paths for the TradingLab project.
This module manages all file paths and ensures required directories exist.
"""
import os
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Determine the base directory
if getattr(sys, 'frozen', False):
    # If the application is frozen (compiled)
    BASE_DIR = Path(os.path.dirname(sys.executable))
else:
    # If running from script
    BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory paths
STORAGE_DIR = BASE_DIR / "storage"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = STORAGE_DIR / "models"
DATA_DIR = STORAGE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = STORAGE_DIR / "database"
REPORTS_DIR = STORAGE_DIR / "reports"
LOGS_DIR = STORAGE_DIR / "logs"
CACHE_DIR = STORAGE_DIR / "cache"
TEMP_DIR = STORAGE_DIR / "temp"
EXPORTS_DIR = STORAGE_DIR / "exports"

# Model-specific directories
STANDARD_MODEL_DIR = MODELS_DIR / "standard"
ADVANCED_MODEL_DIR = MODELS_DIR / "advanced"
SCALER_DIR = MODELS_DIR / "scalers"
MODEL_REPORTS_DIR = REPORTS_DIR / "models"
BACKTEST_REPORTS_DIR = REPORTS_DIR / "backtest"

# Database file paths
SQLITE_DB_PATH = DATABASE_DIR / "tradinglab.db"

# Log file paths
MAIN_LOG_PATH = LOGS_DIR / "tradinglab.log"
ERROR_LOG_PATH = LOGS_DIR / "errors.log"
DATA_LOG_PATH = LOGS_DIR / "data.log"
MODEL_LOG_PATH = LOGS_DIR / "models.log"

# Model file name formats
MODEL_FILENAME_FORMAT = "{model_type}_{symbol}_{timeframe}_{timestamp}.keras"
LEGACY_MODEL_FILENAME_FORMAT = "{model_type}_{symbol}_{timeframe}_{timestamp}.h5"
LATEST_MODEL_FILENAME_FORMAT = "{model_type}_{symbol}_{timeframe}_latest.keras"

# Export formats
CSV_EXPORT_FORMAT = "{symbol}_{timeframe}_{date}.csv"
REPORT_FORMAT = "{report_type}_{symbol}_{timeframe}_{timestamp}.{extension}"

# List of directories to create on startup
DIRS_TO_CREATE = [
    STORAGE_DIR,
    MODELS_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATABASE_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    CACHE_DIR,
    TEMP_DIR,
    EXPORTS_DIR,
    STANDARD_MODEL_DIR,
    ADVANCED_MODEL_DIR,
    SCALER_DIR,
    MODEL_REPORTS_DIR,
    BACKTEST_REPORTS_DIR
]


def create_directories():
    """Create all required directories if they don't exist."""
    for directory in DIRS_TO_CREATE:
        try:
            if not directory.exists():
                directory.mkdir(parents=True)
                logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")


def get_model_path(model_type, symbol, timeframe, timestamp=None, latest=False):
    """
    Get the path for a model file.
    
    Args:
        model_type: Type of model (e.g., 'ta_lstm', 'transformer')
        symbol: Symbol name
        timeframe: Timeframe name
        timestamp: Optional timestamp for versioning
        latest: Whether to get the path for the latest model
        
    Returns:
        Path object for the model file
    """
    if latest:
        filename = LATEST_MODEL_FILENAME_FORMAT.format(
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe
        )
    else:
        from datetime import datetime
        timestamp = timestamp or datetime.now().strftime('%Y%m%d')
        filename = MODEL_FILENAME_FORMAT.format(
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=timestamp
        )
    
    if model_type in ['ta_lstm', 'transformer', 'ensemble']:
        return ADVANCED_MODEL_DIR / filename
    else:
        return STANDARD_MODEL_DIR / filename


def get_report_path(report_type, symbol, timeframe, extension='txt'):
    """
    Get the path for a report file.
    
    Args:
        report_type: Type of report (e.g., 'evaluation', 'backtest')
        symbol: Symbol name
        timeframe: Timeframe name
        extension: File extension
        
    Returns:
        Path object for the report file
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d')
    
    filename = REPORT_FORMAT.format(
        report_type=report_type,
        symbol=symbol,
        timeframe=timeframe,
        timestamp=timestamp,
        extension=extension
    )
    
    if report_type.startswith('backtest'):
        return BACKTEST_REPORTS_DIR / filename
    else:
        return MODEL_REPORTS_DIR / filename


def get_data_path(symbol, timeframe, processed=False):
    """
    Get the path for a data file.
    
    Args:
        symbol: Symbol name
        timeframe: Timeframe name
        processed: Whether the data is processed or raw
        
    Returns:
        Path object for the data file
    """
    filename = f"{symbol}_{timeframe}.csv"
    
    if processed:
        return PROCESSED_DATA_DIR / filename
    else:
        return RAW_DATA_DIR / filename


# Create directories on module import
create_directories()