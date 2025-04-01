"""
Modulo per la gestione dei dati nel progetto TradingLab.
Questo modulo fornisce funzionalità per il download, l'elaborazione, 
la validazione e la memorizzazione dei dati finanziari.
"""

# Importazioni dal modulo validators
from .validators import (
    DataValidator, MarketDataValidator, DataFrameValidator, YahooFinanceValidator,
    is_valid_symbol, is_valid_timeframe, validate_date_str
)

# Importazioni dal modulo database
from .database import (
    DatabaseManager, SQLiteManager, MySQLManager, 
    MarketDataRepository, DatabaseFactory,
    get_default_repository
)

# Importazioni dal modulo downloader
from .downloader import (
    DataDownloader, YahooFinanceDownloader, BatchDownloader,
    HistoricalDataUpdater, RealtimeDataService, DataScheduler,
    create_downloader, get_batch_downloader, get_updater,
    get_realtime_service, get_scheduler
)

# Importazioni dal modulo processor
from .processor import (
    TechnicalIndicators, MarketPatterns, DataProcessor,
    MarketDataBatcher, get_processor, get_batcher
)

# Importazioni dal modulo file_storage
from .file_storage import (
    MarketDataStorage, DataPersistenceManager,
    get_file_storage, get_persistence_manager
)

__all__ = [
    # Da validators.py
    'DataValidator', 'MarketDataValidator', 'DataFrameValidator', 'YahooFinanceValidator',
    'is_valid_symbol', 'is_valid_timeframe', 'validate_date_str',
    
    # Da database.py
    'DatabaseManager', 'SQLiteManager', 'MySQLManager', 
    'MarketDataRepository', 'DatabaseFactory',
    'get_default_repository',
    
    # Da downloader.py
    'DataDownloader', 'YahooFinanceDownloader', 'BatchDownloader',
    'HistoricalDataUpdater', 'RealtimeDataService', 'DataScheduler',
    'create_downloader', 'get_batch_downloader', 'get_updater',
    'get_realtime_service', 'get_scheduler',
    
    # Da processor.py
    'TechnicalIndicators', 'MarketPatterns', 'DataProcessor',
    'MarketDataBatcher', 'get_processor', 'get_batcher',
    
    # Da file_storage.py
    'MarketDataStorage', 'DataPersistenceManager',
    'get_file_storage', 'get_persistence_manager'
]