# Configurazioni globali

"""
Global configuration settings for the TradingLab project.
This module contains general settings, database configuration, and other global parameters.
"""
import os
import logging
from datetime import datetime

# Application version information
APP_NAME = "TradingLab"
APP_VERSION = "1.0.0"
APP_AUTHOR = "TradingLab Team"

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Database configuration
DB_CONFIG = {
    "mysql": {
        "host": "cryptodatabase.cadi6mm060xh.us-east-1.rds.amazonaws.com",
        "port": 3306,
        "database": "CryptoDatabaseV2",
        "user": "admin",
        "password": "CryptoHunterV2"
    },
    "sqlite": {
        "database": "tradinglab.db"
    }
}

# Default database type
DEFAULT_DB_TYPE = "sqlite"  # Options: "mysql", "sqlite"

# Data processing settings
DEFAULT_DATA_LIMIT = 10000  # Maximum records to retrieve from database
DEFAULT_LOOKBACK = 30  # Default lookback period for feature generation
PREDICTION_HORIZON = 5  # Default prediction horizon

# Training settings
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_TEST_SPLIT = 0.15  # 15% for test data
DEFAULT_VALIDATION_SPLIT = 0.2  # 20% of training data for validation

# API settings
API_RATE_LIMIT = 2000  # Maximum API calls per day
API_TIMEOUT = 30  # API timeout in seconds

# GUI settings
DEFAULT_THEME = "light"  # Options: "light", "dark"
DEFAULT_CHART_HEIGHT = 500
DEFAULT_CANDLESTICK_COUNT = 100
REFRESH_INTERVAL = 60  # Auto-refresh interval in seconds

# Miscellaneous
DEBUG_MODE = True
ENABLE_TELEMETRY = False  # Whether to collect anonymous usage statistics
CACHE_ENABLED = True  # Enable data caching
CACHE_EXPIRY = 3600  # Cache expiry in seconds

# Technical indicator default parameters
INDICATOR_PARAMS = {
    "ema": [21, 50, 200],
    "rsi": [14, 21, 50],
    "atr": [14, 21, 50],
    "stochastic": [14],
    "fair_value_gap": 3,  # lookback periods
    "order_block": 5,     # lookback periods
    "liquidity_level": 5  # lookback periods
}