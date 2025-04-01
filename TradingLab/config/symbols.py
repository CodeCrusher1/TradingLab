# Definizione simboli e timeframes

"""
Definition of trading symbols and timeframes.
This module provides structured access to available trading instruments and their properties.
"""
from enum import Enum
from typing import Dict, List, Optional, Union, NamedTuple

class AssetClass(Enum):
    """Classification of financial assets."""
    PRECIOUS_METAL = "Precious Metal"
    INDUSTRIAL_METAL = "Industrial Metal"
    ENERGY = "Energy"
    INDEX = "Index"
    CRYPTOCURRENCY = "Cryptocurrency"
    FOREX = "Forex"


class Symbol(NamedTuple):
    """Trading symbol information."""
    name: str
    ticker: str
    asset_class: AssetClass
    description: str = ""
    active: bool = True
    decimal_places: int = 2


class Timeframe(NamedTuple):
    """Trading timeframe information."""
    name: str
    code: str
    description: str = ""
    minutes: int = 0  # Minutes per candle


# Define available timeframes
TIMEFRAMES = {
    "daily": Timeframe(
        name="daily", 
        code="1d", 
        description="Daily timeframe", 
        minutes=1440
    ),
    "weekly": Timeframe(
        name="weekly", 
        code="1wk", 
        description="Weekly timeframe", 
        minutes=10080
    ),
    "monthly": Timeframe(
        name="monthly", 
        code="1mo", 
        description="Monthly timeframe", 
        minutes=43200
    ),
    "hourly": Timeframe(
        name="hourly", 
        code="1h", 
        description="Hourly timeframe", 
        minutes=60
    ),
    "15min": Timeframe(
        name="15min", 
        code="15m", 
        description="15-minute timeframe", 
        minutes=15
    )
}


# Define available symbols
SYMBOLS = {
    "Gold": Symbol(
        name="Gold", 
        ticker="GC=F", 
        asset_class=AssetClass.PRECIOUS_METAL, 
        description="Gold Futures",
        decimal_places=2
    ),
    "Silver": Symbol(
        name="Silver", 
        ticker="SI=F", 
        asset_class=AssetClass.PRECIOUS_METAL, 
        description="Silver Futures",
        decimal_places=3
    ),
    "Copper": Symbol(
        name="Copper", 
        ticker="HG=F", 
        asset_class=AssetClass.INDUSTRIAL_METAL, 
        description="Copper Futures",
        decimal_places=3
    ),
    "Platinum": Symbol(
        name="Platinum", 
        ticker="PL=F", 
        asset_class=AssetClass.PRECIOUS_METAL, 
        description="Platinum Futures",
        decimal_places=2
    ),
    "Palladium": Symbol(
        name="Palladium", 
        ticker="PA=F", 
        asset_class=AssetClass.PRECIOUS_METAL, 
        description="Palladium Futures",
        decimal_places=2
    ),
    "Crude Oil": Symbol(
        name="Crude Oil", 
        ticker="CL=F", 
        asset_class=AssetClass.ENERGY, 
        description="Crude Oil Futures",
        decimal_places=2
    ),
    "Natural Gas": Symbol(
        name="Natural Gas", 
        ticker="NG=F", 
        asset_class=AssetClass.ENERGY, 
        description="Natural Gas Futures",
        decimal_places=3
    ),
    "S&P 500": Symbol(
        name="S&P 500", 
        ticker="ES=F", 
        asset_class=AssetClass.INDEX, 
        description="E-mini S&P 500 Futures",
        decimal_places=2
    ),
    "Dow Jones": Symbol(
        name="Dow Jones", 
        ticker="YM=F", 
        asset_class=AssetClass.INDEX, 
        description="E-mini Dow Jones Futures",
        decimal_places=0
    ),
    "Nasdaq 100": Symbol(
        name="Nasdaq 100", 
        ticker="NQ=F", 
        asset_class=AssetClass.INDEX, 
        description="E-mini Nasdaq 100 Futures",
        decimal_places=2
    ),
    "Bitcoin": Symbol(
        name="Bitcoin", 
        ticker="BTC-USD", 
        asset_class=AssetClass.CRYPTOCURRENCY, 
        description="Bitcoin USD",
        decimal_places=2
    ),
    "Ethereum": Symbol(
        name="Ethereum", 
        ticker="ETH-USD", 
        asset_class=AssetClass.CRYPTOCURRENCY, 
        description="Ethereum USD",
        decimal_places=2
    )
}


# Function to get symbol by name
def get_symbol(name: str) -> Optional[Symbol]:
    """
    Get a symbol by its name.
    
    Args:
        name: Name of the symbol
        
    Returns:
        Symbol object if found, None otherwise
    """
    return SYMBOLS.get(name)


# Function to get timeframe by name
def get_timeframe(name: str) -> Optional[Timeframe]:
    """
    Get a timeframe by its name.
    
    Args:
        name: Name of the timeframe
        
    Returns:
        Timeframe object if found, None otherwise
    """
    return TIMEFRAMES.get(name)


# Function to get symbols by asset class
def get_symbols_by_asset_class(asset_class: AssetClass) -> List[Symbol]:
    """
    Get all symbols belonging to a specific asset class.
    
    Args:
        asset_class: Asset class to filter by
        
    Returns:
        List of Symbol objects matching the asset class
    """
    return [symbol for symbol in SYMBOLS.values() if symbol.asset_class == asset_class]


# Default symbols and timeframes for different use cases
DEFAULT_SYMBOLS = ["Gold", "Silver"]
DEFAULT_TIMEFRAMES = ["daily", "hourly", "15min"]
DEFAULT_BACKTEST_TIMEFRAME = "daily"
DEFAULT_PREDICTION_TIMEFRAME = "daily"