"""
Modulo per l'analisi tecnica e statistica nel progetto TradingLab.
Questo modulo fornisce strumenti avanzati per l'analisi di mercato, indicatori tecnici, 
riconoscimento di pattern e analisi statistica dei dati finanziari.
"""

# Importazioni dalle classi di indicatori avanzati
from .indicators import (
    TrendIndicators, MomentumIndicators, VolumeIndicators, 
    PivotIndicators, MarketStructureIndicators, CustomIndicators,
    get_indicators
)

# Importazioni dalle classi di pattern
from .patterns import (
    CandlePatterns, HarmonicPatterns, ChartPatterns, PricePatterns,
    get_pattern_recognizers
)

# Importazioni dalle classi di analisi statistica
from .statistics import (
    DescriptiveStats, TimeSeriesAnalysis, MarketEfficiencyAnalysis, EventStudyAnalysis,
    get_statistical_analyzers
)

__all__ = [
    # Da indicators.py
    'TrendIndicators', 'MomentumIndicators', 'VolumeIndicators',
    'PivotIndicators', 'MarketStructureIndicators', 'CustomIndicators',
    'get_indicators',
    
    # Da patterns.py
    'CandlePatterns', 'HarmonicPatterns', 'ChartPatterns', 'PricePatterns',
    'get_pattern_recognizers',
    
    # Da statistics.py
    'DescriptiveStats', 'TimeSeriesAnalysis', 'MarketEfficiencyAnalysis', 'EventStudyAnalysis',
    'get_statistical_analyzers'
]