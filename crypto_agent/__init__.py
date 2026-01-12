"""
Crypto Trading Agent Package
Intelligent trading decision system powered by ML predictions
"""

from .agent import CryptoTradingAgent
from .database import AgentDatabase
from .tools import (
    SmartConsensusAnalyzer,
    TradeQualityScorer,
    MarketContextAnalyzer,
    PredictionOutcomeTracker
)

__version__ = '1.0.0'
__all__ = [
    'CryptoTradingAgent',
    'AgentDatabase',
    'SmartConsensusAnalyzer',
    'TradeQualityScorer',
    'MarketContextAnalyzer',
    'PredictionOutcomeTracker'
]