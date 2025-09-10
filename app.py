import streamlit as st
import pandas as pd
import logging
from datetime import datetime, timedelta
from utils import load_config, validate_date_range, validate_data_quality, safe_calculate_signal_strength
from data_loader import load_data
from ui_components import setup_sidebar
from backtester import calculate_indicators_and_signals, backtest_strategy, calculate_performance_metrics
from plotting import plot_performance

# --- Page and Logging Configuration ---
st.set_page_config(
    page_title="プロフェッショナル株式戦略分析",
    page_icon="🏆",
    layout="wide"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Application ---
def main():
    config = load_config()
    TICKERS = config.get('tickers', {}).get('default_tickers', ["SOXL", "SOXS", "NVDA", "AMD", "TSM"])
    PRESETS = {
        "スイングトレード": {
            'short_window': 10, 'long_window': 40, 'rsi_period': 10, 'macd_fast': 10, 'macd_slow': 20, 'macd_signal': 7,
            'bb_length': 20, 'bb_std': 2.0, 'stoch_k': 14, 'stoch_d': 3,
            'dev_upper': 10, 'dev_lower': -10, 'rsi_upper': 70, 'rsi_lower': 30, 'stoch_upper': 80, 'stoch_lower': 20
        },
        "長期投資": {
            'short_window': 50, 'long_window': 200, 'rsi_period': 30, 'macd_fast': 30, 'macd_slow': 60, 'macd_signal': 15,
            'bb_length': 20, 'bb_std': 2.0, 'stoch_k': 14, 'stoch_d': 3,
            'dev_upper': 10, 'dev_lower': -10, 'rsi_upper': 70, 'rsi_lower': 30, 'stoch_upper': 80, 'stoch_lower': 20
        }
    }
    params_config = {
        'short_window': ('短期SMA', 5, 50, 20),
        'long_window': ('長期SMA', 55, 200, 50),
        'rsi_period': ('RSI期間', 7, 30, 14),
        'macd_fast': ('MACD短期', 5, 25, 12),
        'macd_slow': ('MACD長期', 26, 50, 26),
        'macd_signal': ('MACDシグナル', 5, 15, 9),
        'bb_length': ('BB期間', 10, 50, 20),
        'bb_std': ('BB標準偏差', 1.5, 3.0, 2.0),
        'stoch_k': ('Stoch %K', 5, 20, 14),
        'stoch_d': ('Stoch %D', 3, 10, 3),
        'dev_upper': ('乖離率 上限', 5, 25, 10),
        'dev_lower': ('乖離率 下限', -25, -5, -10),
        'rsi_upper': ('RSI 上限', 6
