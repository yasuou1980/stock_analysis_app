import streamlit as st
import logging
from utils import load_config, validate_date_range, validate_data_quality
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
        'rsi_upper': ('RSI 上限', 60, 80, 70),
        'rsi_lower': ('RSI 下限', 20, 40, 30),
        'stoch_upper': ('Stoch 上限', 70, 90, 80),
        'stoch_lower': ('Stoch 下限', 10, 30, 20)
    }

    ticker, start_date, end_date, params, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params, strategy_type, run_optimization_clicked = setup_sidebar(TICKERS, PRESETS, params_config)

    if run_optimization_clicked:
        from optimizer_ui import run_optimization
        run_optimization(ticker, start_date, end_date, st.session_state.preset_choice, strategy_type)
        st.stop()

    st.title(f"🏆 プロフェッショナル株式戦略分析: {ticker}")

    if not validate_date_range(start_date, end_date): 
        st.stop()

    with st.spinner(f'{ticker} のデータを取得中...'):
        raw_data = load_data(ticker, start_date.isoformat(), end_date.isoformat())

    if raw_data is None: 
        st.stop()

    if quality_issues := validate_data_quality(raw_data):
        with st.expander("⚠️ データ品質の警告", expanded=True):
            for issue in quality_issues: 
                st.warning(issue)

    with st.spinner('テクニカル指標を計算中...'):
        data_hash = hash(str(raw_data.values.tobytes()) + str(params) + strategy_type)
        data = calculate_indicators_and_signals(data_hash, raw_data, params, strategy_type)

    if data.empty: 
        st.error("分析可能なデータがありません。期間やパラメータを調整してください。")
        st.stop()

    with st.spinner('バックテストを実行中...'):
        results_hash = hash(str(data.values.tobytes()) + str(initial_capital) + str(commission_rate) + str(slippage) + position_sizing_strategy + str(ps_params))
        results = backtest_strategy(results_hash, data, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params)
        metrics = calculate_performance_metrics(results['portfolio_values'], results['dates'])

    if metrics:
        plot_
