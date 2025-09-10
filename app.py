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
    # パラメータ設定の辞書
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

    # --- 全ティッカーのサマリー表示機能 ---
    st.header("📈 全ティッカーの市場状況サマリー")
    summary_data = []
    with st.spinner("全ティッカーの最新データを取得・分析中です..."):
        default_params = PRESETS["スイングトレード"]
        default_strategy = "トレンドフォロー"
        
        end_date_summary = datetime.now().date()
        start_date_summary = end_date_summary - timedelta(days=365)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker_item in enumerate(TICKERS):
            status_text.text(f"分析中: {ticker_item} ({i+1}/{len(TICKERS)})...")
            raw_data = load_data(ticker_item, start_date_summary.isoformat(), end_date_summary.isoformat())
            
            if raw_data is not None and not raw_data.empty:
                data_hash = hash(str(raw_data.values.tobytes()) + str(default_params) + default_strategy)
                data = calculate_indicators_and_signals(data_hash, raw_data, default_params, default_strategy)
                
                if not data.empty:
                    latest = data.iloc[-1]
                    summary_data.append({
                        'ティッカー': ticker_item,
                        '現在価格': latest['close'],
                        '総合シグナル': latest['composite_signal'],
                        'RSI': latest['rsi'],
                        'シグナル強度': int(safe_calculate_signal_strength(latest))
                    })
            progress_bar.progress((i + 1) / len(TICKERS))

        status_text.text(f"全{len(TICKERS)}ティッカーの分析が完了しました。")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(
            summary_df,
            column_config={
                "現在価格": st.column_config.NumberColumn("現在価格", format="$%.2f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "シグナル強度": st.column_config.ProgressColumn(
                    "シグナル強度",
                    help="シグナルの強さを0-100で表示します。",
                    format="%d",
                    min_value=0,
                    max_value=100
                )
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("サマリーデータを取得できませんでした。ティッカーリストが空か、データソースに問題がある可能性があります。")

    st.markdown("---")
    
    # --- 個別ティッカーの詳細分析 ---
    if run_optimization_clicked:
        from optimizer_ui import run_optimization
        run_optimization(ticker, start_date, end_date, st.session_state.preset_choice, strategy_type)
        st.stop()

    st.title(f"🏆 プロフェッショナル株式戦略分析: {ticker}")

    if not validate_date_range(start_date, end_date):
        st.stop()
