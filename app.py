import streamlit as st
import pandas as pd
import logging
import hashlib
from utils import load_config, validate_date_range, validate_data_quality
from data_loader import load_data
from ui_components import setup_sidebar
from backtester import calculate_indicators_and_signals, backtest_strategy, calculate_performance_metrics, calculate_trade_metrics, resolve_ticker_class
from plotting import plot_performance
import swing_ui

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
            'short_window': 10, 'long_window': 50, 'rsi_period': 10, 'macd_fast': 10, 'macd_slow': 20, 'macd_signal': 7,
            'bb_length': 20, 'bb_std': 2.0, 'stoch_k': 14, 'stoch_d': 3,
            'dev_upper': 10, 'dev_lower': -10, 'rsi_upper': 70, 'rsi_lower': 30, 'stoch_upper': 80, 'stoch_lower': 20,
            'adx_threshold': 18, 'score_smooth_period': 3, 'ema_slope_period': 5,
            'pyramid_threshold': 0.06
        },
        "長期投資": {
            'short_window': 40, 'long_window': 200, 'rsi_period': 30, 'macd_fast': 30, 'macd_slow': 60, 'macd_signal': 15,
            'bb_length': 20, 'bb_std': 2.0, 'stoch_k': 14, 'stoch_d': 3,
            'dev_upper': 10, 'dev_lower': -10, 'rsi_upper': 70, 'rsi_lower': 30, 'stoch_upper': 80, 'stoch_lower': 20,
            'adx_threshold': 22, 'score_smooth_period': 5, 'ema_slope_period': 8,
            'pyramid_threshold': 0.08
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
        'adx_threshold': ('ADX閾値', 15, 35, 20),
        'score_smooth_period': ('スコア平滑化期間', 2, 7, 3),
        'ema_slope_period': ('EMAスロープ期間', 3, 13, 5),
        'dev_upper': ('乖離率 上限', 5, 25, 10),
        'dev_lower': ('乖離率 下限', -25, -5, -10),
        'rsi_upper': ('RSI 上限', 60, 80, 70),
        'rsi_lower': ('RSI 下限', 20, 40, 30),
        'stoch_upper': ('Stoch 上限', 70, 90, 80),
        'stoch_lower': ('Stoch 下限', 10, 30, 20)
    }

    ticker, start_date, end_date, params, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params, strategy_type, run_optimization_clicked = setup_sidebar(TICKERS, PRESETS, params_config)

    # 商品クラス (config.toml [ticker_classes]) をシグナル計算に渡す
    params = {**params, 'ticker_class': resolve_ticker_class(ticker, config)}

    # --- スイングトレード用の画面 (候補 / 保有) と、従来の個別分析をタブで分ける ---
    swing_settings = swing_ui.settings_inputs(config)
    prices, frames, failed = swing_ui.load_universe(swing_ui.universe(config), config)

    tab_candidates, tab_positions, tab_detail = st.tabs(
        ["🎯 今日のスイング候補", "💼 保有ポジション", "🔬 個別分析・バックテスト"])
    with tab_candidates:
        swing_ui.render_candidates(config, swing_settings, prices, frames, failed)
    with tab_positions:
        swing_ui.render_positions(config, swing_settings, prices)
    with tab_detail:
        render_detail(config, swing_settings, ticker, start_date, end_date, params, initial_capital, commission_rate,
                      slippage, position_sizing_strategy, ps_params, strategy_type, run_optimization_clicked)

    st.markdown("---")
    st.markdown("このツールは教育および情報提供目的のものです。投資判断はご自身の責任で行ってください。")


def render_detail(config, swing_settings, ticker, start_date, end_date, params, initial_capital, commission_rate,
                  slippage, position_sizing_strategy, ps_params, strategy_type, run_optimization_clicked):
    """1 銘柄の詳細分析 (チャート・バックテスト・最適化)。最後のタブなので st.stop() で打ち切ってよい。"""
    # --- 個別ティッカーの詳細分析 ---
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
        st.error(f"❌ ティッカー {ticker} のデータが見つかりません。期間やティッカーシンボルを確認してください。")
        st.stop()

    if quality_issues := validate_data_quality(raw_data):
        with st.expander("⚠️ データ品質の警告", expanded=True):
            for issue in quality_issues: 
                st.warning(issue)

    with st.spinner('テクニカル指標を計算中...'):
        data_hash = hashlib.sha256((str(raw_data.values.tobytes()) + str(params) + strategy_type).encode()).hexdigest()
        data = calculate_indicators_and_signals(data_hash, raw_data, params, strategy_type)

    if data.empty: 
        st.error("分析可能なデータがありません。期間やパラメータを調整してください。")
        st.stop()

    with st.spinner('バックテストを実行中...'):
        pyramid_thr = float(params.get('pyramid_threshold', 0.10))
        results_hash = hashlib.sha256((str(data.values.tobytes()) + str(initial_capital) + str(commission_rate) + str(slippage) + position_sizing_strategy + str(ps_params) + strategy_type + str(pyramid_thr)).encode()).hexdigest()
        results = backtest_strategy(results_hash, data, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params, strategy_type, pyramid_thr)
        metrics = calculate_performance_metrics(results['portfolio_values'], results['dates'])

    if metrics:
        plot_performance(data, metrics, results, params)
        st.caption(
            "このバックテストは、シグナル当日の終値で約定し、利確・トレイリングなど独自の手仕舞いルールで運用した場合です。"
            "トレンドフォロー / 逆張りでは表示期間そのものでパラメータを自動調整しているため、成績は実力より良く見えます。"
            "スイング候補タブの売買プランどおりに売買した場合の成績は、このページの一番下にあります。"
        )

    if results and 'trades' in results and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        st.subheader("取引詳細分析")
        trade_metrics = calculate_trade_metrics(trades_df)
        if trade_metrics:
            st.markdown("##### 保有期間")
            col1, col2, col3 = st.columns(3)
            col1.metric("平均保有日数", f"{trade_metrics['avg_hold_days']:.1f}日")
            col2.metric("最長保有日数", f"{trade_metrics['max_hold_days']}日")
            col3.metric("最短保有日数", f"{trade_metrics['min_hold_days']}日")

            if 'total_trades' in trade_metrics:
                st.markdown("##### 勝敗統計")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("総取引数", f"{trade_metrics['total_trades']}")
                col2.metric("勝率", f"{trade_metrics['win_rate']:.1f}%")
                col3.metric("プロフィットファクター", f"{trade_metrics['profit_factor']:.2f}")
                pf = trade_metrics['profit_factor']
                if pf >= 2.0:
                    col4.success("優秀")
                elif pf >= 1.5:
                    col4.info("良好")
                elif pf >= 1.0:
                    col4.warning("改善余地あり")
                else:
                    col4.error("損失超過")

                st.markdown("##### 損益詳細")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("平均利益", f"{trade_metrics['avg_win']:.2f}%")
                col2.metric("平均損失", f"{trade_metrics['avg_loss']:.2f}%")
                col3.metric("最大連勝", f"{trade_metrics['max_win_streak']}")
                col4.metric("最大連敗", f"{trade_metrics['max_loss_streak']}")

                st.markdown("##### ストップ発動")
                col1, col2 = st.columns(2)
                col1.metric("損切り発動回数", f"{trade_metrics['stop_loss_count']}")
                col2.metric("トレーリングストップ発動回数", f"{trade_metrics['trailing_stop_count']}")

        # #7 Exit理由別分析
        if trade_metrics and 'exit_breakdown' in trade_metrics and trade_metrics['exit_breakdown']:
            st.markdown("##### Exit理由別分析")
            _reason_labels = {
                'stop': '損切り', 'trail': 'トレーリング', 'signal': 'シグナル',
                'target': '目標到達', 'time': '時間切れ', 'partial': '部分利確', 'structure': '構造トレイル'
            }
            breakdown_rows = []
            for reason, stats in trade_metrics['exit_breakdown'].items():
                breakdown_rows.append({
                    'Exit理由': _reason_labels.get(reason, reason),
                    '回数': stats['count'],
                    '勝率(%)': stats['win_rate'],
                    '平均損益(%)': stats['avg_pnl_pct'],
                    '平均保有日数': stats['avg_hold_days'],
                })
            st.dataframe(pd.DataFrame(breakdown_rows), hide_index=True, use_container_width=True)

        with st.expander("取引履歴を表示"):
            st.dataframe(trades_df)

    st.markdown("---")
    swing_ui.render_plan_check(raw_data, data, swing_settings, ticker, strategy_type)


if __name__ == "__main__":
    main()
