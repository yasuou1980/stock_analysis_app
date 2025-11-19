import streamlit as st
import pandas as pd
from data_loader import load_data
from backtester import calculate_indicators_and_signals, backtest_strategy, calculate_performance_metrics
from datetime import datetime, timedelta
import itertools

def evaluate_performance(params, raw_data, strategy_type):
    initial_capital = 10000
    commission_rate = 0.001
    slippage = 0.0005
    position_sizing_strategy = "固定リスク率 (Volatility Adjusted)"
    ps_params = {'target_risk': 0.02, 'max_position_ratio': 0.9}

    data_hash = hash(str(raw_data.values.tobytes()) + str(params) + strategy_type)
    data = calculate_indicators_and_signals(data_hash, raw_data, params, strategy_type)
    if data.empty: return None

    results_hash = hash(str(data.values.tobytes()) + str(initial_capital) + str(commission_rate) + str(slippage) + position_sizing_strategy + str(ps_params))
    results = backtest_strategy(results_hash, data, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params)
    metrics = calculate_performance_metrics(results['portfolio_values'], results['dates'])
    
    return metrics

def get_param_grid(preset_choice, strategy_type):
    if strategy_type == "トレンドフォロー":
        if preset_choice == "スイングトレード":
            return {
                'short_window': [10, 15, 20],
                'long_window': [40, 50, 60],
                'rsi_period': [10, 14, 18],
                'macd_fast': [10, 12, 15],
                'macd_slow': [20, 26, 30],
                'macd_signal': [7, 9, 10],
            }
        else: # 長期投資
            return {
                'short_window': [40, 50, 60],
                'long_window': [180, 200, 220],
                'rsi_period': [20, 25, 30],
                'macd_fast': [20, 25, 30],
                'macd_slow': [40, 50, 60],
                'macd_signal': [10, 15, 20],
            }
    else: # 逆張り (範囲を拡大・調整)
        return {
            'bb_length': [20, 25],              # 2 options
            'bb_std': [2.0, 2.5, 3.0],          # 3 options (3.0を追加: 極端な乖離のみ狙う)
            'stoch_k': [14],                    # 1 option (fixed)
            'stoch_d': [3],                     # 1 option (fixed)
            'dev_upper': [10, 15, 20],          # 3 options
            'dev_lower': [-20, -15, -10],       # 3 options
            'rsi_upper': [70, 75, 80, 85],      # 4 options (より広い範囲を探索)
            'rsi_lower': [15, 20, 25, 30],      # 4 options (より広い範囲を探索)
            'stoch_upper': [80],                # 1 option (fixed)
            'stoch_lower': [20]                 # 1 option (fixed)
        }
        # Total combinations approx: 2*3*1*1*3*3*4*4*1*1 = 864 patterns

def run_optimization(ticker, start_date, end_date, preset_choice, strategy_type):
    st.subheader(f"⚙️ {strategy_type}戦略の最適化を実行中...")
    
    raw_data = load_data(ticker, start_date.isoformat(), end_date.isoformat())
    if raw_data is None: 
        st.error("データが取得できませんでした。最適化を中止します。")
        return

    param_grid = get_param_grid(preset_choice, strategy_type)
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 既存のパラメータを保持しつつ、最適化対象のパラメータで上書きする
    base_params = st.session_state.get('params', {}).copy()
    full_combinations = []
    for p_comb in param_combinations:
        temp_params = base_params.copy()
        temp_params.update(p_comb)
        full_combinations.append(temp_params)

    total_combinations = len(full_combinations)
    st.write(f"{total_combinations}通りの組み合わせをテストします。")
    progress_bar = st.progress(0)

    best_sharpe = -1
    best_params_sharpe = None
    best_return = -100
    best_params_return = None

    for i, params in enumerate(full_combinations):
        metrics = evaluate_performance(params, raw_data, strategy_type)
        progress_bar.progress((i + 1) / total_combinations)
        if metrics:
            total_return = metrics.get('total_return', 0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0)

            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params_sharpe = params
            
            if total_return > best_return:
                best_return = total_return
                best_params_return = params

    st.subheader("🏆 最適化結果")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("最高シャープレシオ", f"{best_sharpe:.2f}")
        st.json(best_params_sharpe)
        if st.button("この設定を適用 (シャープレシオ)"):
            st.session_state.params = best_params_sharpe
            st.rerun()

    with col2:
        st.metric("最高リターン", f"{best_return:.2f}%")
        st.json(best_params_return)
        if st.button("この設定を適用 (リターン)"):
            st.session_state.params = best_params_return
            st.rerun()
