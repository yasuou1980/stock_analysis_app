import streamlit as st
import pandas as pd
import hashlib
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

    data_hash = hashlib.sha256((str(raw_data.values.tobytes()) + str(params) + strategy_type).encode()).hexdigest()
    data = calculate_indicators_and_signals(data_hash, raw_data, params, strategy_type)
    if data.empty: return None

    results_hash = hashlib.sha256((str(data.values.tobytes()) + str(initial_capital) + str(commission_rate) + str(slippage) + position_sizing_strategy + str(ps_params)).encode()).hexdigest()
    results = backtest_strategy(results_hash, data, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params)
    metrics = calculate_performance_metrics(results['portfolio_values'], results['dates'])

    return metrics


def _derive_trend_params(short_window, long_window, adx_threshold, preset_choice):
    """主要3パラメータから関連パラメータを自動導出する。

    トレンドフォローの9パラメータのうち、独立に意味があるのは
    short_window, long_window, adx_threshold の3つ。
    残りの6パラメータ（rsi_period, macd_fast/slow/signal, score_smooth_period,
    ema_slope_period）はこれらから導出できる相関の高いパラメータであるため、
    個別に探索するのは無駄な組み合わせ爆発を招く。
    """
    if preset_choice == "スイングトレード":
        return {
            'short_window': short_window,
            'long_window': long_window,
            'rsi_period': short_window,
            'macd_fast': short_window,
            'macd_slow': max(long_window // 2, short_window + 5),
            'macd_signal': max(short_window // 2, 5),
            'adx_threshold': adx_threshold,
            'score_smooth_period': 3,
            'ema_slope_period': 5,
        }
    else:  # 長期投資
        return {
            'short_window': short_window,
            'long_window': long_window,
            'rsi_period': max(short_window // 2, 14),
            'macd_fast': max(short_window // 2, 15),
            'macd_slow': max(long_window // 3, 30),
            'macd_signal': max(short_window // 4, 10),
            'adx_threshold': adx_threshold,
            'score_smooth_period': 5,
            'ema_slope_period': 8,
        }


def _build_combinations(param_grid, base_params, preset_choice=None, strategy_type=None):
    """パラメータグリッドから全組み合わせを生成する。"""
    keys, values = zip(*param_grid.items())
    combinations = []
    for v in itertools.product(*values):
        p_comb = dict(zip(keys, v))
        temp_params = base_params.copy()

        if strategy_type == "トレンドフォロー":
            derived = _derive_trend_params(
                p_comb['short_window'], p_comb['long_window'],
                p_comb['adx_threshold'], preset_choice
            )
            temp_params.update(derived)
        else:
            temp_params.update(p_comb)

        combinations.append(temp_params)
    return combinations


def _search_best(combinations, raw_data, strategy_type, progress_bar, offset=0, total=None):
    """組み合わせリストを評価して最良のシャープレシオ/リターンを返す。"""
    if total is None:
        total = len(combinations)

    best_sharpe = -999
    best_params_sharpe = None
    best_return = -999
    best_params_return = None

    for i, params in enumerate(combinations):
        metrics = evaluate_performance(params, raw_data, strategy_type)
        progress_bar.progress((offset + i + 1) / total)
        if metrics:
            sharpe = metrics.get('sharpe_ratio', 0)
            ret = metrics.get('total_return', 0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params_sharpe = params.copy()
            if ret > best_return:
                best_return = ret
                best_params_return = params.copy()

    return best_sharpe, best_params_sharpe, best_return, best_params_return


def run_optimization(ticker, start_date, end_date, preset_choice, strategy_type):
    st.subheader(f"⚙️ {strategy_type}戦略の2段階最適化を実行中...")

    raw_data = load_data(ticker, start_date.isoformat(), end_date.isoformat())
    if raw_data is None:
        st.error("データが取得できませんでした。最適化を中止します。")
        return

    base_params = st.session_state.get('params', {}).copy()

    # ========================================
    # Phase 1: 主要パラメータの粗い探索
    # ========================================
    if strategy_type == "トレンドフォロー":
        if preset_choice == "スイングトレード":
            phase1_grid = {
                'short_window': [8, 12, 15, 20],   # 4
                'long_window': [35, 45, 55, 65],    # 4
                'adx_threshold': [15, 20, 25],      # 3
            }  # = 48 通り
        else:  # 長期投資
            phase1_grid = {
                'short_window': [40, 50, 60],       # 3
                'long_window': [150, 200, 250],     # 3
                'adx_threshold': [20, 25, 30],      # 3
            }  # = 27 通り
    else:  # 逆張り
        phase1_grid = {
            'rsi_upper': [70, 75, 80],      # 3
            'rsi_lower': [20, 25, 30],      # 3
            'bb_std': [2.0, 2.5],           # 2
            'dev_upper': [10, 15],          # 2
            'dev_lower': [-15, -10],        # 2
        }  # = 72 通り

    phase1_combos = _build_combinations(phase1_grid, base_params, preset_choice, strategy_type)
    phase2_est = 27  # Phase 2 は最大 3^3 = 27
    total_est = len(phase1_combos) + phase2_est

    # 旧方式との比較表示
    old_total = 19683 if strategy_type == "トレンドフォロー" else 864
    reduction_pct = 100 - (100 * total_est // old_total)

    st.write(f"**Phase 1**: {len(phase1_combos)}通りの粗い探索 → **Phase 2**: 最良付近の精密探索（最大{phase2_est}通り）")
    st.write(f"合計: 約**{total_est}通り**（従来{old_total:,}通りから **{reduction_pct}%削減**）")

    progress_bar = st.progress(0)
    st.markdown("##### Phase 1: 主要パラメータの粗い探索")

    best_sharpe, best_p_sharpe, best_ret, best_p_ret = _search_best(
        phase1_combos, raw_data, strategy_type, progress_bar, 0, total_est
    )

    phase1_best = best_p_sharpe or best_p_ret or base_params
    if phase1_best is None:
        st.error("Phase 1 で有効な結果が得られませんでした。")
        return

    # ========================================
    # Phase 2: 最良パラメータ付近の精密探索
    # ========================================
    st.markdown("##### Phase 2: 最良付近の精密探索")

    if strategy_type == "トレンドフォロー":
        sw = phase1_best['short_window']
        lw = phase1_best['long_window']
        adx = phase1_best['adx_threshold']
        if preset_choice == "スイングトレード":
            step_sw, step_lw, step_adx = 2, 5, 3
            min_sw, min_lw = 5, 25
        else:
            step_sw, step_lw, step_adx = 5, 20, 3
            min_sw, min_lw = 30, 100

        phase2_grid = {
            'short_window': sorted(set([max(min_sw, sw - step_sw), sw, sw + step_sw])),
            'long_window': sorted(set([max(min_lw, lw - step_lw), lw, lw + step_lw])),
            'adx_threshold': sorted(set([max(10, adx - step_adx), adx, adx + step_adx])),
        }
    else:  # 逆張り
        ru = phase1_best.get('rsi_upper', 70)
        rl = phase1_best.get('rsi_lower', 30)
        bs = phase1_best.get('bb_std', 2.0)

        phase2_grid = {
            'rsi_upper': sorted(set([max(60, ru - 3), ru, min(85, ru + 3)])),
            'rsi_lower': sorted(set([max(15, rl - 3), rl, min(40, rl + 3)])),
            'bb_std': sorted(set([max(1.5, round(bs - 0.25, 2)), bs, min(3.0, round(bs + 0.25, 2))])),
        }

    phase2_combos = _build_combinations(phase2_grid, base_params, preset_choice, strategy_type)

    # Phase 1 で既に評価済みの組み合わせを除外
    phase1_hashes = {tuple(sorted(p.items())) for p in phase1_combos}
    phase2_combos = [p for p in phase2_combos if tuple(sorted(p.items())) not in phase1_hashes]

    actual_total = len(phase1_combos) + len(phase2_combos)

    s2, p_s2, r2, p_r2 = _search_best(
        phase2_combos, raw_data, strategy_type, progress_bar,
        len(phase1_combos), actual_total
    )

    # Phase 1 + Phase 2 の全体最良を決定
    if s2 is not None and s2 > best_sharpe:
        best_sharpe = s2
        best_p_sharpe = p_s2
    if r2 is not None and r2 > best_ret:
        best_ret = r2
        best_p_ret = p_r2

    # ========================================
    # 結果表示
    # ========================================
    st.subheader("🏆 最適化結果")
    st.write(f"テスト済み: **{actual_total}通り**（従来{old_total:,}通り）")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("最高シャープレシオ", f"{best_sharpe:.2f}")
        st.json(best_p_sharpe)
        if st.button("この設定を適用 (シャープレシオ)"):
            st.session_state.params = best_p_sharpe
            st.rerun()

    with col2:
        st.metric("最高リターン", f"{best_ret:.2f}%")
        st.json(best_p_ret)
        if st.button("この設定を適用 (リターン)"):
            st.session_state.params = best_p_ret
            st.rerun()
