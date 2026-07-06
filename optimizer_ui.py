import streamlit as st
import pandas as pd
import hashlib
from data_loader import load_data
from backtester import calculate_indicators_and_signals, backtest_strategy, calculate_performance_metrics, resolve_ticker_class
from utils import load_config
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

    pyramid_thr = float(params.get('pyramid_threshold', 0.10))
    results_hash = hashlib.sha256((str(data.values.tobytes()) + str(initial_capital) + str(commission_rate) + str(slippage) + position_sizing_strategy + str(ps_params) + strategy_type + str(pyramid_thr)).encode()).hexdigest()
    results = backtest_strategy(results_hash, data, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params, strategy_type, pyramid_thr)
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


def _phase1_grid(preset_choice, strategy_type):
    """Phase 1 の粗い探索グリッドを返す。高速化のため要素数を絞っている。"""
    if strategy_type == "トレンドフォロー":
        if preset_choice == "スイングトレード":
            return {
                'short_window': [10, 15, 20],   # 3
                'long_window': [40, 50, 65],     # 3 (プリセット 50 を中央に配置)
                'adx_threshold': [18, 22],       # 2 (プリセット 18 を下限に合わせる)
            }  # = 18 通り
        else:  # 長期投資
            return {
                'short_window': [30, 40, 50],    # 3 (プリセット 40 を中央に配置)
                'long_window': [150, 200],       # 2
                'adx_threshold': [22, 28],       # 2 (プリセット 22 を下限に合わせる)
            }  # = 12 通り
    else:  # 逆張り
        return {
            'rsi_upper': [70, 75],   # 2
            'rsi_lower': [25, 30],   # 2
            'bb_std': [2.0, 2.5],    # 2
        }  # = 8 通り


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


@st.cache_data(show_spinner=False, ttl=3600)
def auto_optimize_silent(ticker, start_date_iso, end_date_iso, preset_choice, strategy_type, base_params):
    """UI なしで Phase 1 のみを実行して最良パラメータ (sharpe) を返す。

    戦略銘柄選択時の自動最適化用。キャッシュが効くため、同一条件での再実行は即時。
    対応戦略: トレンドフォロー / 逆張り（レジーム切替は None を返す）。
    """
    if strategy_type not in ("トレンドフォロー", "逆張り"):
        return None

    raw_data = load_data(ticker, start_date_iso, end_date_iso)
    if raw_data is None:
        return None

    grid = _phase1_grid(preset_choice, strategy_type)
    base = {**dict(base_params), 'ticker_class': resolve_ticker_class(ticker, load_config())}
    combos = _build_combinations(grid, base, preset_choice, strategy_type)

    best_sharpe = -999
    best_params = None
    for p in combos:
        metrics = evaluate_performance(p, raw_data, strategy_type)
        if metrics:
            sharpe = metrics.get('sharpe_ratio', -999)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = p.copy()

    return best_params


def run_optimization(ticker, start_date, end_date, preset_choice, strategy_type):
    st.subheader(f"⚙️ {strategy_type}戦略の2段階最適化を実行中...")

    raw_data = load_data(ticker, start_date.isoformat(), end_date.isoformat())
    if raw_data is None:
        st.error("データが取得できませんでした。最適化を中止します。")
        return

    base_params = st.session_state.get('params', {}).copy()
    base_params['ticker_class'] = resolve_ticker_class(ticker, load_config())

    # ========================================
    # Phase 1: 主要パラメータの粗い探索
    # ========================================
    phase1_grid = _phase1_grid(preset_choice, strategy_type)

    phase1_combos = _build_combinations(phase1_grid, base_params, preset_choice, strategy_type)
    phase2_est = 8  # Phase 2 は最大 2^3 = 8
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
            step_sw, step_lw, step_adx = 3, 7, 3
            min_sw, min_lw = 5, 25
        else:
            step_sw, step_lw, step_adx = 5, 25, 3
            min_sw, min_lw = 30, 100

        phase2_grid = {
            'short_window': sorted(set([max(min_sw, sw - step_sw), sw + step_sw])),
            'long_window': sorted(set([max(min_lw, lw - step_lw), lw + step_lw])),
            'adx_threshold': sorted(set([max(10, adx - step_adx), adx + step_adx])),
        }
    else:  # 逆張り
        ru = phase1_best.get('rsi_upper', 70)
        rl = phase1_best.get('rsi_lower', 30)
        bs = phase1_best.get('bb_std', 2.0)

        phase2_grid = {
            'rsi_upper': sorted(set([max(60, ru - 3), min(85, ru + 3)])),
            'rsi_lower': sorted(set([max(15, rl - 3), min(40, rl + 3)])),
            'bb_std': sorted(set([max(1.5, round(bs - 0.25, 2)), min(3.0, round(bs + 0.25, 2))])),
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
