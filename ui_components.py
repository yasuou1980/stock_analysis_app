import streamlit as st
import logging
from datetime import datetime, timedelta
from utils import save_settings, load_settings
from optimizer_ui import run_optimization, auto_optimize_silent

def setup_sidebar(TICKERS, PRESETS, params_config):
    st.sidebar.title("設定")

    # --- Session State Initialization ---
    if 'ticker' not in st.session_state: 
        st.session_state.ticker = TICKERS[0]
    if 'params' not in st.session_state:
        st.session_state.params = PRESETS["スイングトレード"].copy()
    if 'preset_choice' not in st.session_state:
        st.session_state.preset_choice = "スイングトレード"
    if 'strategy_type' not in st.session_state:
        st.session_state.strategy_type = "逆張り"

    # --- UI Components ---
    ticker_choice = st.sidebar.selectbox("ティッカー", TICKERS, index=TICKERS.index(st.session_state.ticker) if st.session_state.ticker in TICKERS else 0)
    if ticker_choice != st.session_state.ticker:
        # キャッシュはティッカー・期間ごとにキーが分かれているため、切替時に
        # 全消去しない。全消去すると全銘柄サマリーの再ダウンロードが一斉に走り、
        # Yahoo Finance のレート制限で全銘柄のデータ取得が失敗する。
        st.session_state.ticker = ticker_choice
        st.rerun()

    _strategy_options = ["逆張り", "トレンドフォロー", "レジーム切替"]
    _strategy_idx = _strategy_options.index(st.session_state.strategy_type) if st.session_state.strategy_type in _strategy_options else 0
    strategy_type = st.sidebar.selectbox("戦略タイプ", _strategy_options, index=_strategy_idx)
    if strategy_type != st.session_state.strategy_type:
        st.session_state.strategy_type = strategy_type
        st.rerun()

    preset_choice = st.sidebar.selectbox("設定プリセット", list(PRESETS.keys()), index=list(PRESETS.keys()).index(st.session_state.preset_choice))
    if preset_choice != st.session_state.preset_choice:
        st.session_state.preset_choice = preset_choice
        st.session_state.params = PRESETS[preset_choice].copy()
        st.rerun()

    end_date = datetime.now().date()
    start_date = st.sidebar.date_input("開始日", end_date - timedelta(days=3*365))
    end_date = st.sidebar.date_input("終了日", end_date)

    # --- 戦略銘柄選択時の自動最適化 ---
    # ticker / strategy / preset / 期間 が変わったときに限り自動で軽量最適化を実行し、
    # 結果を params に反映する。キャッシュにより同一条件の再実行は即時。
    opt_key = (ticker_choice, strategy_type, preset_choice,
               start_date.isoformat(), end_date.isoformat())
    if st.session_state.get('_last_opt_key') != opt_key:
        base_params_for_opt = PRESETS[preset_choice].copy()
        # 自動最適化はあくまで補助機能。データ取得失敗や計算エラーで
        # アプリ全体が落ちないよう、失敗時はプリセット値のまま続行する。
        try:
            with st.sidebar.spinner("🔍 自動最適化中..."):
                optimized = auto_optimize_silent(
                    ticker_choice, start_date.isoformat(), end_date.isoformat(),
                    preset_choice, strategy_type, base_params_for_opt
                )
        except Exception as e:
            logging.getLogger(__name__).error(f"自動最適化に失敗しました ({ticker_choice}): {e}")
            optimized = None
            st.sidebar.warning("⚠️ 自動最適化に失敗したため、プリセット値を使用します")
        if optimized:
            st.session_state.params = optimized
            st.sidebar.success("✅ 自動最適化を適用しました")
        st.session_state._last_opt_key = opt_key

    st.sidebar.header("技術分析設定")

    # 動的に表示するパラメータを決定
    visible_params = list(params_config.keys())
    if strategy_type == "トレンドフォロー":
        visible_params = [p for p in visible_params if 'dev' not in p and 'lower' not in p and 'upper' not in p]
    elif strategy_type == "逆張り":
        visible_params = [p for p in visible_params if 'macd' not in p and 'window' not in p and p != 'adx_threshold'
                          and p != 'score_smooth_period' and p != 'ema_slope_period']
    # レジーム切替: 全パラメータを表示

    # セッションステートのパラメータを直接更新
    for key in visible_params:
        label, min_val, max_val, _ = params_config[key]
        current_val = st.session_state.params.get(key, PRESETS[st.session_state.preset_choice].get(key, _))
        
        if isinstance(min_val, float) or isinstance(max_val, float) or isinstance(current_val, float):
             new_val = st.sidebar.slider(label, float(min_val), float(max_val), float(current_val))
        else:
             new_val = st.sidebar.slider(label, int(min_val), int(max_val), int(current_val))
        st.session_state.params[key] = new_val

    st.sidebar.header("バックテスト設定")
    initial_capital = st.sidebar.number_input("初期資金 ($)", 1000, 1000000, 10000, 1000)
    commission_rate = st.sidebar.slider("取引手数料 (%)", 0.0, 1.0, 0.1, 0.01) / 100
    slippage = st.sidebar.slider("スリッページ (%)", 0.0, 1.0, 0.05, 0.01) / 100

    st.sidebar.header("ポジションサイジング設定")
    position_sizing_strategy = st.sidebar.selectbox("戦略", ["固定リスク率 (Volatility Adjusted)", "固定ポートフォリオ比率"])
    ps_params = {}
    if position_sizing_strategy == "固定リスク率 (Volatility Adjusted)":
        ps_params['target_risk'] = st.sidebar.slider("目標リスク/取引 (%)", 0.5, 5.0, 2.0, 0.1) / 100
        ps_params['max_position_ratio'] = st.sidebar.slider("最大ポジション比率 (%)", 10.0, 100.0, 90.0, 1.0) / 100
    elif position_sizing_strategy == "固定ポートフォリオ比率":
        ps_params['fixed_ratio'] = st.sidebar.slider("固定比率 (%)", 1.0, 100.0, 25.0, 1.0) / 100

    st.sidebar.header("設定管理")
    if st.sidebar.button("設定保存"):
        save_settings({
            **st.session_state.params,
            **ps_params,
            'position_sizing_strategy': position_sizing_strategy,
            'ticker': st.session_state.ticker, 
            'start_date': start_date, 
            'end_date': end_date,
            'preset_choice': st.session_state.preset_choice,
            'strategy_type': st.session_state.strategy_type
        })

    if st.sidebar.button("設定読込"):
        if loaded := load_settings():
            st.session_state.params.update({k: v for k, v in loaded.items() if k in params_config})
            if 'ticker' in loaded and loaded['ticker'] in TICKERS:
                st.session_state.ticker = loaded['ticker']
            if 'preset_choice' in loaded and loaded['preset_choice'] in PRESETS:
                st.session_state.preset_choice = loaded['preset_choice']
            if 'strategy_type' in loaded:
                st.session_state.strategy_type = loaded['strategy_type']
            st.rerun()

    st.sidebar.header("最適化")
    run_optimization_clicked = st.sidebar.button("最適化を実行")

    st.sidebar.header("キャッシュ管理")
    if st.sidebar.button("データキャッシュをクリア"):
        st.cache_data.clear()
        st.success("✅ キャッシュをクリアしました")
        st.rerun()

    if st.sidebar.checkbox("デバッグ情報を表示"):
        st.sidebar.subheader("🐞 デバッグ情報")
        st.sidebar.write(f"**選択中のティッカー**: `{st.session_state.ticker}`")
        st.sidebar.write(f"**現在のプリセット**: `{st.session_state.preset_choice}`")
        st.sidebar.write(f"**現在の戦略**: `{st.session_state.strategy_type}`")
        st.sidebar.json(st.session_state.params, expanded=False)

    return st.session_state.ticker, start_date, end_date, st.session_state.params, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params, st.session_state.strategy_type, run_optimization_clicked
