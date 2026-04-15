import streamlit as st
from datetime import datetime, timedelta
from utils import save_settings, load_settings
from optimizer_ui import run_optimization

def setup_sidebar(TICKERS, PRESETS, params_config):
    st.sidebar.title("設定")

    # --- Session State Initialization ---
    if 'ticker' not in st.session_state: 
        st.session_state.ticker = TICKERS[0]
    if 'params' not in st.session_state: 
        st.session_state.params = PRESETS["スイングトレード"]
    if 'preset_choice' not in st.session_state:
        st.session_state.preset_choice = "スイングトレード"
    if 'strategy_type' not in st.session_state:
        st.session_state.strategy_type = "トレンドフォロー"

    # --- UI Components ---
    ticker_choice = st.sidebar.selectbox("ティッカー", TICKERS, index=TICKERS.index(st.session_state.ticker) if st.session_state.ticker in TICKERS else 0)
    if ticker_choice != st.session_state.ticker:
        st.session_state.ticker = ticker_choice
        st.cache_data.clear()
        st.rerun()

    strategy_type = st.sidebar.selectbox("戦略タイプ", ["トレンドフォロー", "逆張り"], index=["トレンドフォロー", "逆張り"].index(st.session_state.strategy_type))
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

    st.sidebar.header("技術分析設定")
    
    # 動的に表示するパラメータを決定
    visible_params = list(params_config.keys())
    if strategy_type == "トレンドフォロー":
        visible_params = [p for p in visible_params if 'dev' not in p and 'lower' not in p and 'upper' not in p]
    else: # 逆張り
        visible_params = [p for p in visible_params if 'macd' not in p and 'window' not in p and p != 'adx_threshold'
                          and p != 'score_smooth_period' and p != 'ema_slope_period']

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
            st.cache_data.clear()
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
