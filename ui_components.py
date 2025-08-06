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

    # --- UI Components ---
    ticker_choice = st.sidebar.selectbox("ティッカー", TICKERS, index=TICKERS.index(st.session_state.ticker) if st.session_state.ticker in TICKERS else 0)
    if ticker_choice != st.session_state.ticker:
        st.session_state.ticker = ticker_choice
        st.cache_data.clear()
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
    params = {}
    for key, (label, min_val, max_val, _) in params_config.items():
        current_val = st.session_state.params.get(key, PRESETS[st.session_state.preset_choice][key])
        new_val = st.sidebar.slider(label, min_val, max_val, current_val)
        params[key] = new_val
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
            **params, 
            **ps_params,
            'position_sizing_strategy': position_sizing_strategy,
            'ticker': st.session_state.ticker, 
            'start_date': start_date, 
            'end_date': end_date,
            'preset_choice': st.session_state.preset_choice
        })

    if st.sidebar.button("設定読込"):
        if loaded := load_settings():
            st.session_state.params.update({k: loaded[k] for k in params if k in loaded})
            if 'ticker' in loaded and loaded['ticker'] in TICKERS:
                st.session_state.ticker = loaded['ticker']
            if 'preset_choice' in loaded and loaded['preset_choice'] in PRESETS:
                st.session_state.preset_choice = loaded['preset_choice']
            st.cache_data.clear()
            st.rerun()

    st.sidebar.header("最適化")
    if st.sidebar.button("最適化を実行"):
        run_optimization(st.session_state.ticker, start_date, end_date, st.session_state.preset_choice)

    st.sidebar.header("キャッシュ管理")
    if st.sidebar.button("データキャッシュをクリア"):
        st.cache_data.clear()
        st.success("✅ キャッシュをクリアしました")
        st.rerun()

    if st.sidebar.checkbox("デバッグ情報を表示"):
        st.sidebar.subheader("🐞 デバッグ情報")
        st.sidebar.write(f"**選択中のティッカー**: `{st.session_state.ticker}`")
        st.sidebar.write(f"**現在のプリセット**: `{st.session_state.preset_choice}`")
        st.sidebar.json(st.session_state.params, expanded=False)

    return st.session_state.ticker, start_date, end_date, params, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params
