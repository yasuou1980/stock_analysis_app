import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import logging
import json

# --- Page and Logging Configuration ---
st.set_page_config(
    page_title="プロフェッショナル株式戦略分析",
    page_icon="🏆",
    layout="wide"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions & Classes ---

def validate_date_range(start_date, end_date):
    """日付範囲の入力を検証する"""
    if start_date >= end_date:
        st.error("開始日は終了日より前の日付を選択してください。")
        return False
    return True

def validate_data_quality(data):
    """データ品質の包括的チェック"""
    issues = []
    if (data['high'] < data['low']).any():
        issues.append("⚠️ 価格データに矛盾があります（高値 < 安値）")
    if 'volume' in data and (data['volume'] == 0).sum() > len(data) * 0.1:
        issues.append("⚠️ 出来高0の日が10%以上あります")
    price_changes = data['close'].pct_change().abs()
    extreme_gaps = (price_changes > 0.2).sum()
    if extreme_gaps > 0:
        issues.append(f"⚠️ {extreme_gaps}日に20%以上の価格ギャップがあります")
    return issues

def save_settings(settings, filename="settings.json"):
    """現在の設定をファイルに保存する"""
    settings_serializable = {k: v.isoformat() if isinstance(v, datetime.date) else v for k, v in settings.items()}
    with open(filename, 'w') as f:
        json.dump(settings_serializable, f, indent=4)
    st.sidebar.success(f"設定を {filename} に保存しました。")

def load_settings(filename="settings.json"):
    """ファイルから設定を読み込む"""
    try:
        with open(filename, 'r') as f:
            settings = json.load(f)
            for key, value in settings.items():
                if key in ['start_date', 'end_date'] and isinstance(value, str):
                    settings[key] = datetime.fromisoformat(value).date()
            return settings
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.sidebar.error(f"設定ファイルの読み込みに失敗しました: {e}")
        return {}

# --- Data Caching and Processing ---

@st.cache_data
def load_data(_ticker, start, end):
    """Yahoo Financeから株価データを読み込む（エラー修正版）"""
    try:
        data = yf.download(_ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            st.error(f"ティッカー {_ticker} のデータが見つかりません。")
            return None

        # --- ここからが修正部分 ---
        # yfinanceがMultiIndexを返す場合に対応
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 列名を確実に文字列に変換してから小文字化
        data.columns = [str(col).lower() for col in data.columns]
        # --- 修正部分ここまで ---

        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"データの読み込み中に予期せぬエラーが発生しました: {e}")
        logger.error(f"Data loading error for ticker {_ticker}: {e}")
        return None

@st.cache_data
def calculate_indicators_and_signals(_data, params):
    """技術指標、シグナル、およびリスク管理指標を計算する"""
    data = _data.copy()
    strategy = ta.Strategy(name="Custom Strategy", ta=[
        {"kind": "sma", "length": params['short_window'], "col_names": "sma_short"},
        {"kind": "sma", "length": params['long_window'], "col_names": "sma_long"},
        {"kind": "rsi", "length": params['rsi_period'], "col_names": "rsi"},
        {"kind": "macd", "fast": params['macd_fast'], "slow": params['macd_slow'], "signal": params['macd_signal'], "col_names": ("macd", "macdh", "macds")},
    ])
    data.ta.strategy(strategy)
    data['volatility'] = data['close'].pct_change().rolling(window=30).std().bfill()
    scores = (np.where(data['sma_short'] > data['sma_long'], 2, -2) +
              np.select([data['rsi'] < 30, data['rsi'] > 70], [1.5, -1.5], default=0) +
              np.where(data['macdh'] > 0, 2, -2))
    data['composite_signal'] = np.select([scores >= 3.5, scores <= -3.5], ["BUY", "SELL"], default="HOLD")
    data.dropna(inplace=True)
    return data

# --- Analysis & Calculation Functions ---
def check_alerts(latest_data):
    """重要なシグナルを検出し、アラートリストを返す"""
    alerts = []
    if 'rsi' in latest_data and latest_data['rsi'] < 25:
        alerts.append(f"🔴 RSI ({latest_data['rsi']:.1f}) が極度の売られ過ぎ水準です")
    elif 'rsi' in latest_data and latest_data['rsi'] > 75:
        alerts.append(f"🟢 RSI ({latest_data['rsi']:.1f}) が極度の買われ過ぎ水準です")
    
    if 'composite_signal' in latest_data and latest_data['composite_signal'] != 'HOLD':
        alerts.append(f"📊 強い「{latest_data['composite_signal']}」シグナルが発生中です")
    return alerts

def safe_calculate_signal_strength(row):
    """エラー処理を含むシグナル強度計算"""
    try:
        if pd.isna(row['sma_long']) or row['sma_long'] == 0: return 50
        strength = 50
        sma_divergence = (row['sma_short'] - row['sma_long']) / row['sma_long'] * 100
        strength += min(max(sma_divergence * 2, -25), 25)
        if not pd.isna(row['rsi']):
            if row['rsi'] < 30: strength += (30 - row['rsi']) * 0.5
            elif row['rsi'] > 70: strength -= (row['rsi'] - 70) * 0.5
        return max(0, min(100, strength))
    except Exception as e:
        logger.warning(f"Signal strength calculation error: {e}")
        return 50

def calculate_position_size(price, volatility, portfolio_value):
    """ボラティリティベースのポジションサイズを計算"""
    if volatility == 0: return 0
    target_risk, max_position_ratio = 0.02, 0.9
    size_in_currency = (portfolio_value * target_risk) / volatility
    return min(size_in_currency / price, (portfolio_value * max_position_ratio) / price)

@st.cache_data
def backtest_strategy(_data, initial_capital, commission_rate, slippage):
    """ポジションサイジングを導入した高度なバックテスト"""
    portfolio = {'cash': initial_capital, 'shares': 0}
    portfolio_values = []
    trades = []
    for i in range(len(_data)):
        row = _data.iloc[i]
        current_portfolio_value = portfolio['cash'] + (portfolio['shares'] * row['close'])
        portfolio_values.append(current_portfolio_value)
        if row['composite_signal'] == "BUY" and portfolio['shares'] == 0:
            buy_price = row['close'] * (1 + slippage)
            shares_to_buy = calculate_position_size(buy_price, row['volatility'], current_portfolio_value)
            cost = shares_to_buy * buy_price * (1 + commission_rate)
            if shares_to_buy > 0 and portfolio['cash'] >= cost:
                portfolio.update({'shares': shares_to_buy, 'cash': portfolio['cash'] - cost})
                trades.append({'date': _data.index[i], 'action': 'BUY', 'shares': shares_to_buy, 'price': buy_price, 'value': cost})
        elif row['composite_signal'] == "SELL" and portfolio['shares'] > 0:
            sell_price = row['close'] * (1 - slippage)
            revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
            trades.append({'date': _data.index[i], 'action': 'SELL', 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
            portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
    return {'portfolio_values': portfolio_values, 'dates': _data.index, 'trades': trades}

@st.cache_data
def calculate_performance_metrics(_portfolio_values, _dates):
    """パフォーマンス指標の計算を分離してキャッシュ"""
    if len(_portfolio_values) < 2: return {}
    returns = pd.Series(_portfolio_values, index=_dates).pct_change().dropna()
    if returns.empty: return {}
    
    total_return = (_portfolio_values[-1] / _portfolio_values[0] - 1) * 100
    pv_arr = np.array(_portfolio_values)
    running_max = np.maximum.accumulate(pv_arr)
    max_dd = np.min((pv_arr - running_max) / running_max) * 100 if running_max.any() and (running_max > 0).any() else 0
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    volatility = returns.std() * np.sqrt(252) * 100
    var_95 = returns.quantile(0.05) * 100
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    return {'total_return': total_return, 'max_drawdown': max_dd, 'sharpe_ratio': sharpe, 
            'volatility': volatility, 'var_95': var_95, 'skewness': skew, 'kurtosis': kurt}

def calculate_trade_metrics(trades_df):
    """取引の詳細分析"""
    if trades_df.empty: return {}
    buys = trades_df[trades_df['action'] == 'BUY']
    sells = trades_df[trades_df['action'] == 'SELL']
    if buys.empty or sells.empty: return {}
    hold_periods = [(sells.iloc[i]['date'] - buys.iloc[i]['date']).days for i in range(min(len(buys), len(sells)))]
    if not hold_periods: return {}
    return {'avg_hold_days': np.mean(hold_periods), 'max_hold_days': np.max(hold_periods), 'min_hold_days': np.min(hold_periods)}

# --- Sidebar UI ---
st.sidebar.title("設定")

PRESETS = {
    "スイングトレード": {'short_window': 20, 'long_window': 50, 'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
    "長期投資": {'short_window': 50, 'long_window': 200, 'rsi_period': 20, 'macd_fast': 20, 'macd_slow': 40, 'macd_signal': 10}
}
if 'params' not in st.session_state:
    st.session_state.params = PRESETS["スイングトレード"]

ticker = st.sidebar.selectbox("ティッカー", ("SOXL", "SOXS", "NVDA", "AMD", "TSM"), index=0)
end_date = datetime.now().date()
start_date = st.sidebar.date_input("開始日", end_date - timedelta(days=3*365))
end_date = st.sidebar.date_input("終了日", end_date)

preset_choice = st.sidebar.selectbox("設定プリセット", list(PRESETS.keys()), index=0)
if preset_choice: st.session_state.params = PRESETS[preset_choice]

st.sidebar.header("技術分析設定")
params_config = {
    'short_window': ('短期SMA', 5, 50, 20), 'long_window': ('長期SMA', 55, 200, 50), 
    'rsi_period': ('RSI期間', 7, 30, 14), 'macd_fast': ('MACD短期', 5, 25, 12), 
    'macd_slow': ('MACD長期', 26, 50, 26), 'macd_signal': ('MACDシグナル', 5, 15, 9)
}
params = {key: st.sidebar.slider(label.capitalize(), min_val, max_val, st.session_state.params.get(key, def_val)) for key, (label, min_val, max_val, def_val) in params_config.items()}
st.session_state.params = params

st.sidebar.header("バックテスト設定")
initial_capital = st.sidebar.number_input("初期資金 ($)", 1000, 1000000, 10000, 1000)
commission_rate = st.sidebar.slider("取引手数料 (%)", 0.0, 1.0, 0.1, 0.01) / 100
slippage = st.sidebar.slider("スリッページ (%)", 0.0, 1.0, 0.05, 0.01) / 100

st.sidebar.header("設定管理")
if st.sidebar.button("設定保存"): save_settings({**params, 'ticker': ticker, 'start_date': start_date, 'end_date': end_date})
if st.sidebar.button("設定読込"):
    if loaded := load_settings(): 
        st.session_state.params.update({k: loaded[k] for k in params if k in loaded})
        st.rerun()

# --- Main Application ---
st.title(f"プロフェッショナル株式戦略分析: {ticker}")

raw_data = load_data(ticker, start_date, end_date)
if raw_data is None or not validate_date_range(start_date, end_date): st.stop()
if quality_issues := validate_data_quality(raw_data):
    with st.expander("⚠️ データ品質の警告", expanded=True): 
        for issue in quality_issues:
            st.warning(issue)

data = calculate_indicators_and_signals(raw_data, params)
if data.empty: st.error("分析可能なデータがありません。期間やパラメータを調整してください。"); st.stop()

latest = data.iloc[-1]
if alerts := check_alerts(latest):
    st.subheader("🚨 重要アラート"); [st.info(alert) for alert in alerts]

st.header("現在の市場状況")
cols = st.columns([1.5, 1.5, 1.5, 2])
cols[0].metric("現在価格", f"${latest['close']:.2f}")
cols[1].metric("総合シグナル", latest['composite_signal'])
cols[2].metric("RSI", f"{latest['rsi']:.1f}")
with cols[3]: st.markdown("**シグナル強度**"); st.progress(int(safe_calculate_signal_strength(latest)))

st.header("バックテスト分析")
results = backtest_strategy(data, initial_capital, commission_rate, slippage)
metrics = calculate_performance_metrics(results['portfolio_values'], results['dates'])

if metrics:
    tab1, tab2, tab3 = st.tabs(["パフォーマンス概要", "リスク＆取引分析", "詳細チャート"])
    with tab1:
        st.subheader("パフォーマンスサマリー")
        c1, c2, c3 = st.columns(3)
        c1.metric("最終リターン", f"{metrics.get('total_return', 0):.2f}%")
        c2.metric("最大ドローダウン", f"{metrics.get('max_drawdown', 0):.2f}%")
        c3.metric("シャープレシオ", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.line_chart(pd.DataFrame({'戦略ポートフォリオ': results['portfolio_values']}, index=results['dates']))

    with tab2:
        st.subheader("リスクプロファイル")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("年率ボラティリティ", f"{metrics.get('volatility', 0):.2f}%")
        c2.metric("VaR 95% (日次)", f"{metrics.get('var_95', 0):.2f}%")
        c3.metric("歪度 (Skewness)", f"{metrics.get('skewness', 0):.2f}", help="0より大きいと右に長い裾野（大きな利益が時々）、小さいと左に長い裾野（大きな損失が時々）")
        c4.metric("尖度 (Kurtosis)", f"{metrics.get('kurtosis', 0):.2f}", help="3より大きいと正規分布より尖った分布（テールリスク大）")

        st.subheader("取引詳細分析")
        trades_df = pd.DataFrame(results['trades'])
        trade_metrics = calculate_trade_metrics(trades_df)
        if trade_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("平均保有日数", f"{trade_metrics.get('avg_hold_days', 0):.1f}日")
            c2.metric("最大保有日数", f"{trade_metrics.get('max_hold_days', 0):.0f}日")
            c3.metric("最小保有日数", f"{trade_metrics.get('min_hold_days', 0):.0f}日")

    with tab3:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='価格'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_short'], mode='lines', name=f'SMA {params["short_window"]}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_long'], mode='lines', name=f'SMA {params["long_window"]}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data[data['composite_signal'] == 'BUY'].index, y=data[data['composite_signal'] == 'BUY']['close'], mode='markers', name='買い', marker=dict(symbol='triangle-up', size=10, color='lime')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data[data['composite_signal'] == 'SELL'].index, y=data[data['composite_signal'] == 'SELL']['close'], mode='markers', name='売り', marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['macdh'], name='MACDヒストグラム', marker_color=np.where(data['macdh'] > 0, 'green', 'tomato')), row=2, col=1)
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("🏆 **完成版**: このツールは教育および情報提供目的のものです。投資判断はご自身の責任で行ってください。")
