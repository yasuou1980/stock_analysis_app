import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="SOXL/SOXS Investment Strategy",
    page_icon="💹",
    layout="wide"
)

# --- Technical Analysis Functions ---
def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    exp1 = data.ewm(span=fast).mean()
    exp2 = data.ewm(span=slow).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram, signal_line

# --- Helper Functions ---
def validate_date_range(start_date, end_date):
    """Validate date range inputs."""
    if start_date >= end_date:
        st.error("開始日は終了日より前の日付を選択してください。")
        return False
    
    if (end_date - start_date).days < 100:
        st.warning("技術指標の計算には最低100日のデータが推奨されます。")
    
    return True

def generate_composite_signal(data):
    """Generate composite trading signals based on multiple indicators."""
    signals = []
    
    for i in range(len(data)):
        signal = "HOLD"
        score = 0
        
        # SMA Signal (Weight: 2)
        if data.iloc[i]['sma_short'] > data.iloc[i]['sma_long']:
            score += 2
        else:
            score -= 2
            
        # RSI Signal (Weight: 1)
        rsi_val = data.iloc[i]['rsi']
        if rsi_val < 30:  # Oversold - Buy signal
            score += 1
        elif rsi_val > 70:  # Overbought - Sell signal
            score -= 1
            
        # Stochastic Signal (Weight: 1)
        stoch_val = data.iloc[i]['stochk']
        if stoch_val < 20:  # Oversold
            score += 1
        elif stoch_val > 80:  # Overbought
            score -= 1
            
        # MACD Signal (Weight: 2)
        if data.iloc[i]['macdh'] > 0:
            score += 2
        else:
            score -= 2
            
        # Generate final signal
        if score >= 3:
            signal = "BUY"
        elif score <= -3:
            signal = "SELL"
            
        signals.append(signal)
    
    return signals

def backtest_strategy(data, initial_capital=10000):
    """Perform backtesting on the trading strategy."""
    if data.empty:
        return None
    
    portfolio = {
        'cash': initial_capital,
        'shares': 0,
        'portfolio_value': [],
        'trades': [],
        'positions': []
    }
    
    buy_and_hold_shares = initial_capital / data.iloc[0]['close']
    buy_and_hold_values = []
    
    for i in range(len(data)):
        current_price = data.iloc[i]['close']
        current_signal = data.iloc[i]['composite_signal']
        current_date = data.index[i]
        
        # Calculate current portfolio value
        current_portfolio_value = portfolio['cash'] + (portfolio['shares'] * current_price)
        portfolio['portfolio_value'].append(current_portfolio_value)
        
        # Buy and hold benchmark
        buy_and_hold_value = buy_and_hold_shares * current_price
        buy_and_hold_values.append(buy_and_hold_value)
        
        # Execute trades based on signals
        if current_signal == "BUY" and portfolio['shares'] == 0 and portfolio['cash'] > current_price:
            # Buy signal - enter position
            shares_to_buy = int(portfolio['cash'] // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                portfolio['shares'] = shares_to_buy
                portfolio['cash'] -= cost
                portfolio['trades'].append({
                    'date': current_date,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': cost
                })
                
        elif current_signal == "SELL" and portfolio['shares'] > 0:
            # Sell signal - exit position
            revenue = portfolio['shares'] * current_price
            portfolio['trades'].append({
                'date': current_date,
                'action': 'SELL',
                'shares': portfolio['shares'],
                'price': current_price,
                'value': revenue
            })
            portfolio['cash'] += revenue
            portfolio['shares'] = 0
    
    # Calculate performance metrics
    if len(portfolio['portfolio_value']) > 0:
        final_value = portfolio['portfolio_value'][-1]
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate maximum drawdown
        portfolio_values = np.array(portfolio['portfolio_value'])
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max * 100
        max_drawdown = np.min(drawdown)
        
        # Buy and hold performance
        buy_and_hold_return = (buy_and_hold_values[-1] - initial_capital) / initial_capital * 100
        
        # Calculate Sharpe ratio (simplified - assuming daily data)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        return {
            'portfolio_values': portfolio['portfolio_value'],
            'buy_and_hold_values': buy_and_hold_values,
            'trades': portfolio['trades'],
            'final_value': final_value,
            'total_return': total_return,
            'buy_and_hold_return': buy_and_hold_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(portfolio['trades'])
        }
    
    return None

# --- Data Loading and Caching ---
@st.cache_data
def load_data(_ticker, start, end):
    """Loads and robustly cleans historical stock data from Yahoo Finance."""
    try:
        # Validate inputs
        if not _ticker or not isinstance(_ticker, str):
            raise ValueError("無効なティッカーシンボルです。")
        
        logger.info(f"Loading data for {_ticker} from {start} to {end}")
        
        data = yf.download(_ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if data.empty:
            raise ValueError(f"ティッカー {_ticker} のデータが見つかりません。シンボルが正しいか確認してください。")

        # Flatten MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)

        # Standardize column names to lowercase
        data.columns = [str(col).lower().strip() for col in data.columns]

        # Definitive Renaming Logic
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        rename_map = {}
        current_cols = data.columns.tolist()

        for req_col in required_cols:
            found = False
            for col in current_cols:
                if col == req_col or col.startswith(f"{req_col}_"):
                    rename_map[col] = req_col
                    found = True
                    break
            if not found:
                raise ValueError(f"必要な列 '{req_col}' が見つかりません。利用可能な列: {current_cols}")
        
        data.rename(columns=rename_map, inplace=True)

        # Data quality checks
        if len(data) < 50:
            raise ValueError("データ量が不足しています。より長い期間を選択してください。")
        
        # Remove rows with all NaN values
        data.dropna(how='all', inplace=True)
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"必要な列が不足しています: {missing_cols}")

        logger.info(f"Successfully loaded {len(data)} rows of data")
        return data
        
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        st.error(f"データの読み込み中にエラーが発生しました: {str(e)}")
        return None

# --- Sidebar for User Input ---
st.sidebar.title("設定")
ticker = st.sidebar.selectbox(
    "ティッカー",
    ("SOXL", "SOXS"),
    index=0
)

# Default date range: last 2 years
end_date = datetime.now().date()
start_date = end_date - timedelta(days=2*365)

start_date = st.sidebar.date_input("開始日", start_date)
end_date = st.sidebar.date_input("終了日", end_date)

# Validate date range
if not validate_date_range(start_date, end_date):
    st.stop()

st.sidebar.header("技術分析設定")
# Technical analysis parameters with validation
short_window = st.sidebar.slider("短期SMA期間", min_value=5, max_value=50, value=25, step=1)
long_window = st.sidebar.slider("長期SMA期間", min_value=short_window+5, max_value=200, value=75, step=1)
rsi_period = st.sidebar.slider("RSI期間", min_value=7, max_value=30, value=14, step=1)
stoch_k = st.sidebar.slider("ストキャスティクス %K", min_value=5, max_value=30, value=14, step=1)
stoch_d = st.sidebar.slider("ストキャスティクス %D", min_value=2, max_value=10, value=3, step=1)
macd_fast = st.sidebar.slider("MACD短期", min_value=5, max_value=25, value=12, step=1)
macd_slow = st.sidebar.slider("MACD長期", min_value=macd_fast+5, max_value=50, value=26, step=1)
macd_signal = st.sidebar.slider("MACDシグナル", min_value=5, max_value=15, value=9, step=1)

st.sidebar.header("バックテスト設定")
initial_capital = st.sidebar.number_input("初期資金 ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

# --- Main Application ---
st.title(f"{ticker} 株式分析")
st.markdown("技術指標に基づく取引シグナルの生成とバックテスト機能を備えた投資分析ツール")

# Load data
data = load_data(ticker, start_date, end_date)

if data is not None and not data.empty:
    try:
        # Calculate technical indicators
        data['sma_short'] = calculate_sma(data['close'], short_window)
        data['sma_long'] = calculate_sma(data['close'], long_window)
        data['rsi'] = calculate_rsi(data['close'], rsi_period)
        data['stochk'], data['stochd'] = calculate_stochastic(data['high'], data['low'], data['close'], stoch_k, stoch_d)
        data['macd'], data['macdh'], data['macds'] = calculate_macd(data['close'], macd_fast, macd_slow, macd_signal)
        
        # Remove rows with NaN values from indicators
        initial_rows = len(data)
        data.dropna(inplace=True)
        final_rows = len(data)
        
        if final_rows < initial_rows * 0.7:  # If we lost more than 30% of data
            st.warning(f"技術指標の計算により {initial_rows - final_rows} 行のデータが除外されました。")
        
        if data.empty:
            st.error("技術指標の計算後にデータが残りませんでした。期間を長くするか、パラメータを調整してください。")
            st.stop()
        
        # Generate composite signals
        data['composite_signal'] = generate_composite_signal(data)
        
        # Current Status
        st.header("現在のシグナル")
        latest_data = data.iloc[-1]
        
        cols = st.columns(5)
        
        # Current Price
        cols[0].metric("現在価格", f"${latest_data['close']:.2f}")
        
        # SMA Signal
        sma_signal = "ゴールデンクロス (買い)" if latest_data['sma_short'] > latest_data['sma_long'] else "デッドクロス (売り)"
        cols[1].metric("SMAシグナル", sma_signal)

        # RSI Signal
        rsi_val = latest_data['rsi']
        if rsi_val > 70:
            rsi_signal = "買われ過ぎ"
        elif rsi_val < 30:
            rsi_signal = "売られ過ぎ"
        else:
            rsi_signal = "中立"
        cols[2].metric("RSI", f"{rsi_val:.1f}", rsi_signal)

        # Stochastic Signal
        stoch_val = latest_data['stochk']
        if stoch_val > 80:
            stoch_signal = "買われ過ぎ"
        elif stoch_val < 20:
            stoch_signal = "売られ過ぎ"
        else:
            stoch_signal = "中立"
        cols[3].metric("ストキャスティクス %K", f"{stoch_val:.1f}", stoch_signal)

        # Composite Signal
        composite_signal = latest_data['composite_signal']
        cols[4].metric("総合シグナル", composite_signal)

        # Backtesting
        st.header("バックテスト結果")
        
        with st.spinner("バックテストを実行中..."):
            backtest_results = backtest_strategy(data, initial_capital)
        
        if backtest_results:
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "総リターン", 
                f"{backtest_results['total_return']:.2f}%",
                f"{backtest_results['total_return'] - backtest_results['buy_and_hold_return']:.2f}% vs B&H"
            )
            col2.metric("最大ドローダウン", f"{backtest_results['max_drawdown']:.2f}%")
            col3.metric("シャープレシオ", f"{backtest_results['sharpe_ratio']:.2f}")
            col4.metric("取引回数", backtest_results['num_trades'])
            
            # Performance comparison
            st.subheader("戦略 vs バイアンドホールド")
            perf_data = pd.DataFrame({
                '戦略': backtest_results['portfolio_values'],
                'バイアンドホールド': backtest_results['buy_and_hold_values']
            }, index=data.index)
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Scatter(x=perf_data.index, y=perf_data['戦略'], 
                                        mode='lines', name='戦略', line=dict(color='blue')))
            fig_perf.add_trace(go.Scatter(x=perf_data.index, y=perf_data['バイアンドホールド'], 
                                        mode='lines', name='バイアンドホールド', line=dict(color='gray')))
            fig_perf.update_layout(
                title="パフォーマンス比較",
                xaxis_title="日付",
                yaxis_title="ポートフォリオ価値 ($)",
                height=400
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Trade history
            if backtest_results['trades']:
                st.subheader("取引履歴")
                trades_df = pd.DataFrame(backtest_results['trades'])
                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.date
                st.dataframe(trades_df, use_container_width=True)

        # Price Chart & Technical Indicators
        st.header("価格チャートと技術指標")

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=[0.5, 0.15, 0.15, 0.2])

        # Price and SMA with signals
        fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'],
                                     low=data['low'], close=data['close'], name='価格'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_short'], mode='lines', 
                                name=f'SMA {short_window}', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_long'], mode='lines', 
                                name=f'SMA {long_window}', line=dict(color='purple')), row=1, col=1)

        # Add buy/sell signals
        buy_signals = data[data['composite_signal'] == 'BUY']
        sell_signals = data[data['composite_signal'] == 'SELL']
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                                    mode='markers', name='買いシグナル',
                                    marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                                    mode='markers', name='売りシグナル',
                                    marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], mode='lines', name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Stochastic
        fig.add_trace(go.Scatter(x=data.index, y=data['stochk'], mode='lines', name='%K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['stochd'], mode='lines', name='%D'), row=3, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

        # MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['macd'], mode='lines', name='MACD'), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['macds'], mode='lines', name='シグナルライン'), row=4, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['macdh'], name='ヒストグラム'), row=4, col=1)

        # Update layout
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="価格 (USD)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="ストキャスティクス", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Historical Data Table
        st.header("履歴データ")
        display_data = data[['open', 'high', 'low', 'close', 'volume', 
                           'sma_short', 'sma_long', 'rsi', 'stochk', 'macd', 'composite_signal']].tail(20)
        display_data.columns = ['始値', '高値', '安値', '終値', '出来高', 
                              f'SMA{short_window}', f'SMA{long_window}', 'RSI', 'Stoch%K', 'MACD', '総合シグナル']
        st.dataframe(display_data, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Technical analysis error: {str(e)}")
        st.error(f"技術分析の計算中にエラーが発生しました: {str(e)}")

else:
    st.warning("有効なティッカーと日付範囲を選択して分析を開始してください。")

# Footer
st.markdown("---")
st.markdown("⚠️ **免責事項**: このツールは教育目的のみで提供されています。投資判断は自己責任で行ってください。")
