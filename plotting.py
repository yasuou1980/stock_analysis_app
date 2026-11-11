import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from utils import safe_calculate_signal_strength

def plot_performance(data, metrics, results, params):
    latest = data.iloc[-1]
    st.header("📊 現在の市場状況")
    cols = st.columns([1.5, 1.5, 1.5, 2])
    cols[0].metric("現在価格", f"${latest['close']:.2f}")
    cols[1].metric("総合シグナル", latest['composite_signal'])
    cols[2].metric("RSI", f"{latest['rsi']:.1f}")
    with cols[3]: 
        st.markdown("**シグナル強度**")
        st.progress(int(safe_calculate_signal_strength(latest)))

    tab1, tab2, tab3 = st.tabs(["📊 パフォーマンス概要", "⚠️ リスク＆取引分析", "📈 詳細チャート"])
    
    with tab1:
        st.subheader("パフォーマンスサマリー")
        c1, c2, c3 = st.columns(3)
        c1.metric("最終リターン", f"{metrics.get('total_return', 0):.2f}%")
        c2.metric("最大ドローダウン", f"{metrics.get('max_drawdown', 0):.2f}%")
        c3.metric("シャープレシオ", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.line_chart(results['portfolio_values'])

    with tab2:
        st.subheader("リスクプロファイル")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("年率ボラティリティ", f"{metrics.get('volatility', 0):.2f}%")
        c2.metric("VaR 95% (日次)", f"{metrics.get('var_95', 0):.2f}%")
        c3.metric("歪度 (Skewness)", f"{metrics.get('skewness', 0):.2f}", help="0より大きいと右に長い裾野（大きな利益が時々）、小さいと左に長い裾野（大きな損失が時々）")
        c4.metric("尖度 (Kurtosis)", f"{metrics.get('kurtosis', 0):.2f}", help="3より大きいと正規分布より尖った分布（テールリスク大）")
        
        st.subheader("取引詳細分析")
        # ... (trade metrics calculation and display)

    with tab3:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='価格'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_short'], mode='lines', name=f'EMA {params["short_window"]}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_long'], mode='lines', name=f'EMA {params["long_window"]}'), row=1, col=1)
        
        if 'sma200' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['sma200'], mode='lines', name='200日平均', line=dict(dash='dot')), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['bbu'], mode='lines', line=dict(width=0.5), name='BB Upper', line_color='rgba(255,165,0,0.5)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['bbl'], mode='lines', line=dict(width=0.5), name='BB Lower', fill='tonexty', fillcolor='rgba(255,165,0,0.2)', line_color='rgba(255,165,0,0.5)'), row=1, col=1)

        buy_signals = data[data['composite_signal'] == 'BUY']
        sell_signals = data[data['composite_signal'] == 'SELL']
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='買い', marker=dict(symbol='triangle-up', size=10, color='lime')), row=1, col=1)
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='売り', marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)
            
        fig.add_trace(go.Bar(x=data.index, y=data['macdh'], name='MACDヒストグラム', marker_color=np.where(data['macdh'] > 0, 'green', 'tomato')), row=2, col=1)
        
        # Stochastic Oscillator
        fig.add_trace(go.Scatter(x=data.index, y=data['stoch_k'], mode='lines', name='Stoch %K'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['stoch_d'], mode='lines', name='Stoch %D'), row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
