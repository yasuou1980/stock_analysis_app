import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

@st.cache_data
def calculate_indicators_and_signals(data_hash, _data, params):
    """技術指標とシグナルを計算する"""
    data = _data.copy()
    try:
        strategy = ta.Strategy(name="Custom Strategy", ta=[
            {"kind": "sma", "length": params['short_window'], "col_names": "sma_short"},
            {"kind": "sma", "length": params['long_window'], "col_names": "sma_long"},
            {"kind": "rsi", "length": params['rsi_period'], "col_names": "rsi"},
            {"kind": "macd", "fast": params['macd_fast'], "slow": params['macd_slow'], 
             "signal": params['macd_signal'], "col_names": ("macd", "macdh", "macds")},
            {"kind": "bbands", "length": params['bb_length'], "std": params['bb_std'], "col_names": ("bbl", "bbm", "bbu", "bbb", "bbp")},
            {"kind": "stoch", "k": params['stoch_k'], "d": params['stoch_d'], "col_names": ("stoch_k", "stoch_d")}
        ])
        data.ta.strategy(strategy)
        data['volatility'] = data['close'].pct_change().rolling(window=30).std().bfill()
        scores = (
            np.where(data['sma_short'] > data['sma_long'], 2, -2) + 
            np.select([data['rsi'] < 30, data['rsi'] > 70], [1.5, -1.5], default=0) + 
            np.where(data['macdh'] > 0, 2, -2)
        )
        data['composite_signal'] = np.select([scores >= 3.5, scores <= -3.5], ["BUY", "SELL"], default="HOLD")
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"指標計算中にエラーが発生しました: {e}")
        logger.error(f"Indicator calculation error: {e}")
        return _data.copy()

def calculate_position_size(price, volatility, portfolio_value, strategy, params):
    if price <= 0: return 0
    if strategy == "固定リスク率 (Volatility Adjusted)":
        if volatility == 0: return 0
        target_risk = params['target_risk']
        max_position_ratio = params['max_position_ratio']
        size_in_currency = (portfolio_value * target_risk) / volatility
        return min(size_in_currency / price, (portfolio_value * max_position_ratio) / price)
    elif strategy == "固定ポートフォリオ比率":
        fixed_ratio = params['fixed_ratio']
        return (portfolio_value * fixed_ratio) / price
    return 0

@st.cache_data
def backtest_strategy(data_hash, _data, initial_capital, commission_rate, slippage, position_sizing_strategy, ps_params):
    """バックテスト実行"""
    portfolio = {'cash': initial_capital, 'shares': 0}
    portfolio_values = []
    trades = []
    for i in range(len(_data)):
        row = _data.iloc[i]
        current_portfolio_value = portfolio['cash'] + (portfolio['shares'] * row['close'])
        portfolio_values.append(current_portfolio_value)
        if row['composite_signal'] == "BUY" and portfolio['shares'] == 0:
            buy_price = row['close'] * (1 + slippage)
            shares_to_buy = calculate_position_size(buy_price, row['volatility'], current_portfolio_value, position_sizing_strategy, ps_params)
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

def calculate_performance_metrics(_portfolio_values, _dates):
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
    return {'total_return': total_return, 'max_drawdown': max_dd, 'sharpe_ratio': sharpe, 'volatility': volatility, 'var_95': var_95, 'skewness': skew, 'kurtosis': kurt}

def calculate_trade_metrics(trades_df):
    if trades_df.empty: return {}
    buys = trades_df[trades_df['action'] == 'BUY']
    sells = trades_df[trades_df['action'] == 'SELL']
    if buys.empty or sells.empty: return {}
    hold_periods = [(sells.iloc[i]['date'] - buys.iloc[i]['date']).days for i in range(min(len(buys), len(sells)))]
    if not hold_periods: return {}
    return {'avg_hold_days': np.mean(hold_periods), 'max_hold_days': np.max(hold_periods), 'min_hold_days': np.min(hold_periods)}
