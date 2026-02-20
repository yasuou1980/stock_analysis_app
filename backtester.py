import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

@st.cache_data
def calculate_indicators_and_signals(data_hash, _data, params, strategy_type="トレンドフォロー"):
    """技術指標とシグナルを計算する"""
    data = _data.copy()
    try:
        # 【変更点1】SMA（単純移動平均）から EMA（指数平滑移動平均）に変更して反応速度を向上
        data.ta.ema(length=params['short_window'], append=True, col_names="sma_short") # 変数名は互換性のためsma_shortのまま
        data.ta.ema(length=params['long_window'], append=True, col_names="sma_long")  # 変数名は互換性のためsma_longのまま
        
        data.ta.rsi(length=params['rsi_period'], append=True, col_names="rsi")
        data.ta.macd(fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'], append=True, col_names=("macd", "macdh", "macds"))
        data.ta.bbands(length=params['bb_length'], std=params['bb_std'], append=True, col_names=("bbl", "bbm", "bbu", "bbb", "bbp"))
        
        # ADX (トレンド強度) の計算
        data.ta.adx(length=14, append=True)
        adx_col = next((col for col in data.columns if col.startswith('ADX')), None)

        # 出来高の移動平均 (20日)
        if 'volume' in data.columns:
            data['vol_sma'] = data['volume'].rolling(window=20).mean()
        else:
            data['vol_sma'] = 0

        # ストキャスティクス
        data.ta.stoch(k=params['stoch_k'], d=params['stoch_d'], append=True)
        stoch_k_col = next((col for col in data.columns if col.startswith('STOCHk')), None)
        stoch_d_col = next((col for col in data.columns if col.startswith('STOCHd')), None)
        if stoch_k_col:
            data.rename(columns={stoch_k_col: 'stoch_k'}, inplace=True)
        if stoch_d_col:
            data.rename(columns={stoch_d_col: 'stoch_d'}, inplace=True)

        # 移動平均乖離率
        if 'sma_long' in data.columns and not data['sma_long'].isnull().all() and not (data['sma_long'] == 0).any():
            data['deviation'] = ((data['close'] - data['sma_long']) / data['sma_long']) * 100
        else:
            data['deviation'] = 0

        data['volatility'] = data['close'].pct_change().rolling(window=30).std().bfill()

        if strategy_type == "トレンドフォロー":
            # 1. ADXとDI(方向性)のフィルタ
            adx_score = 0
            if adx_col:
                # pandas_taのADX計算で生成されるDMP(+DI)とDMN(-DI)のカラムを取得
                dmp_col = next((col for col in data.columns if col.startswith('DMP')), None)
                dmn_col = next((col for col in data.columns if col.startswith('DMN')), None)
                
                if dmp_col and dmn_col:
                    # 上昇トレンド: トレンドが強く(ADX>20)、かつ上方向(DMP > DMN)
                    adx_buy = (data[adx_col] > 20) & (data[dmp_col] > data[dmn_col])
                    # 下降トレンド: トレンドが強く(ADX>20)、かつ下方向(DMN > DMP)
                    adx_sell = (data[adx_col] > 20) & (data[dmn_col] > data[dmp_col])
                    adx_score = np.select([adx_buy, adx_sell], [1.5, -1.5], default=0)
                else:
                    adx_score = np.where(data[adx_col] > 20, 1, -1)

            # 2. RSIの順張り的活用 (50以上で上昇モメンタム)
            rsi_score = np.select([data['rsi'] > 50, data['rsi'] < 50], [1.0, -1.0], default=0)

            # 3. スコアの合算
            scores = (
                np.where(data['sma_short'] > data['sma_long'], 2, -2) + 
                rsi_score + 
                np.where(data['macdh'] > 0, 1.5, -1.5) +
                adx_score
            )
            
            # 出来高フィルタ
            vol_condition = (data['volume'] > data['vol_sma']) if 'volume' in data.columns else True

            # トレンドフォローのシグナル判定（スコアの閾値を調整）
            # 最高スコアは 2 + 1.0 + 1.5 + 1.5 = 6.0 になるため、閾値を4.0程度に設定
            buy_signal = (scores >= 4.0) & vol_condition
            
            # 決済/売りシグナル: トレンド崩壊を早めに察知する
            sell_signal = (scores <= -2.0)

            data['composite_signal'] = np.select([buy_signal, sell_signal], ["BUY", "SELL"], default="HOLD")

        elif strategy_type == "逆張り":
            # 1. ストキャスティクスの反転確認 (クロス)
            # %Kが%Dを上抜けた(ゴールデンクロス) かつ 売られすぎ圏にいる
            stoch_buy_cross = (data['stoch_k'] > data['stoch_d']) & (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) & (data['stoch_k'] < params.get('stoch_lower', 20))
            # %Kが%Dを下抜けた(デッドクロス) かつ 買われすぎ圏にいる
            stoch_sell_cross = (data['stoch_k'] < data['stoch_d']) & (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)) & (data['stoch_k'] > params.get('stoch_upper', 80))
            
            # クロスに対する強いスコア
            stoch_cross_score = np.select([stoch_buy_cross, stoch_sell_cross], [2.0, -2.0], default=0)
            # 単なる「状態」に対する弱いスコア
            stoch_state_score = np.select([data['stoch_k'] < params.get('stoch_lower', 20), data['stoch_k'] > params.get('stoch_upper', 80)], [1.0, -1.0], default=0)

            # 2. RSIの反発確認
            # 売られすぎ圏内で、前日よりRSIが上昇した（反発の兆し）
            rsi_rebound_buy = (data['rsi'] < params.get('rsi_lower', 30)) & (data['rsi'] > data['rsi'].shift(1))
            # 買われすぎ圏内で、前日よりRSIが下落した
            rsi_rebound_sell = (data['rsi'] > params.get('rsi_upper', 70)) & (data['rsi'] < data['rsi'].shift(1))
            
            rsi_rebound_score = np.select([rsi_rebound_buy, rsi_rebound_sell], [2.0, -2.0], default=0)
            rsi_state_score = np.select([data['rsi'] < params.get('rsi_lower', 30), data['rsi'] > params.get('rsi_upper', 70)], [1.0, -1.0], default=0)

            # 3. ボリンジャーバンドと移動平均乖離率 (極端な価格の歪み)
            bb_score = np.select([data['close'] < data['bbl'], data['close'] > data['bbu']], [1.5, -1.5], default=0)
            dev_score = np.select([data['deviation'] < params.get('dev_lower', -10), data['deviation'] > params.get('dev_upper', 10)], [1.5, -1.5], default=0)

            # スコアの合算
            counter_scores = stoch_cross_score + stoch_state_score + rsi_rebound_score + rsi_state_score + bb_score + dev_score

            # 閾値の調整 (最高スコアが9.0になるため、複数の条件が重なる5.0を閾値とする)
            data['composite_signal'] = np.select([counter_scores >= 5.0, counter_scores <= -5.0], ["BUY", "SELL"], default="HOLD")

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
    current_entry_price = 0
    stop_loss_pct = -0.05    # 損切りライン (-5%)

    for i in range(len(_data)):
        row = _data.iloc[i]
        current_portfolio_value = portfolio['cash'] + (portfolio['shares'] * row['close'])
        portfolio_values.append(current_portfolio_value)

        # --- 損切りロジック ---
        if portfolio['shares'] > 0 and current_entry_price > 0:
            pnl_pct = (row['close'] - current_entry_price) / current_entry_price
            if pnl_pct <= stop_loss_pct:
                sell_price = row['close'] * (1 - slippage)
                revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
                trades.append({'date': _data.index[i], 'action': 'STOP_LOSS', 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
                portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
                current_entry_price = 0
                continue

        # --- 売買ロジック ---
        if row['composite_signal'] == "BUY" and portfolio['shares'] == 0:
            buy_price = row['close'] * (1 + slippage)
            shares_to_buy = calculate_position_size(buy_price, row['volatility'], current_portfolio_value, position_sizing_strategy, ps_params)
            cost = shares_to_buy * buy_price * (1 + commission_rate)
            if shares_to_buy > 0 and portfolio['cash'] >= cost:
                portfolio.update({'shares': shares_to_buy, 'cash': portfolio['cash'] - cost})
                current_entry_price = buy_price
                trades.append({'date': _data.index[i], 'action': 'BUY', 'shares': shares_to_buy, 'price': buy_price, 'value': cost})
        
        elif row['composite_signal'] == "SELL" and portfolio['shares'] > 0:
            sell_price = row['close'] * (1 - slippage)
            revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
            trades.append({'date': _data.index[i], 'action': 'SELL', 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
            portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
            current_entry_price = 0

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
    sells = trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS'])]
    
    if buys.empty or sells.empty: return {}
    
    min_len = min(len(buys), len(sells))
    hold_periods = [(sells.iloc[i]['date'] - buys.iloc[i]['date']).days for i in range(min_len)]
    
    if not hold_periods: return {}
    return {'avg_hold_days': np.mean(hold_periods), 'max_hold_days': np.max(hold_periods), 'min_hold_days': np.min(hold_periods)}
