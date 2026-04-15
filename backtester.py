import streamlit as st
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# --- Pure pandas/numpy technical indicator implementations ---
# Replaces pandas-ta to avoid numba dependency (incompatible with Python 3.14)

def _ema(series, length):
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series, length=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(series, fast=12, slow=26, signal=9):
    """MACD (Moving Average Convergence Divergence)
    Returns DataFrame with columns: macd, macdh (histogram), macds (signal)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'macdh': histogram, 'macds': signal_line}, index=series.index)


def _bbands(series, length=20, std=2.0):
    """Bollinger Bands
    Returns DataFrame with columns: bbl, bbm, bbu, bbb (bandwidth), bbp (percent)
    """
    bbm = series.rolling(window=length).mean()
    bb_std = series.rolling(window=length).std()
    bbu = bbm + std * bb_std
    bbl = bbm - std * bb_std
    bbb = (bbu - bbl) / bbm  # bandwidth
    bbp = (series - bbl) / (bbu - bbl)  # percent b
    return pd.DataFrame({'bbl': bbl, 'bbm': bbm, 'bbu': bbu, 'bbb': bbb, 'bbp': bbp}, index=series.index)


def _adx(df, length=14):
    """Average Directional Index with +DI / -DI
    Appends ADX_{length}, DMP_{length}, DMN_{length} columns to df.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0.0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr = true_range.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()

    df[f'ADX_{length}'] = adx
    df[f'DMP_{length}'] = plus_di
    df[f'DMN_{length}'] = minus_di


def _stoch(df, k=14, d=3):
    """Stochastic Oscillator
    Appends stoch_k, stoch_d columns to df.
    """
    lowest_low = df['low'].rolling(window=k).min()
    highest_high = df['high'].rolling(window=k).max()
    stoch_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d).mean()
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d


@st.cache_data
def calculate_indicators_and_signals(data_hash, _data, params, strategy_type="トレンドフォロー"):
    """技術指標とシグナルを計算する"""
    data = _data.copy()
    try:
        # EMA（指数平滑移動平均）
        data['sma_short'] = _ema(data['close'], length=params['short_window'])
        data['sma_long']  = _ema(data['close'], length=params['long_window'])

        # RSI
        data['rsi'] = _rsi(data['close'], length=params['rsi_period'])

        # MACD
        macd_df = _macd(data['close'], fast=params['macd_fast'], slow=params['macd_slow'], signal=params['macd_signal'])
        data['macd']  = macd_df['macd']
        data['macdh'] = macd_df['macdh']
        data['macds'] = macd_df['macds']

        # ボリンジャーバンド
        bb_df = _bbands(data['close'], length=params['bb_length'], std=params['bb_std'])
        data['bbl'] = bb_df['bbl']
        data['bbm'] = bb_df['bbm']
        data['bbu'] = bb_df['bbu']
        data['bbb'] = bb_df['bbb']
        data['bbp'] = bb_df['bbp']

        # ADX (トレンド強度)
        _adx(data, length=14)
        adx_col = next((col for col in data.columns if col.startswith('ADX')), None)

        # 出来高の移動平均 (20日)
        if 'volume' in data.columns:
            data['vol_sma'] = data['volume'].rolling(window=20).mean()
        else:
            data['vol_sma'] = 0

        # ストキャスティクス
        _stoch(data, k=params['stoch_k'], d=params['stoch_d'])

        # 移動平均乖離率
        if 'sma_long' in data.columns and not data['sma_long'].isnull().all() and not (data['sma_long'] == 0).any():
            data['deviation'] = ((data['close'] - data['sma_long']) / data['sma_long']) * 100
        else:
            data['deviation'] = 0

        data['volatility'] = data['close'].pct_change().rolling(window=30).std().bfill(limit=5)

        if strategy_type == "トレンドフォロー":
            adx_threshold = params.get('adx_threshold', 20)

            # 1. ADX+DI スコア & レンジ相場フィルタ
            adx_score = 0
            adx_trend_filter = np.ones(len(data), dtype=bool)  
            if adx_col:
                dmp_col = next((col for col in data.columns if col.startswith('DMP')), None)
                dmn_col = next((col for col in data.columns if col.startswith('DMN')), None)

                if dmp_col and dmn_col:
                    adx_buy = (data[adx_col] > adx_threshold) & (data[dmp_col] > data[dmn_col])
                    adx_sell = (data[adx_col] > adx_threshold) & (data[dmn_col] > data[dmp_col])
                    adx_score = np.where(adx_buy, 1.5, np.where(adx_sell, -1.5, 0))
                else:
                    adx_score = np.where(data[adx_col] > adx_threshold, 1.0, -1.0)

                adx_trend_filter = (data[adx_col] >= adx_threshold).values

            # 2. RSI ゾーンスコア
            rsi_zone_score = np.where(data['rsi'] > 50, 0.5, np.where(data['rsi'] < 50, -0.5, 0))

            # 3. EMA クロスイベントボーナス
            ema_cross_up   = (data['sma_short'] > data['sma_long']) & (data['sma_short'].shift(1) <= data['sma_long'].shift(1))
            ema_cross_down = (data['sma_short'] < data['sma_long']) & (data['sma_short'].shift(1) >= data['sma_long'].shift(1))
            cross_bonus = np.where(ema_cross_up, 1.0, np.where(ema_cross_down, -1.0, 0))

            # 4. ブレイクアウトスコア
            data['recent_high'] = data['close'].rolling(window=20).max().shift(1)
            data['recent_low']  = data['close'].rolling(window=20).min().shift(1)
            breakout_score = np.where(data['close'] > data['recent_high'], 1.0, np.where(data['close'] < data['recent_low'], -1.0, 0))

            # 5. ボリンジャーバンドブレイクアウトスコア
            bb_trend_score = np.where(data['close'] > data['bbu'], 0.5, np.where(data['close'] < data['bbl'], -0.5, 0))

            # 6. MACDゼロライン越え
            macd_zero_cross_up   = (data['macd'] > 0) & (data['macd'].shift(1) <= 0)
            macd_zero_cross_down = (data['macd'] < 0) & (data['macd'].shift(1) >= 0)
            macd_zero_cross_score = np.where(macd_zero_cross_up, 2.0, np.where(macd_zero_cross_down, -2.0, 0))

            # 7. RSI 底・天井からの反転確認
            rsi_cross_up35   = (data['rsi'] > 35) & (data['rsi'].shift(1) <= 35)
            rsi_cross_down65 = (data['rsi'] < 65) & (data['rsi'].shift(1) >= 65)
            rsi_reversal_score = np.where(rsi_cross_up35, 2.0, np.where(rsi_cross_down65, -2.0, 0))

            # 8. ベアリッシュダイバージェンス
            rolling_max_close = data['close'].rolling(window=15).max().shift(1).fillna(0)
            rolling_max_macdh = data['macdh'].rolling(window=15).max().shift(1).fillna(0)
            at_price_high  = data['close'] >= rolling_max_close * 0.97
            macd_diverging = (data['macdh'] > 0) & (rolling_max_macdh > 0) & (data['macdh'] < rolling_max_macdh * 0.7)
            bearish_div_score = np.where(at_price_high & macd_diverging, -1.5, 0)

            # 9. スコア合算
            ema_state_score = np.where(data['sma_short'] > data['sma_long'], 0.5, -0.5)
            macd_hist_score = np.where(data['macdh'] > 0, 1.5, -1.5)
            scores = (ema_state_score + rsi_zone_score + macd_hist_score + adx_score
                      + cross_bonus + breakout_score + bb_trend_score
                      + macd_zero_cross_score + rsi_reversal_score + bearish_div_score)

            # レンジ相場ではスコア半減
            scores = np.where(adx_trend_filter, scores, scores * 0.5)

            # 出来高フィルタ（条件を1.2倍に緩和し、シグナルを発生しやすく調整）
            if 'volume' in data.columns:
                vol_condition = (data['volume'] > data['vol_sma'] * 1.2).values
            else:
                vol_condition = np.ones(len(data), dtype=bool)

            # マルチタイムフレームフィルタ
            try:
                wk_short = max(params['short_window'] // 5, 2)
                wk_long  = max(params['long_window'] // 5, 3)
                weekly_close = data['close'].resample('W').last().dropna()
                if len(weekly_close) >= max(wk_long, 5):
                    weekly_ema_s = weekly_close.ewm(span=wk_short, adjust=False).mean()
                    weekly_ema_l = weekly_close.ewm(span=wk_long,  adjust=False).mean()
                    weekly_bull = (weekly_ema_s > weekly_ema_l)
                    data['weekly_trend_up'] = weekly_bull.reindex(data.index, method='ffill').fillna(False)
                else:
                    data['weekly_trend_up'] = True
            except Exception:
                data['weekly_trend_up'] = True

            mtf_buy_filter  = data['weekly_trend_up'].values
            mtf_sell_filter = ~data['weekly_trend_up'].values

            # シグナルの閾値を 6.0 -> 5.5 に緩和
            buy_signal  = (scores >= 5.5) & vol_condition & mtf_buy_filter
            sell_signal = (scores <= -5.0) & vol_condition & mtf_sell_filter

            data['composite_signal'] = np.where(buy_signal, "BUY", np.where(sell_signal, "SELL", "HOLD"))

        elif strategy_type == "逆張り":
            # 1. ストキャスティクスの反転確認
            stoch_buy_cross = (data['stoch_k'] > data['stoch_d']) & (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) & (data['stoch_k'] < params.get('stoch_lower', 20))
            stoch_sell_cross = (data['stoch_k'] < data['stoch_d']) & (data['stoch_k'].shift(1) >= data['stoch_d'].shift(1)) & (data['stoch_k'] > params.get('stoch_upper', 80))
            
            stoch_cross_score = np.where(stoch_buy_cross, 2.0, np.where(stoch_sell_cross, -2.0, 0))
            stoch_state_score = np.where(data['stoch_k'] < params.get('stoch_lower', 20), 1.0, np.where(data['stoch_k'] > params.get('stoch_upper', 80), -1.0, 0))

            # 2. RSIの反発確認
            rsi_rebound_buy = (data['rsi'] < params.get('rsi_lower', 30)) & (data['rsi'] > data['rsi'].shift(1))
            rsi_rebound_sell = (data['rsi'] > params.get('rsi_upper', 70)) & (data['rsi'] < data['rsi'].shift(1))
            
            rsi_rebound_score = np.where(rsi_rebound_buy, 2.0, np.where(rsi_rebound_sell, -2.0, 0))
            rsi_state_score = np.where(data['rsi'] < params.get('rsi_lower', 30), 1.0, np.where(data['rsi'] > params.get('rsi_upper', 70), -1.0, 0))

            # 3. ボリンジャーバンドと移動平均乖離率
            bb_score = np.where(data['close'] < data['bbl'], 1.5, np.where(data['close'] > data['bbu'], -1.5, 0))
            dev_score = np.where(data['deviation'] < params.get('dev_lower', -10), 1.5, np.where(data['deviation'] > params.get('dev_upper', 10), -1.5, 0))

            # スコア合算とシグナル判定
            counter_scores = stoch_cross_score + stoch_state_score + rsi_rebound_score + rsi_state_score + bb_score + dev_score
            data['composite_signal'] = np.where(counter_scores >= 5.0, "BUY", np.where(counter_scores <= -5.0, "SELL", "HOLD"))

        data.dropna(inplace=True)
        return data
    except (KeyError, ValueError) as e:
        st.error(f"指標計算中にパラメータエラーが発生しました: {e}")
        logger.error(f"Indicator calculation parameter error: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"指標計算中に予期しないエラーが発生しました: {e}")
        logger.error(f"Indicator calculation unexpected error: {e}", exc_info=True)
        return pd.DataFrame()

def calculate_position_size(price, volatility, portfolio_value, strategy, params):
    if price <= 0 or portfolio_value <= 0: return 0
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
    stop_loss_pct = -0.05       # 固定損切りライン (-5%)
    trailing_stop_pct = 0.08    # トレーリングストップ (高値から-8%)
    highest_price_since_entry = 0
    pyramid_count = 0           # ピラミッディング実施回数
    max_pyramid_count = 2       # 最大2回の買い増し (初期+2=最大3ポジション層)
    pyramid_threshold = 0.10    # 含み益+10%以上で買い増しを許可

    for i in range(len(_data)):
        row = _data.iloc[i]
        current_portfolio_value = portfolio['cash'] + (portfolio['shares'] * row['close'])
        portfolio_values.append(current_portfolio_value)

        # --- 損切り / トレーリングストップロジック ---
        if portfolio['shares'] > 0 and current_entry_price > 0:
            highest_price_since_entry = max(highest_price_since_entry, row['close'])
            pnl_pct = (row['close'] - current_entry_price) / current_entry_price
            trailing_drawdown = (row['close'] - highest_price_since_entry) / highest_price_since_entry

            triggered = False
            stop_action = 'STOP_LOSS'
            if pnl_pct <= stop_loss_pct:
                triggered = True
                stop_action = 'STOP_LOSS'
            elif trailing_drawdown <= -trailing_stop_pct:
                triggered = True
                stop_action = 'TRAILING_STOP'

            if triggered:
                sell_price = row['close'] * (1 - slippage)
                revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
                trades.append({'date': _data.index[i], 'action': stop_action, 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
                portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
                current_entry_price = 0
                highest_price_since_entry = 0
                pyramid_count = 0
                continue

        # --- 売買ロジック ---
        if row['composite_signal'] == "BUY":
            if portfolio['shares'] == 0:
                buy_price = row['close'] * (1 + slippage)
                shares_to_buy = calculate_position_size(buy_price, row['volatility'], current_portfolio_value, position_sizing_strategy, ps_params)
                cost = shares_to_buy * buy_price * (1 + commission_rate)
                if shares_to_buy > 0 and portfolio['cash'] >= cost:
                    portfolio.update({'shares': shares_to_buy, 'cash': portfolio['cash'] - cost})
                    current_entry_price = buy_price
                    highest_price_since_entry = buy_price
                    pyramid_count = 0
                    trades.append({'date': _data.index[i], 'action': 'BUY', 'shares': shares_to_buy, 'price': buy_price, 'value': cost})
            elif pyramid_count < max_pyramid_count:
                pnl_pct = (row['close'] - current_entry_price) / current_entry_price
                if pnl_pct >= pyramid_threshold:
                    buy_price = row['close'] * (1 + slippage)
                    shares_to_add = calculate_position_size(buy_price, row['volatility'], current_portfolio_value, position_sizing_strategy, ps_params) * 0.5
                    cost = shares_to_add * buy_price * (1 + commission_rate)
                    if shares_to_add > 0 and portfolio['cash'] >= cost:
                        total_shares = portfolio['shares'] + shares_to_add
                        current_entry_price = (portfolio['shares'] * current_entry_price + shares_to_add * buy_price) / total_shares
                        portfolio.update({'shares': total_shares, 'cash': portfolio['cash'] - cost})
                        pyramid_count += 1
                        trades.append({'date': _data.index[i], 'action': 'PYRAMID_ADD', 'shares': shares_to_add, 'price': buy_price, 'value': cost})

        elif row['composite_signal'] == "SELL" and portfolio['shares'] > 0:
            sell_price = row['close'] * (1 - slippage)
            revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
            trades.append({'date': _data.index[i], 'action': 'SELL', 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
            portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
            current_entry_price = 0
            highest_price_since_entry = 0
            pyramid_count = 0

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
    sells = trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS', 'TRAILING_STOP'])]
    
    if buys.empty or sells.empty: return {}
    
    min_len = min(len(buys), len(sells))
    
    # 日付型への確実な変換（エラー回避）
    hold_periods = []
    for i in range(min_len):
        buy_date = pd.to_datetime(buys.iloc[i]['date'])
        sell_date = pd.to_datetime(sells.iloc[i]['date'])
        hold_periods.append((sell_date - buy_date).days)
        
    if not hold_periods: return {}
    return {'avg_hold_days': np.mean(hold_periods), 'max_hold_days': np.max(hold_periods), 'min_hold_days': np.min(hold_periods)}
