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


def _atr(df, length=14):
    """Average True Range - ボラティリティに基づく動的な値幅指標"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    df['atr'] = atr


def _psar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    """Parabolic SAR (Stop and Reverse) - 加速するストップライン指標"""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(close)
    psar = np.zeros(n)
    psar_dir = np.ones(n)  # 1 = uptrend, -1 = downtrend

    if n < 2:
        df['psar'] = psar
        df['psar_dir'] = psar_dir
        return

    # 初期トレンド方向の判定
    if close[1] >= close[0]:
        psar_dir[0] = 1
        psar[0] = low[0]
        ep = high[0]
    else:
        psar_dir[0] = -1
        psar[0] = high[0]
        ep = low[0]

    af = af_start

    for i in range(1, n):
        prev_psar = psar[i - 1]
        prev_dir = psar_dir[i - 1]

        if prev_dir == 1:  # 上昇トレンド
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1])
            if i >= 2:
                psar[i] = min(psar[i], low[i - 2])

            if low[i] < psar[i]:  # トレンド反転 → 下降へ
                psar_dir[i] = -1
                psar[i] = ep
                ep = low[i]
                af = af_start
            else:
                psar_dir[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:  # 下降トレンド
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1])
            if i >= 2:
                psar[i] = max(psar[i], high[i - 2])

            if high[i] > psar[i]:  # トレンド反転 → 上昇へ
                psar_dir[i] = 1
                psar[i] = ep
                ep = high[i]
                af = af_start
            else:
                psar_dir[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    df['psar'] = psar
    df['psar_dir'] = psar_dir


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

        # ATR（動的ストップロス用）
        _atr(data, length=14)

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

            # 8. ベアリッシュダイバージェンス（強化版：MACD + RSIダイバージェンス）
            rolling_max_close = data['close'].rolling(window=15).max().shift(1).fillna(0)
            rolling_max_macdh = data['macdh'].rolling(window=15).max().shift(1).fillna(0)
            rolling_max_rsi = data['rsi'].rolling(window=15).max().shift(1).fillna(100)
            at_price_high = data['close'] >= rolling_max_close * 0.97
            macd_diverging = (data['macdh'] > 0) & (rolling_max_macdh > 0) & (data['macdh'] < rolling_max_macdh * 0.7)
            rsi_diverging = at_price_high & (data['rsi'] > 50) & (data['rsi'] < rolling_max_rsi * 0.9)
            # ダブルダイバージェンス(MACD+RSI) = -6.0、MACD単体 = -4.0、RSI単体 = -3.0
            double_div = at_price_high & macd_diverging & rsi_diverging
            bearish_div_score = np.where(double_div, -6.0,
                                np.where(at_price_high & macd_diverging, -4.0,
                                np.where(rsi_diverging, -3.0, 0)))

            # 9. EMAスロープスコア（トレンドの勢いを反映）
            ema_slope_period = params.get('ema_slope_period', 5)
            ema_slope = (data['sma_short'] - data['sma_short'].shift(ema_slope_period)) / data['sma_short'].shift(ema_slope_period) * 100
            ema_slope = ema_slope.fillna(0)
            ema_slope_score = np.where(ema_slope > 1.0, 1.0, np.where(ema_slope > 0.3, 0.5,
                              np.where(ema_slope < -1.0, -1.0, np.where(ema_slope < -0.3, -0.5, 0))))
            data['ema_slope'] = ema_slope

            # 10. パラボリックSARスコア（加速するストップライン）
            _psar(data)
            psar_bearish_cross = (data['close'] < data['psar']) & (data['close'].shift(1) >= data['psar'].shift(1))
            psar_bullish_cross = (data['close'] > data['psar']) & (data['close'].shift(1) <= data['psar'].shift(1))
            psar_event_score = np.where(psar_bullish_cross, 1.5, np.where(psar_bearish_cross, -3.5, 0))
            psar_state_score = np.where(data['psar_dir'] == 1, 0.5, -0.5)

            # 11. スコア合算
            ema_state_score = np.where(data['sma_short'] > data['sma_long'], 0.5, -0.5)
            macd_hist_score = np.where(data['macdh'] > 0, 1.5, -1.5)

            # 常時スコア（状態ベース：毎日評価）
            state_scores = (ema_state_score + rsi_zone_score + macd_hist_score
                            + adx_score + bb_trend_score + ema_slope_score + psar_state_score)
            # イベントスコア（クロスオーバー等：発火日のみ非ゼロ）
            event_scores = (cross_bonus + breakout_score
                            + macd_zero_cross_score + rsi_reversal_score + bearish_div_score
                            + psar_event_score)

            # イベントスコアをEMA平滑化（数日間持続させる）
            score_smooth_period = params.get('score_smooth_period', 3)
            event_scores_series = pd.Series(event_scores, index=data.index)
            smoothed_events = event_scores_series.ewm(span=score_smooth_period, adjust=False).mean()
            # 平滑化で振幅が縮むため、元のスケールに近づける補正
            smoothed_events = smoothed_events * (score_smooth_period * 0.6)

            scores = state_scores + smoothed_events.values

            # レンジ相場ではスコア半減
            scores = np.where(adx_trend_filter, scores, scores * 0.5)

            data['trend_score'] = scores

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

            # トレンド強度に応じた適応的シグナル閾値
            if adx_col:
                adx_values = data[adx_col].values
                # 強いトレンド(ADX>30): 閾値を下げてエントリーしやすく
                # 通常トレンド(ADX 20-30): 標準閾値
                # 弱いトレンド(ADX<20): スコア半減で自然にフィルタ
                buy_threshold = np.where(adx_values > 30, 3.5,
                                np.where(adx_values > adx_threshold, 4.5, 5.5))
                sell_threshold = np.where(adx_values > 30, -3.0,
                                 np.where(adx_values > adx_threshold, -4.0, -5.0))
            else:
                buy_threshold = 5.5
                sell_threshold = -5.0

            buy_signal  = (scores >= buy_threshold) & vol_condition & mtf_buy_filter
            sell_signal = (scores <= sell_threshold) & vol_condition & mtf_sell_filter

            data['composite_signal'] = np.where(buy_signal, "BUY", np.where(sell_signal, "SELL", "HOLD"))

            # --- エグジット専用シグナル（利確スコア制）---
            exit_score = np.zeros(len(data))

            # 1. 価格とSMAの関係
            price_below_sma = (data['close'] < data['sma_short']) & (data['close'].shift(1) >= data['sma_short'].shift(1))
            sma_slope_negative = ema_slope < 0
            
            # 2. オシレーターのピークアウト検知
            macd_cross_down = (data['macdh'] < 0) & (data['macdh'].shift(1) >= 0)
            # RSIが高値圏（70）から下落し始めた初動を捉える
            rsi_overbought_turn = (data['rsi'].shift(1) >= 70) & (data['rsi'] < 65)
            
            # 3. トレンド系の崩れ
            psar_bearish_cross = (data['close'] < data['psar']) & (data['close'].shift(1) >= data['psar'].shift(1))

            # スコアの加算（シグナルの強さに応じて重み付け）
            exit_score += np.where(price_below_sma, 1.5, 0)
            exit_score += np.where(sma_slope_negative, 1.0, 0)
            exit_score += np.where(macd_cross_down, 1.5, 0)
            exit_score += np.where(psar_bearish_cross, 2.0, 0)
            exit_score += np.where(rsi_overbought_turn, 1.5, 0)
            # すでに計算済みのベアリッシュダイバージェンス（負の値）を利確スコアにも転用
            exit_score += np.where(bearish_div_score <= -4.0, 2.0, 0)

            data['exit_score'] = exit_score
            # バックテスト側で動的判定するため、ここでの exit_signal は一旦 HOLD 固定にするか列自体を作らない
            data['exit_signal'] = "HOLD"
        elif strategy_type == "逆張り":
            stoch_upper = params.get('stoch_upper', 80)
            stoch_lower = params.get('stoch_lower', 20)
            rsi_upper = params.get('rsi_upper', 70)
            rsi_lower = params.get('rsi_lower', 30)

            # --- 方針1: 過熱圏「脱出」型トリガー（早すぎる売りの防止）---

            # 1. ストキャスティクスの反転確認
            # 買い: 売られすぎ圏でのゴールデンクロス（従来通り）
            stoch_buy_cross = ((data['stoch_k'] > data['stoch_d'])
                               & (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1))
                               & (data['stoch_k'] < stoch_lower))
            # 売り: %Kが過熱圏を上から下抜けた瞬間（従来: 過熱圏内のデッドクロス）
            stoch_exit_overbought = ((data['stoch_k'] < stoch_upper)
                                     & (data['stoch_k'].shift(1) >= stoch_upper))
            stoch_cross_score = np.where(stoch_buy_cross, 2.0,
                                np.where(stoch_exit_overbought, -2.0, 0))
            # 状態スコア: 過熱圏にいる間は0（従来: -1.0）→ 脱出時に-1.0
            stoch_state_score = np.where(data['stoch_k'] < stoch_lower, 1.0,
                                np.where(stoch_exit_overbought, -1.0, 0))

            # 2. RSIの反発確認
            # 買い: 売られすぎ圏からの反発（従来通り）
            rsi_rebound_buy = (data['rsi'] < rsi_lower) & (data['rsi'] > data['rsi'].shift(1))
            # 売り: RSIが過熱圏を上から下抜けた瞬間（従来: 過熱圏内で前日比マイナス）
            rsi_exit_overbought = (data['rsi'] < rsi_upper) & (data['rsi'].shift(1) >= rsi_upper)
            rsi_rebound_score = np.where(rsi_rebound_buy, 2.0,
                                np.where(rsi_exit_overbought, -2.0, 0))
            # 状態スコア: 過熱圏にいる間は0（従来: -1.0）→ 脱出時に-1.0
            rsi_state_score = np.where(data['rsi'] < rsi_lower, 1.0,
                              np.where(rsi_exit_overbought, -1.0, 0))

            # 3. ボリンジャーバンドと移動平均乖離率
            bb_score = np.where(data['close'] < data['bbl'], 1.5,
                      np.where(data['close'] > data['bbu'], -1.5, 0))
            dev_score = np.where(data['deviation'] < params.get('dev_lower', -10), 1.5,
                       np.where(data['deviation'] > params.get('dev_upper', 10), -1.5, 0))

            # --- 方針2: 上昇トレンド中の売りスコア減衰（バンドウォーク対策）---
            ema_slope_period_c = params.get('ema_slope_period', 5)
            ema_slope_c = ((data['sma_short'] - data['sma_short'].shift(ema_slope_period_c))
                           / data['sma_short'].shift(ema_slope_period_c) * 100).fillna(0)
            uptrend_momentum = (data['sma_short'] > data['sma_long']) & (ema_slope_c > 0)

            # 売りコンポーネントと買いコンポーネントを分離し、上昇中は売り側を減衰
            all_scores = [stoch_cross_score, stoch_state_score, rsi_rebound_score,
                          rsi_state_score, bb_score, dev_score]
            sell_component = sum(np.minimum(s, 0) for s in all_scores)
            buy_component = sum(np.maximum(s, 0) for s in all_scores)
            dampening = np.where(uptrend_momentum, 0.5, 1.0)
            counter_scores = buy_component + sell_component * dampening

            # --- 方針3: 直近安値ブレイクフィルター（価格ベースの確認）---
            recent_low_5d = data['close'].rolling(window=5).min().shift(1)
            price_break_down = data['close'] < recent_low_5d

            # --- 方針4: モメンタム連動型売り閾値 ---
            macdh_positive = data['macdh'] > 0
            ema_slope_strong = ema_slope_c > 0.3
            sell_threshold_c = np.where(macdh_positive & ema_slope_strong, -7.0,
                               np.where(~macdh_positive | (ema_slope_c < 0), -3.0, -5.0))

            # 最終シグナル判定
            buy_signal_c = counter_scores >= 5.0
            
            # 従来の売り（ダウ理論的な直近安値割れ ＋ スコア低下）
            sell_signal_c = (counter_scores <= sell_threshold_c) & price_break_down

            # ▼ 修正: 逆張りの「利益を伸ばす」利確シグナル
            # 1. 利益確定: 過去5日以内にRSIが過熱圏に達しており、かつ今日「短期SMA」を下抜けた場合
            recently_overbought = data['rsi'].rolling(window=5).max() >= rsi_upper
            sma_break_down = (data['close'] < data['sma_short']) & (data['close'].shift(1) >= data['sma_short'].shift(1))
            profit_take_signal = recently_overbought & sma_break_down
            
            # 2. 早期撤退: エントリー後に勢いがなく、MACDがデッドクロスしてしまった場合
            macd_cross_down_c = (data['macdh'] < 0) & (data['macdh'].shift(1) >= 0)
            
            early_sell_signal_c = profit_take_signal | macd_cross_down_c

            data['counter_score'] = counter_scores
            data['composite_signal'] = np.where(buy_signal_c, "BUY",
                                       np.where(sell_signal_c | early_sell_signal_c, "SELL", "HOLD"))

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
    stop_loss_price = 0         # ATRベースの動的損切り価格
    trailing_stop_price = 0     # ATRベースの価格
    highest_price_since_entry = 0
    pyramid_count = 0           # ピラミッディング実施回数
    max_pyramid_count = 2       # 最大2回の買い増し (初期+2=最大3ポジション層)
    pyramid_threshold = 0.10    # 含み益+10%以上で買い増しを許可
    stop_loss_atr_mult = 2.0    # ストップロス = エントリー価格 - ATR × 2.0
    trailing_atr_mult = 3.0     # トレーリング = 最高値 - ATR × 3.0

    for i in range(len(_data)):
        row = _data.iloc[i]
        current_portfolio_value = portfolio['cash'] + (portfolio['shares'] * row['close'])
        portfolio_values.append(current_portfolio_value)

        # --- ATRベースの動的損切り / ロジック ---
        if portfolio['shares'] > 0 and current_entry_price > 0:
            highest_price_since_entry = max(highest_price_since_entry, row['close'])
            current_atr = row.get('atr', 0)

            # トレーリングストップを最高値に追従して引き上げる
            if current_atr > 0:
                pnl_pct = (highest_price_since_entry - current_entry_price) / current_entry_price
                
                # 含み益とRSI（過熱感）を考慮したタイト化
                if pnl_pct > 0.20 and row.get('rsi', 0) > 70:
                    dynamic_trailing_mult = 1.8  # 大幅な含み益かつ過熱圏ならガチガチに詰める
                elif pnl_pct > 0.10:
                    dynamic_trailing_mult = 2.2
                else:
                    dynamic_trailing_mult = trailing_atr_mult # 初期値 (通常は3.0)
                    
                new_trailing = highest_price_since_entry - current_atr * dynamic_trailing_mult
                trailing_stop_price = max(trailing_stop_price, new_trailing)

            triggered = False
            stop_action = 'STOP_LOSS'
            if stop_loss_price > 0 and row['close'] <= stop_loss_price:
                triggered = True
                stop_action = 'STOP_LOSS'
            elif trailing_stop_price > 0 and row['close'] <= trailing_stop_price:
                triggered = True
                stop_action = 'TRAILING_STOP'

            if triggered:
                sell_price = row['close'] * (1 - slippage)
                revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
                trades.append({'date': _data.index[i], 'action': stop_action, 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
                portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
                current_entry_price = 0
                highest_price_since_entry = 0
                stop_loss_price = 0
                trailing_stop_price = 0
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
                    # ATRベースのストップロス価格を設定
                    current_atr = row.get('atr', 0)
                    if current_atr > 0:
                        stop_loss_price = buy_price - current_atr * stop_loss_atr_mult
                        trailing_stop_price = buy_price - current_atr * trailing_atr_mult
                    else:
                        stop_loss_price = buy_price * 0.95  # フォールバック: -5%
                        trailing_stop_price = buy_price * 0.92
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

        # (中略: ストップロス等の処理の後)

        elif row['composite_signal'] == "SELL" and portfolio['shares'] > 0:
            # 従来の戦略ベースの売りシグナル（トレンド転換など）は全決済
            sell_price = row['close'] * (1 - slippage)
            revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
            trades.append({'date': _data.index[i], 'action': 'SELL', 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
            portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
            current_entry_price = 0
            highest_price_since_entry = 0
            stop_loss_price = 0
            trailing_stop_price = 0
            pyramid_count = 0

        elif portfolio['shares'] > 0 and 'exit_score' in _data.columns:
            # --- 攻めの売り（利確スコアと含み益に応じた動的エグジット） ---
            pnl_pct = (row['close'] - current_entry_price) / current_entry_price if current_entry_price > 0 else 0
            exit_score = row['exit_score']

            # 含み益に応じた閾値の動的変化
            if pnl_pct >= 0.15:
                # 利益が15%以上乗っている → 利益保全を優先（少しの失速で売る）
                exit_threshold = 2.5
                partial_threshold = 1.5
            elif pnl_pct >= 0.08:
                # 利益が8%以上 → 中程度の警戒度
                exit_threshold = 3.5
                partial_threshold = 2.5
            elif pnl_pct > 0.02:
                # 微益 → だましを避けるため厳しめに
                exit_threshold = 4.5
                partial_threshold = 99.0 # 部分利確はしない
            else:
                # 含み損、または建値付近 → 利確ロジックは発動しない（ストップロスに任せる）
                exit_threshold = 99.0
                partial_threshold = 99.0

            # 判定と実行
            if exit_score >= exit_threshold:
                # 閾値を完全に超えたら全決済（利確）
                sell_price = row['close'] * (1 - slippage)
                revenue = portfolio['shares'] * sell_price * (1 - commission_rate)
                trades.append({'date': _data.index[i], 'action': 'TAKE_PROFIT', 'shares': portfolio['shares'], 'price': sell_price, 'value': revenue})
                portfolio.update({'shares': 0, 'cash': portfolio['cash'] + revenue})
                current_entry_price = 0
                highest_price_since_entry = 0
                stop_loss_price = 0
                trailing_stop_price = 0
                pyramid_count = 0

            elif exit_score >= partial_threshold and portfolio['shares'] > 1:
                # 警戒水準に達したら 50% だけ部分利確
                # ※ 同じ価格帯で何度も部分利確しないためのフラグ管理（pyramid_countをマイナスにする等）を入れるとより安全です
                shares_to_sell = portfolio['shares'] / 2.0
                sell_price = row['close'] * (1 - slippage)
                revenue = shares_to_sell * sell_price * (1 - commission_rate)
                trades.append({'date': _data.index[i], 'action': 'PARTIAL_TAKE_PROFIT', 'shares': shares_to_sell, 'price': sell_price, 'value': revenue})
                portfolio.update({'shares': portfolio['shares'] - shares_to_sell, 'cash': portfolio['cash'] + revenue})
                
                # 建値は変わらないが、ストップを建値まで引き上げる（負けを無くす）
                stop_loss_price = max(stop_loss_price, current_entry_price)
                # 一度部分利確したら閾値をリセット（連続発動防止）
                pyramid_count = 99

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
    trade_returns = []
    for i in range(min_len):
        buy_date = pd.to_datetime(buys.iloc[i]['date'])
        sell_date = pd.to_datetime(sells.iloc[i]['date'])
        hold_periods.append((sell_date - buy_date).days)
        # 各トレードの損益率を計算
        buy_price = buys.iloc[i]['price']
        sell_price = sells.iloc[i]['price']
        if buy_price > 0:
            trade_returns.append((sell_price - buy_price) / buy_price)

    if not hold_periods: return {}

    metrics = {
        'avg_hold_days': np.mean(hold_periods),
        'max_hold_days': np.max(hold_periods),
        'min_hold_days': np.min(hold_periods),
    }

    # 勝率・プロフィットファクター・平均損益
    if trade_returns:
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        total_trades = len(trade_returns)
        metrics['total_trades'] = total_trades
        metrics['win_rate'] = len(wins) / total_trades * 100 if total_trades > 0 else 0
        metrics['avg_win'] = np.mean(wins) * 100 if wins else 0
        metrics['avg_loss'] = np.mean(losses) * 100 if losses else 0
        # プロフィットファクター = 総利益 / 総損失
        total_profit = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0
        metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        # 最大連勝・連敗
        streaks = []
        current_streak = 0
        for r in trade_returns:
            if r > 0:
                current_streak = max(current_streak + 1, 1) if current_streak >= 0 else 1
            else:
                current_streak = min(current_streak - 1, -1) if current_streak <= 0 else -1
            streaks.append(current_streak)
        metrics['max_win_streak'] = max(streaks) if streaks else 0
        metrics['max_loss_streak'] = abs(min(streaks)) if streaks else 0
        # ストップロス / 発動回数
        metrics['stop_loss_count'] = len(trades_df[trades_df['action'] == 'STOP_LOSS'])
        metrics['trailing_stop_count'] = len(trades_df[trades_df['action'] == 'TRAILING_STOP'])

    return metrics
