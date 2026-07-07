"""シグナルゲート (商品クラス別の負けパターン遮断) の回帰テスト

実測データ (results/signals_history.csv) の検証に基づくルールが
意図通りに機能することを保証する。ルールの根拠は
docs/signal_accuracy_2026-07.md を参照。
"""
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtester import (  # noqa: E402
    TICKER_CLASS_INVERSE_LEV,
    TICKER_CLASS_LONG_LEV,
    TICKER_CLASS_PLAIN,
    calculate_indicators_and_signals,
    compute_signal_gates,
    resolve_ticker_class,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def gate_frame(close, rsi=55.0, deviation=5.0):
    """compute_signal_gates 単体テスト用の最小 DataFrame"""
    close = np.asarray(close, dtype=float)
    n = len(close)
    return pd.DataFrame({
        "close": close,
        "rsi": np.full(n, float(rsi)),
        "deviation": np.full(n, float(deviation)),
    })


def make_ohlcv(closes, volumes=None):
    """統合テスト用の OHLCV DataFrame"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if volumes is None:
        volumes = np.where(np.arange(n) % 3 == 0, 2_000_000, 1_000_000)
    idx = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "volume": volumes,
    }, index=idx)


def default_params(**overrides):
    """batch_runner.py と同じデフォルトパラメータ"""
    params = {
        "short_window": 10, "long_window": 40, "rsi_period": 10,
        "macd_fast": 10, "macd_slow": 20, "macd_signal": 7,
        "bb_length": 20, "bb_std": 2.0, "stoch_k": 14, "stoch_d": 3,
        "dev_upper": 10, "dev_lower": -10,
        "rsi_upper": 70, "rsi_lower": 30,
        "stoch_upper": 80, "stoch_lower": 20,
        "score_smooth_period": 3, "ema_slope_period": 5,
    }
    params.update(overrides)
    return params


def run_signals(df, params, strategy):
    """キャッシュ衝突を避けるため一意なハッシュで呼び出す"""
    return calculate_indicators_and_signals(uuid.uuid4().hex, df, params, strategy)


# ---------------------------------------------------------------------------
# compute_signal_gates 単体
# ---------------------------------------------------------------------------
def test_inverse_lev_blocks_all_buys():
    """インバース型レバETFは構造的減価があるため BUY 禁止 (実測 勝率0-19%)"""
    df = gate_frame(np.full(30, 100.0))
    gates = compute_signal_gates(df, TICKER_CLASS_INVERSE_LEV)
    assert not gates["buy_ok"].any()
    # 減価継続の売りは常に許可 (実測 10日勝率75%)
    assert gates["trend_sell_ok"].all()


def test_long_lev_blocks_sells():
    """ロング型レバETFは急落後の反発が大きいため SELL 禁止 (実測 勝率7-18%)"""
    df = gate_frame(np.full(30, 100.0))
    gates = compute_signal_gates(df, TICKER_CLASS_LONG_LEV)
    assert gates["buy_ok"].all()
    assert not gates["trend_sell_ok"].any()
    assert not gates["counter_sell_ok"].any()


def test_plain_crash_cooldown_blocks_trend_sell():
    """5日で-12%超の急落直後は投げ売りの底になりやすく新規SELL禁止"""
    flat = np.full(30, 100.0)
    crash = 100.0 * np.array([1.0, 0.95, 0.90, 0.87, 0.85, 0.84])
    closes = np.concatenate([flat, crash])
    df = gate_frame(closes)
    gates = compute_signal_gates(df, TICKER_CLASS_PLAIN)
    assert gates["trend_sell_ok"][10]          # 平常時は許可
    assert not gates["trend_sell_ok"][-1]      # 5日で-16%の直後は禁止
    assert gates["buy_ok"].all()               # BUY は制限しない


@pytest.mark.parametrize("rsi,dev,expected", [
    (55.0, 5.0, True),     # 強さが残る状態からの反落 → 許可
    (45.0, 5.0, False),    # RSI<50 = すでに売られた後 → 禁止
    (55.0, -5.0, False),   # 乖離マイナス圏 = 下げた後の売り → 禁止
    (55.0, 20.0, False),   # 乖離+15%超 = 強モメンタム (踏み上げ) → 禁止
])
def test_counter_sell_gate_conditions(rsi, dev, expected):
    """逆張りSELLは RSI>=50 かつ 乖離率-3〜+15% のみ許可 (実測 勝率32%→45%)"""
    df = gate_frame(np.full(30, 100.0), rsi=rsi, deviation=dev)
    gates = compute_signal_gates(df, TICKER_CLASS_PLAIN)
    assert bool(gates["counter_sell_ok"][-1]) is expected


def test_counter_sell_gate_crash_cooldown():
    """急落直後 (5日で-10%超) は逆張りSELLも禁止 (実測 勝率12-17%)"""
    closes = np.concatenate([np.full(30, 100.0), [95.0, 92.0, 90.0, 89.0, 88.0]])
    df = gate_frame(closes, rsi=60.0, deviation=5.0)
    gates = compute_signal_gates(df, TICKER_CLASS_PLAIN)
    assert not gates["counter_sell_ok"][-1]


# ---------------------------------------------------------------------------
# resolve_ticker_class
# ---------------------------------------------------------------------------
def test_dip_buy_gate_is_plain_only():
    """押し目リバウンドBUYは plain のみ (long_lev は実測 10日勝率18.8%)"""
    df = gate_frame(np.full(30, 100.0))
    assert compute_signal_gates(df, TICKER_CLASS_PLAIN)["dip_buy_ok"].all()
    assert not compute_signal_gates(df, TICKER_CLASS_LONG_LEV)["dip_buy_ok"].any()
    assert not compute_signal_gates(df, TICKER_CLASS_INVERSE_LEV)["dip_buy_ok"].any()


def test_resolve_ticker_class():
    config = {"ticker_classes": {
        "inverse_lev": ["SOXS", "SQQQ"],
        "long_lev": ["TQQQ"],
    }}
    assert resolve_ticker_class("SOXS", config) == "inverse_lev"
    assert resolve_ticker_class("TQQQ", config) == "long_lev"
    assert resolve_ticker_class("NVDA", config) == TICKER_CLASS_PLAIN
    assert resolve_ticker_class("NVDA", {}) == TICKER_CLASS_PLAIN
    assert resolve_ticker_class("NVDA", None) == TICKER_CLASS_PLAIN


# ---------------------------------------------------------------------------
# calculate_indicators_and_signals 統合
# ---------------------------------------------------------------------------
def _strong_uptrend_closes(n=300, seed=7):
    """BUYシグナルが出やすい強い上昇トレンドの合成データ"""
    rng = np.random.default_rng(seed)
    drift = 0.005 + rng.normal(0, 0.012, n)
    return 100.0 * np.cumprod(1 + drift)


@pytest.mark.parametrize("strategy", ["トレンドフォロー", "逆張り", "レジーム切替"])
def test_strategies_run_and_emit_valid_signals(strategy):
    """3戦略とも例外なく実行でき、シグナル値が正しい語彙に収まる"""
    df = make_ohlcv(_strong_uptrend_closes())
    data = run_signals(df, default_params(), strategy)
    assert not data.empty
    assert set(data["composite_signal"].unique()) <= {"BUY", "SELL", "HOLD"}


def test_uptrend_produces_buy_for_plain_but_not_inverse():
    """同一データでも inverse_lev クラスでは BUY が一切出ない"""
    df = make_ohlcv(_strong_uptrend_closes())

    plain = run_signals(df.copy(), default_params(ticker_class=TICKER_CLASS_PLAIN),
                        "トレンドフォロー")
    inverse = run_signals(df.copy(), default_params(ticker_class=TICKER_CLASS_INVERSE_LEV),
                          "トレンドフォロー")

    assert (plain["composite_signal"] == "BUY").any(), \
        "前提条件エラー: plain で BUY が出ない合成データではテストにならない"
    assert not (inverse["composite_signal"] == "BUY").any()


def test_dip_rebound_buy_fires_for_plain_only():
    """上昇トレンド内の押し目 (dd20<=-8% & 長期MA上維持 & 反発初動) で
    逆張り戦略が BUY を出す。long_lev クラスでは同一データでも出ない。"""
    rng = np.random.default_rng(21)
    # 強い上昇 (長期MAとの乖離を+15%超に積み上げる) → 5日で約-10%の押し目 → 反発
    # 乖離の貯金があるため押し目後も deviation>=0 を維持する = 質の高い押し目
    up = 100.0 * np.cumprod(1 + 0.009 + rng.normal(0, 0.003, 150))
    dip = up[-1] * np.cumprod(np.full(5, 0.98))
    rebound = dip[-1] * np.cumprod(np.full(4, 1.01))
    df = make_ohlcv(np.concatenate([up, dip, rebound]))

    plain = run_signals(df.copy(), default_params(ticker_class=TICKER_CLASS_PLAIN),
                        "逆張り")
    tail = plain["composite_signal"].iloc[-9:]
    assert (tail == "BUY").any(), \
        "押し目リバウンド局面で逆張りBUYが発火するべき"

    long_lev = run_signals(df.copy(), default_params(ticker_class=TICKER_CLASS_LONG_LEV),
                           "逆張り")
    assert not (long_lev["composite_signal"].iloc[-9:] == "BUY").any()


def test_dip_rebound_buy_requires_deviation_above_zero():
    """同じ押し目でも長期MA割れ (deviation<0) では発火しない (崩壊銘柄の除外)"""
    rng = np.random.default_rng(22)
    # 横ばい→急落: dd20 は -8% を超えるが deviation は負になる
    flat = 100.0 + rng.normal(0, 0.3, 150)
    crash = flat[-1] * np.cumprod(np.full(8, 0.982))
    rebound = crash[-1] * np.cumprod(np.full(4, 1.01))
    df = make_ohlcv(np.concatenate([flat, crash, rebound]))

    data = run_signals(df, default_params(ticker_class=TICKER_CLASS_PLAIN), "逆張り")
    assert not (data["composite_signal"].iloc[-13:] == "BUY").any()


def test_downtrend_long_lev_never_sells():
    """下落データでも long_lev クラスではトレンドSELLが出ない (トレイリングストップが担当)"""
    rng = np.random.default_rng(11)
    up = 100.0 * np.cumprod(1 + 0.004 + rng.normal(0, 0.01, 150))
    down = up[-1] * np.cumprod(1 - 0.008 + rng.normal(0, 0.02, 150))
    df = make_ohlcv(np.concatenate([up, down]))

    plain = run_signals(df.copy(), default_params(ticker_class=TICKER_CLASS_PLAIN),
                        "トレンドフォロー")
    long_lev = run_signals(df.copy(), default_params(ticker_class=TICKER_CLASS_LONG_LEV),
                           "トレンドフォロー")

    assert (plain["composite_signal"] == "SELL").any(), \
        "前提条件エラー: plain で SELL が出ない合成データではテストにならない"
    assert not (long_lev["composite_signal"] == "SELL").any()
