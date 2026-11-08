"""押し目候補戦略 (上昇トレンド中の RSI 売られ過ぎ) の回帰テスト"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtester import (  # noqa: E402
    PULLBACK_RSI_MAX,
    TICKER_CLASS_INVERSE_LEV,
    TICKER_CLASS_PLAIN,
)
from test_signal_gates import default_params, make_ohlcv, run_signals  # noqa: E402

STRATEGY = "押し目候補"


def _uptrend_then_dip(n_up=320, n_dip=8, dip=-0.035):
    rng = np.random.default_rng(3)
    up = 100.0 * np.cumprod(1 + 0.003 + rng.normal(0, 0.006, n_up))
    down = up[-1] * np.cumprod(np.full(n_dip, 1 + dip))
    return np.concatenate([up, down])


def _downtrend(n=320):
    rng = np.random.default_rng(5)
    return 300.0 * np.cumprod(1 - 0.003 + rng.normal(0, 0.006, n))


def test_buy_fires_on_dip_inside_uptrend():
    data = run_signals(make_ohlcv(_uptrend_then_dip()), default_params(ticker_class=TICKER_CLASS_PLAIN), STRATEGY)
    buys = data[data["composite_signal"] == "BUY"]
    assert not buys.empty, "上昇トレンド中の急な押しで BUY 候補が出ない"
    assert (buys["rsi"] < PULLBACK_RSI_MAX).all()
    assert (buys["close"] > buys["sma200"]).all()
    # 押しの深さスコアは RSI が低いほど大きい
    assert (buys["pullback_score"] > 0).all()


def test_never_emits_sell():
    for closes in (_uptrend_then_dip(), _downtrend()):
        data = run_signals(make_ohlcv(closes), default_params(ticker_class=TICKER_CLASS_PLAIN), STRATEGY)
        assert set(data["composite_signal"]) <= {"BUY", "HOLD"}
        assert set(data["raw_signal"]) <= {"BUY", "HOLD"}


def test_no_buy_below_200d_average():
    """下降トレンド (終値が200日線の下) の売られ過ぎは候補にしない (落ちるナイフ回避)"""
    data = run_signals(make_ohlcv(_downtrend()), default_params(ticker_class=TICKER_CLASS_PLAIN), STRATEGY)
    assert (data["rsi"] < PULLBACK_RSI_MAX).any(), "前提条件エラー: RSI が下がらない合成データ"
    assert not (data["composite_signal"] == "BUY").any()


def test_inverse_etf_gate_blocks_buy_but_keeps_raw():
    closes = _uptrend_then_dip()
    plain = run_signals(make_ohlcv(closes), default_params(ticker_class=TICKER_CLASS_PLAIN), STRATEGY)
    inv = run_signals(make_ohlcv(closes), default_params(ticker_class=TICKER_CLASS_INVERSE_LEV), STRATEGY)
    assert (plain["composite_signal"] == "BUY").any()
    assert not (inv["composite_signal"] == "BUY").any()
    assert (inv["raw_signal"] == "BUY").sum() == (plain["composite_signal"] == "BUY").sum()


def test_causal_truncation_gives_same_signal():
    """未来を見ていない: 途中で切って計算しても、その日のシグナルは全期間計算と一致"""
    closes = _uptrend_then_dip()
    full = run_signals(make_ohlcv(closes), default_params(), STRATEGY)
    cut = run_signals(make_ohlcv(closes[:-3]), default_params(), STRATEGY)
    common = cut.index.intersection(full.index)
    assert len(common) > 10
    assert (cut.loc[common, "composite_signal"] == full.loc[common, "composite_signal"]).all()
