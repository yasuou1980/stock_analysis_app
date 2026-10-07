"""batch_runner が「記録済みより新しいバーをすべて」履歴に追加することのテスト

Yahoo Finance の更新は銘柄ごとにタイミングが違い、ある日の実行で
銘柄 A は最新バーが D、銘柄 B は D-1 のままということが起きる。
最新バー 1 本だけを追記していると、B は D-1 が履歴から欠落する。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batch_runner  # noqa: E402
import signal_tracker  # noqa: E402

N_BARS = 120
STRATEGIES = ["トレンドフォロー", "逆張り"]


def _ohlcv(seed, n=N_BARS):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n))
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": np.full(n, 1_000_000),
    }, index=idx)


def _run(monkeypatch, tmp_path, frames):
    """frames: {ticker: DataFrame} を Yahoo の返却として batch_runner.run() を実行"""
    config = {
        "batch": {"tickers": list(frames), "results_dir": str(tmp_path)},
        "ticker_classes": {},
    }
    monkeypatch.setattr(batch_runner.toml, "load", lambda _path: config)
    monkeypatch.setattr(batch_runner, "load_data",
                        lambda ticker, start, end: frames[ticker].copy())
    batch_runner.run(no_db=True)
    return signal_tracker.load_history(tmp_path)


def _dates(hist, ticker, strategy):
    sub = hist[(hist["ticker"] == ticker) & (hist["strategy"] == strategy)]
    return sorted(sub["signal_date"])


def _d(frame, i):
    return str(frame.index[i].date())


def test_gap_day_is_filled_on_next_run(monkeypatch, tmp_path):
    a, b = _ohlcv(1), _ohlcv(2)

    # 1 日目: 両銘柄とも最新バーは D-2
    day1 = _run(monkeypatch, tmp_path, {"AAA": a.iloc[:-2], "BBB": b.iloc[:-2]})
    before = day1.copy()

    # 2 日目: A は最新バー D (D-1 を飛ばして更新)、B は D-1 止まり
    day2 = _run(monkeypatch, tmp_path, {"AAA": a, "BBB": b.iloc[:-1]})
    for strategy in STRATEGIES:
        assert _dates(day2, "AAA", strategy) == [_d(a, -3), _d(a, -2), _d(a, -1)]  # D-1 が埋まる
        assert _dates(day2, "BBB", strategy) == [_d(b, -3), _d(b, -2)]

    # 3 日目: B にも D が反映。両銘柄とも欠落なし
    day3 = _run(monkeypatch, tmp_path, {"AAA": a, "BBB": b})
    for strategy in STRATEGIES:
        assert _dates(day3, "AAA", strategy) == [_d(a, -3), _d(a, -2), _d(a, -1)]
        assert _dates(day3, "BBB", strategy) == [_d(b, -3), _d(b, -2), _d(b, -1)]

    # 既存行は書き換わらない
    key = ["signal_date", "ticker", "strategy"]
    merged = before.merge(day3, on=key, suffixes=("_old", "_new"))
    assert len(merged) == len(before)
    for col in signal_tracker.HISTORY_COLUMNS:
        if col in key:
            continue
        old, new = merged[f"{col}_old"], merged[f"{col}_new"]
        assert ((old == new) | (old.isna() & new.isna())).all(), col


def test_skipped_day_is_backfilled_in_one_run(monkeypatch, tmp_path):
    """実行が 1 日飛んだ (D-1 を記録しないまま D に進む) 場合も D-1 が埋まり、特徴量も入る"""
    a = _ohlcv(3)
    _run(monkeypatch, tmp_path, {"AAA": a.iloc[:-2]})
    hist = _run(monkeypatch, tmp_path, {"AAA": a})

    assert _dates(hist, "AAA", STRATEGIES[0]) == [_d(a, -3), _d(a, -2), _d(a, -1)]
    filled = hist[hist["signal_date"] == _d(a, -2)]
    assert len(filled) == 2
    assert filled["raw_signal"].notna().all()
    assert filled["ticker_class"].notna().all()
    assert filled["score"].notna().all()
    expected_ret = round(float(a["close"].pct_change(5).iloc[-2]), 4)
    assert (filled["ret_5d"] == expected_ret).all()


def test_new_ticker_gets_only_latest_bar(monkeypatch, tmp_path):
    """履歴に行が無い銘柄は従来どおり最新バー 1 本のみ"""
    a, b = _ohlcv(4), _ohlcv(5)
    _run(monkeypatch, tmp_path, {"AAA": a.iloc[:-1]})
    hist = _run(monkeypatch, tmp_path, {"AAA": a, "BBB": b})
    assert _dates(hist, "BBB", STRATEGIES[0]) == [_d(b, -1)]


def test_rerun_same_day_adds_nothing(monkeypatch, tmp_path):
    a = _ohlcv(6)
    first = _run(monkeypatch, tmp_path, {"AAA": a})
    second = _run(monkeypatch, tmp_path, {"AAA": a})
    assert len(first) == len(second) == 2


def test_build_signal_rows_after_filters_strictly_newer():
    data = _ohlcv(7)
    data["composite_signal"] = "HOLD"
    data["raw_signal"] = "HOLD"
    rows = batch_runner.build_signal_rows(
        data, "X", "逆張り", "plain", "2026-10-06", after=_d(data, -3))
    assert [r["signal_date"] for r in rows] == [_d(data, -2), _d(data, -1)]
    assert batch_runner.build_signal_rows(
        data, "X", "逆張り", "plain", "2026-10-06", after=_d(data, -1)) == []
    assert len(batch_runner.build_signal_rows(data, "X", "逆張り", "plain", "2026-10-06")) == 1
