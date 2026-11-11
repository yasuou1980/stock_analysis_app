"""スイング用の売買プラン・保有ポジション判定・プラン検証 (swing.py) の回帰テスト"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import swing  # noqa: E402
import signal_tracker  # noqa: E402
from backtester import _atr  # noqa: E402

SETTINGS = dict(swing.DEFAULT_SETTINGS)  # 資金 $10,000 / 許容損失 1% / ATR×2 / 10 日 / 上限 25%


def ohlc(closes, lows=None, opens=None, start="2026-01-05"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "open": closes if opens is None else np.asarray(opens, dtype=float),
        "high": closes + 1.0,
        "low": closes - 1.0 if lows is None else np.asarray(lows, dtype=float),
        "close": closes,
    }, index=idx)


def signal_frame(signals, close=100.0, atr=2.0):
    idx = pd.bdate_range("2026-01-05", periods=len(signals))
    return pd.DataFrame({"close": close, "atr": atr, "rsi": 30.0, "composite_signal": signals}, index=idx)


# ---------------------------------------------------------------------------
# 売買プラン
# ---------------------------------------------------------------------------
def test_trade_plan_sizes_position_from_allowed_loss():
    plan = swing.trade_plan(100.0, 2.0, SETTINGS)
    # 損切り幅 = ATR 2 × 2 = $4。許容損失 $100 ÷ $4 = 25 株 (建玉 $2,500 = 上限ちょうど)
    assert plan["stop"] == pytest.approx(96.0)
    assert plan["stop_pct"] == pytest.approx(-4.0)
    assert plan["shares"] == 25
    assert plan["max_loss"] == pytest.approx(100.0)
    assert plan["position_pct"] == pytest.approx(25.0)


def test_trade_plan_caps_position_for_low_volatility():
    """値動きが小さい銘柄は許容損失から逆算すると建玉が膨らむので、上限で止める"""
    plan = swing.trade_plan(100.0, 0.5, SETTINGS)  # 損切り幅 $1 → 逆算 100 株 = 資金の 100%
    assert plan["capped"]
    assert plan["shares"] == 25
    assert plan["max_loss"] == pytest.approx(25.0)  # 許容損失 $100 より小さくなる
    assert plan["position_pct"] == pytest.approx(25.0)


def test_trade_plan_rejects_unusable_inputs():
    assert swing.trade_plan(100.0, float("nan"), SETTINGS) is None
    assert swing.trade_plan(0.0, 2.0, SETTINGS) is None
    assert swing.trade_plan(3.0, 2.0, SETTINGS) is None  # 損切り価格がマイナスになる


def test_swing_settings_overrides_defaults_and_ignores_unknown_keys():
    s = swing.swing_settings({"swing": {"risk_pct": 2, "max_hold_days": 20.0, "unknown": 1}})
    assert s["risk_pct"] == 2.0 and s["max_hold_days"] == 20 and isinstance(s["max_hold_days"], int)
    assert "unknown" not in s
    assert swing.swing_settings(None) == swing.DEFAULT_SETTINGS


# ---------------------------------------------------------------------------
# 候補スキャン
# ---------------------------------------------------------------------------
def test_buy_streak_counts_trailing_buys_only():
    assert swing.buy_streak(pd.Series(["BUY", "HOLD", "BUY", "BUY"])) == 2
    assert swing.buy_streak(pd.Series(["BUY", "BUY", "HOLD"])) == 0
    assert swing.buy_streak(pd.Series([], dtype=object)) == 0


def test_scan_lists_only_latest_buys_with_new_onsets_first():
    frames = {
        ("AAA", "トレンドフォロー"): signal_frame(["HOLD", "HOLD", "BUY"]),
        ("BBB", "押し目候補"): signal_frame(["HOLD", "BUY", "BUY"]),
        ("CCC", "押し目候補"): signal_frame(["HOLD", "HOLD", "BUY"]),
        ("DDD", "押し目候補"): signal_frame(["BUY", "BUY", "HOLD"]),   # 最新が HOLD → 対象外
        ("EEE", "逆張り"): signal_frame(["HOLD", "HOLD", "SELL"]),
    }
    got = swing.scan_candidates(frames, {"AAA": "long_lev"}, SETTINGS)
    assert [(c["strategy"], c["ticker"], c["streak"]) for c in got] == [
        ("押し目候補", "CCC", 1), ("押し目候補", "BBB", 2), ("トレンドフォロー", "AAA", 1)]
    first = got[0]
    assert first["stop"] == pytest.approx(96.0) and first["shares"] == 25
    assert got[2]["ticker_class"] == "long_lev" and first["ticker_class"] == "plain"
    # シグナル日 2026-01-07 (水) の 10 営業日後 = 01-21
    assert str(first["signal_date"]) == "2026-01-07" and str(first["exit_date"]) == "2026-01-21"


def test_report_lines_do_not_collide_with_history_parser():
    """日次レポートに追記する行が、履歴再構築 (backfill_from_text) にシグナル行として拾われないこと"""
    frames = {("AAA", "押し目候補"): signal_frame(["HOLD", "BUY"])}
    lines = swing.format_candidates_text(swing.scan_candidates(frames, {}, SETTINGS), SETTINGS)
    assert any("AAA" in line for line in lines)
    for line in lines:
        assert not signal_tracker._TXT_LINE_RE.match(line)
        assert not line.strip().startswith("【")
    # 資金額・株数は公開レポートに出さない
    assert not any("10000" in line or "10,000" in line for line in lines)
    assert "該当なし" in "\n".join(swing.format_candidates_text([], SETTINGS))


def test_market_context_reports_gap_to_200d_average():
    prices = {"SPY": ohlc(np.linspace(100, 200, 250)), "QQQ": ohlc([100.0] * 50)}
    got = swing.market_context(prices)
    assert [m["ticker"] for m in got] == ["SPY"]  # 200 本に満たない QQQ は出さない
    assert got[0]["above"] and got[0]["gap_pct"] > 0


# ---------------------------------------------------------------------------
# 保有ポジションの判定
# ---------------------------------------------------------------------------
def position(**kw):
    base = {"ticker": "AAA", "entry_date": "2026-02-02", "entry_price": 100.0, "shares": 10,
            "stop_price": 95.0, "note": None}
    base.update(kw)
    return base


def test_position_hold_reports_pnl_and_remaining_days():
    prices = ohlc([100.0] * 20 + [100, 102, 104], start="2026-01-05")  # 02-02 が買付日 (21 本目)
    r = swing.evaluate_position(prices, position(), SETTINGS)
    assert r["status"] == swing.STATUS_HOLD
    assert r["days_in_trade"] == 3
    assert r["pnl_pct"] == pytest.approx(4.0) and r["pnl"] == pytest.approx(40.0)
    assert r["r_multiple"] == pytest.approx(0.8)  # +$4 ÷ リスク $5
    assert str(r["exit_date"]) == "2026-02-13"    # 買付日を 1 日目として 10 営業日目


def test_position_stop_hit_wins_over_time_exit():
    closes = [100.0] * 20 + [100.0] * 12
    lows = [99.0] * 32
    lows[25] = 94.0  # 買付後に安値が損切り価格を割る
    r = swing.evaluate_position(ohlc(closes, lows=lows), position(), SETTINGS)
    assert r["status"] == swing.STATUS_STOP
    assert str(r["stop_hit_date"]) == str(pd.bdate_range("2026-01-05", periods=32)[25].date())


def test_position_entry_day_low_does_not_trigger_stop():
    """買付日当日の安値は、買う前に付けた値かもしれないので損切り判定に使わない (終値で判定)"""
    closes = [100.0] * 23
    lows = [99.0] * 23
    lows[20] = 90.0
    r = swing.evaluate_position(ohlc(closes, lows=lows), position(), SETTINGS)
    assert r["status"] == swing.STATUS_HOLD


def test_position_time_exit_on_max_hold_day():
    r9 = swing.evaluate_position(ohlc([100.0] * 29), position(), SETTINGS)
    r10 = swing.evaluate_position(ohlc([100.0] * 30), position(), SETTINGS)
    assert (r9["status"], r9["days_in_trade"]) == (swing.STATUS_HOLD, 9)
    assert (r10["status"], r10["days_in_trade"]) == (swing.STATUS_TIME, 10)


def test_position_auto_stop_uses_atr_before_entry():
    prices = ohlc([100.0] * 23)  # high-low = 2 で一定 → ATR = 2
    r = swing.evaluate_position(prices, position(stop_price=None), SETTINGS)
    assert r["stop_is_auto"] and r["stop"] == pytest.approx(96.0)


def test_position_without_data_is_reported_not_raised():
    assert swing.evaluate_position(None, position(), SETTINGS)["status"] == swing.STATUS_NO_DATA
    future = swing.evaluate_position(ohlc([100.0] * 5), position(), SETTINGS)
    assert future["status"] == swing.STATUS_NO_DATA


def test_positions_roundtrip_drops_incomplete_rows(tmp_path):
    path = tmp_path / "positions.csv"
    assert swing.load_positions(path).empty
    edited = pd.DataFrame([
        {"ticker": " nvda ", "entry_date": "2026-02-02", "entry_price": 100.0, "shares": 5, "stop_price": None, "note": "x"},
        {"ticker": "AMD", "entry_date": None, "entry_price": 50.0, "shares": 1, "stop_price": None, "note": None},
        {"ticker": None, "entry_date": "2026-02-02", "entry_price": 50.0, "shares": 1, "stop_price": None, "note": None},
    ])
    saved = swing.save_positions(edited, path)
    assert list(saved["ticker"]) == ["NVDA"]
    loaded = swing.load_positions(path)
    assert str(loaded.loc[0, "entry_date"]) == "2026-02-02" and pd.isna(loaded.loc[0, "stop_price"])


# ---------------------------------------------------------------------------
# プランの過去シミュレーション
# ---------------------------------------------------------------------------
def test_atr_matches_backtester():
    rng = np.random.default_rng(1)
    prices = ohlc(100 + np.cumsum(rng.normal(0, 1, 80)))
    expected = prices.copy()
    _atr(expected)
    pd.testing.assert_series_equal(swing.atr_series(prices), expected["atr"], check_names=False)


def test_simulate_plan_time_exit_and_stop_exit():
    closes = [100.0] * 30 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] + [110.0] * 5
    prices = ohlc(closes)
    # シグナル = 30 本目の前日 (29)。翌バー始値 101 で買い、10 日目 (39) の終値 110 で売る
    t = swing.simulate_plan(prices, [29], stop_atr_mult=2.0, max_hold_days=10).iloc[0]
    assert (t["entry"], t["exit"], t["reason"], t["days"]) == (101.0, 110.0, "time", 10)
    assert t["stop"] == pytest.approx(101.0 - 2 * 2.0)
    assert t["r_multiple"] == pytest.approx(9.0 / 4.0)

    lows = [c - 1.0 for c in closes]
    lows[33] = 90.0  # 4 日目に損切り価格 97 を割る
    s = swing.simulate_plan(ohlc(closes, lows=lows), [29], 2.0, 10).iloc[0]
    assert (s["exit"], s["reason"], s["days"]) == (97.0, "stop", 4)
    assert s["r_multiple"] == pytest.approx(-1.0)


def test_simulate_plan_target_and_trailing_variants():
    """比較検証用の利確指値・トレイリング (画面のプランでは使わない)"""
    closes = [100.0] * 30 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] + [110.0] * 5
    prices = ohlc(closes)
    # リスク $4 → 2R の指値は 109。高値 (終値+1) が 109 に届く 108 の日に約定
    t = swing.simulate_plan(prices, [29], 2.0, 10, target_r=2.0).iloc[0]
    assert (t["exit"], t["reason"]) == (109.0, "target") and t["r_multiple"] == pytest.approx(2.0)

    lows = [c - 1.0 for c in closes]
    lows[37] = 100.0  # 買値 101 を上回ったまま、引き上げ後の損切り (終値最高値 107 − 4 = 103) に掛かる
    tr = swing.simulate_plan(ohlc(closes, lows=lows), [29], 2.0, 10, trail=True).iloc[0]
    assert tr["reason"] == "stop" and tr["exit"] == pytest.approx(103.0)
    assert tr["stop"] == pytest.approx(97.0)  # 記録するのは最初の損切り価格
    # 引き上げなしなら同じ値動きで最後まで保有する
    assert swing.simulate_plan(ohlc(closes, lows=lows), [29], 2.0, 10).iloc[0]["reason"] == "time"


def test_simulate_plan_gap_down_exits_at_open_and_skips_unfinished():
    closes = [100.0] * 45
    opens = list(closes)
    lows = [99.0] * 45
    opens[32], lows[32] = 90.0, 89.0  # 窓を開けて損切り価格 96 の下で寄る → 始値で約定
    prices = ohlc(closes, lows=lows, opens=opens)
    trades = swing.simulate_plan(prices, [29, 40], 2.0, 10)  # 40 は手仕舞い日までデータが無い
    assert len(trades) == 1
    assert trades.iloc[0]["exit"] == 90.0 and trades.iloc[0]["r_multiple"] < -1.0


def test_onset_bars_and_summary():
    prices = ohlc([100.0] * 8)
    signals = pd.Series(["BUY", "BUY", "HOLD", "BUY", "HOLD"], index=prices.index[3:])
    assert swing.onset_bars(signals, prices) == [3, 6]
    summary = swing.summarize_trades(pd.DataFrame({
        "ret_pct": [4.0, -2.0], "r_multiple": [1.0, -1.0], "reason": ["time", "stop"]}))
    assert summary["n"] == 2 and summary["win_rate"] == 50.0 and summary["stop_rate"] == 50.0
    assert swing.summarize_trades(pd.DataFrame(columns=swing.TRADE_COLUMNS)) == {"n": 0}
