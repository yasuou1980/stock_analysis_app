"""signal_tracker の履歴スキーマ互換・版別集計のテスト"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import signal_tracker  # noqa: E402


def _batch_row(**overrides):
    row = {
        "run_date": "2026-07-06",
        "ticker": "NVDA",
        "strategy": "トレンドフォロー",
        "signal_date": "2026-07-06",
        "close": 100.0,
        "composite_signal": "BUY",
        "rsi": 60.0,
        "deviation": 5.0,
        "score": 6.2,
        "adx": 25.0,
        "ret_5d": 0.03,
        "ticker_class": "plain",
    }
    row.update(overrides)
    return row


def test_ingest_writes_extended_columns(tmp_path):
    signal_tracker.ingest([_batch_row()], tmp_path)
    hist = signal_tracker.load_history(tmp_path)
    assert list(hist.columns) == signal_tracker.HISTORY_COLUMNS
    assert hist.iloc[0]["score"] == 6.2
    assert hist.iloc[0]["ticker_class"] == "plain"


def test_ingest_backward_compatible_with_old_rows(tmp_path):
    """旧フォーマット (新列なし) の履歴 CSV に追記しても壊れない"""
    old = pd.DataFrame([{
        "signal_date": "2026-07-03", "ticker": "NVDA", "strategy": "トレンドフォロー",
        "signal": "HOLD", "close": 95.0, "rsi": 50.0, "deviation": 1.0,
    }])
    (tmp_path / signal_tracker.HISTORY_NAME).write_text(old.to_csv(index=False))

    signal_tracker.ingest([_batch_row()], tmp_path)
    hist = signal_tracker.load_history(tmp_path)
    assert len(hist) == 2
    old_row = hist[hist["signal_date"] == "2026-07-03"].iloc[0]
    assert np.isnan(old_row["score"])  # 旧行は NaN 埋め


def test_rows_without_new_fields_are_accepted(tmp_path):
    """新フィールドを持たない行 (旧 batch_runner 形式) も ingest できる"""
    row = _batch_row()
    for key in ("score", "adx", "ret_5d", "ticker_class"):
        row.pop(key)
    signal_tracker.ingest([row], tmp_path)
    hist = signal_tracker.load_history(tmp_path)
    assert len(hist) == 1
    assert np.isnan(hist.iloc[0]["score"])


def test_summarize_by_version_splits_on_boundaries():
    onsets = pd.DataFrame({
        "signal_date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-06-15", "2026-07-06"]),
        "strategy": ["トレンドフォロー"] * 4,
        "signal": ["BUY"] * 4,
        "fwd_5d": [0.05, -0.02, 0.03, 0.01],
        "fwd_10d": [0.08, -0.04, 0.06, 0.02],
        "fwd_20d": [np.nan] * 4,
    })
    out = signal_tracker.summarize_by_version(onsets)
    versions = set(out["version"])
    assert any(v.startswith("v1") for v in versions)
    assert any(v.startswith("v2") for v in versions)
    assert any(v.startswith("v3") for v in versions)
    v1_5d = out[(out["version"].str.startswith("v1")) & (out["horizon"] == 5)].iloc[0]
    assert v1_5d["n"] == 2
    assert v1_5d["win_rate"] == 50.0


def test_write_report_contains_version_section(tmp_path):
    # 版別集計が出るだけの履歴を用意 (v1/v2 期間に BUY onset)
    rows = []
    for d, sig in [("2026-05-01", "BUY"), ("2026-06-15", "BUY")]:
        rows.append(_batch_row(signal_date=d, composite_signal=sig))
        # フォワードリターン計算用に後続日の HOLD 行 (終値) を足す
    for i, d in enumerate(pd.bdate_range("2026-05-04", periods=40)):
        rows.append(_batch_row(signal_date=str(d.date()), composite_signal="HOLD",
                               close=100.0 + i))
    signal_tracker.ingest(rows, tmp_path)
    path = signal_tracker.write_report(tmp_path)
    text = path.read_text()
    assert "シグナル実績レポート" in text
    assert "ロジック版別実績" in text
    assert "ゲートで抑制されたシグナルの成績" in text


# ---------------------------------------------------------------------------
# 株式分割/併合の補正 (2026-09 の計測バグ修正)
# ---------------------------------------------------------------------------
def test_adjust_for_splits_makes_series_continuous():
    """未調整の株式併合 (1:10) を検出し、遡って価格を揃える"""
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    s = pd.Series([10.0, 10.5, 10.2, 102.0, 101.0, 110.0], index=idx)

    adj = signal_tracker.adjust_for_splits(s)

    # 併合日をまたぐリターンが桁違いにならない
    assert abs(adj.pct_change().iloc[3]) < 0.10
    # 併合後の価格はそのまま、併合前は約10倍にスケールされる
    assert adj.iloc[-1] == 110.0
    assert 99.0 < adj.iloc[0] < 106.0   # 10.0 が併合比率 10 倍でスケールされる


def test_adjust_for_splits_leaves_normal_series_untouched():
    """通常の値動き (3倍レバETFの急変を含む) は補正しない"""
    idx = pd.date_range("2026-01-01", periods=5, freq="D")
    s = pd.Series([100.0, 130.0, 95.0, 140.0, 120.0], index=idx)
    pd.testing.assert_series_equal(signal_tracker.adjust_for_splits(s), s)


def test_split_artifact_does_not_distort_forward_returns():
    """併合をまたぐ SELL の fwd リターンが -1000% 級にならない"""
    dates = pd.bdate_range("2026-01-01", periods=8)
    rows = []
    for i, d in enumerate(dates):
        # 5日目に 1:10 の併合 (10 → 100)
        close = 10.0 if i < 5 else 100.0
        rows.append({"signal_date": d.strftime("%Y-%m-%d"), "ticker": "SOXS",
                     "strategy": "トレンドフォロー",
                     "signal": "SELL" if i == 0 else "HOLD",
                     "close": close, "rsi": 50.0, "deviation": 0.0,
                     "score": np.nan, "adx": np.nan, "ret_5d": np.nan,
                     "ticker_class": "inverse_lev"})
    onsets = signal_tracker.compute_onset_performance(pd.DataFrame(rows), horizons=(5,))
    assert len(onsets) == 1
    assert abs(onsets["fwd_5d"].iloc[0]) < 0.10


# ---------------------------------------------------------------------------
# シャドー計測 (ゲートが止めたシグナルのフォワード検証, 2026-09)
# ---------------------------------------------------------------------------
def test_ingest_records_raw_signal(tmp_path):
    rows = [{"signal_date": "2026-09-02", "ticker": "AAPL", "strategy": "トレンドフォロー",
             "close": 100.0, "composite_signal": "HOLD", "raw_signal": "SELL",
             "rsi": 40.0, "deviation": -2.0}]
    signal_tracker.ingest(rows, tmp_path)
    hist = signal_tracker.load_history(tmp_path)
    assert hist.loc[0, "signal"] == "HOLD"
    assert hist.loc[0, "raw_signal"] == "SELL"


def test_suppressed_performance_tracks_gated_signals():
    """ゲートが止めた SELL のその後 (価格上昇 = 止めて正解) を直接計測できる"""
    dates = pd.bdate_range("2026-09-01", periods=8)
    rows = [{
        "signal_date": d.strftime("%Y-%m-%d"), "ticker": "AAPL", "strategy": "トレンドフォロー",
        "signal": "HOLD",                              # ゲート適用後
        "raw_signal": "SELL" if i == 0 else "HOLD",    # ゲート適用前
        "close": 100.0 + 2.0 * i,                      # その後上昇 → SELL は不正解 = 止めて正解
        "rsi": 40.0, "deviation": -2.0,
    } for i, d in enumerate(dates)]
    sup = signal_tracker.compute_suppressed_performance(pd.DataFrame(rows), horizons=(5,))
    assert len(sup) == 1
    assert sup["signal"].iloc[0] == "SELL" and sup["actual_signal"].iloc[0] == "HOLD"
    assert sup["fwd_5d"].iloc[0] < 0


def test_suppressed_performance_ignores_rows_without_raw_signal():
    """raw_signal の無い旧形式の行 (2026-09 以前) は対象外"""
    hist = pd.DataFrame([{"signal_date": "2026-06-01", "ticker": "AAPL", "strategy": "トレンドフォロー",
                          "signal": "SELL", "close": 100.0, "rsi": 40.0, "deviation": 0.0,
                          "raw_signal": np.nan}])
    assert signal_tracker.compute_suppressed_performance(hist).empty


# ---------------------------------------------------------------------------
# ベースレート超過 (エッジ) — 勝率だけでは「市場が上がっただけ」と区別できない
# ---------------------------------------------------------------------------
def test_compute_base_rates_splits_evenly():
    """A は毎日上昇・B は毎日下落 → 全観測を混ぜると上昇率はちょうど50%"""
    dates = pd.bdate_range("2026-01-01", periods=40)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"signal_date": d.strftime("%Y-%m-%d"), "ticker": "A", "close": 100.0 + i})
        rows.append({"signal_date": d.strftime("%Y-%m-%d"), "ticker": "B", "close": 200.0 - i})
    hist = pd.DataFrame(rows)

    rates = signal_tracker.compute_base_rates(hist, (5,))

    assert rates[5]["up_rate"] == 50.0
    assert rates[5]["n"] > 0


def test_summarize_edge_buy():
    """BUY のエッジ = シグナルの勝率からベース上昇率を引いたもの"""
    onsets = pd.DataFrame({
        "strategy": ["トレンドフォロー"] * 3,
        "signal": ["BUY"] * 3,
        "fwd_5d": [0.05, -0.02, 0.03],
    })
    base_rates = {5: {"n": 50, "up_rate": 52.0, "avg_pct": 1.5}}

    out = signal_tracker.summarize(onsets, horizons=(5,), base_rates=base_rates)
    r = out.iloc[0]

    assert r["base_win_rate"] == 52.0
    assert r["edge_win_rate"] == r["win_rate"] - r["base_win_rate"]
    assert r["edge_avg_pct"] == r["avg_pct"] - r["base_avg_pct"]


def test_summarize_edge_sell_inverts_base():
    """SELL は fwd が符号反転済みのため、ベース側も反転して比較する"""
    onsets = pd.DataFrame({
        "strategy": ["トレンドフォロー"] * 3,
        "signal": ["SELL"] * 3,
        "fwd_5d": [0.05, -0.02, 0.03],
    })
    base_rates = {5: {"n": 50, "up_rate": 52.0, "avg_pct": 1.5}}

    out = signal_tracker.summarize(onsets, horizons=(5,), base_rates=base_rates)
    r = out.iloc[0]

    assert r["base_win_rate"] == 48.0     # 100 - up_rate
    assert r["base_avg_pct"] == -1.5      # -avg_pct


def test_summarize_without_base_rates_keeps_nan():
    """base_rates 省略時 (None) は4列とも NaN で、既存列は従来通り"""
    onsets = pd.DataFrame({
        "strategy": ["トレンドフォロー"] * 3,
        "signal": ["BUY"] * 3,
        "fwd_5d": [0.05, -0.02, 0.03],
    })

    out = signal_tracker.summarize(onsets, horizons=(5,))
    r = out.iloc[0]

    assert np.isnan(r["base_win_rate"])
    assert np.isnan(r["base_avg_pct"])
    assert np.isnan(r["edge_win_rate"])
    assert np.isnan(r["edge_avg_pct"])
    assert r["win_rate"] == 100.0 * (2 / 3)
    assert r["n"] == 3


def test_write_report_contains_edge_columns(tmp_path):
    rows = []
    for i, d in enumerate(pd.bdate_range("2026-05-01", periods=45)):
        sig = "BUY" if i == 0 else "HOLD"
        rows.append(_batch_row(signal_date=str(d.date()), composite_signal=sig, close=100.0 + i))
    signal_tracker.ingest(rows, tmp_path)

    path = signal_tracker.write_report(tmp_path)
    text = path.read_text()

    assert "ベース" in text
    assert "エッジ" in text
