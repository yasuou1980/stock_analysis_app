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
