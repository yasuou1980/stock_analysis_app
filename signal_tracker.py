#!/usr/bin/env python3
"""
シグナル実績トラッキング
日次バッチが出したシグナルを CSV に蓄積し、N営業日後のフォワードリターンで
「新規シグナル発生ベースの勝率」を自動計測してレポートを生成する。

設計:
- 蓄積先は results/signals_history.csv (git にコミットできるテキスト形式)
- 重複排除キーは (signal_date, ticker, strategy)。土日や休場日の再実行は
  signal_date が変わらないため自然に重複排除される
- 勝率は「新規シグナル発生」(前営業日と異なるシグナルに変わった日) のみで計測。
  連日同じシグナルが続く分を数えると重複カウントで精度が歪むため
- SELL シグナルの勝ちは「その後に価格が下がったこと」と定義する

使い方:
    python signal_tracker.py backfill   # 既存の signals_*.txt から履歴を再構築
    python signal_tracker.py report     # レポートのみ再生成
    (通常は batch_runner.py から自動で ingest + report が呼ばれる)
"""
import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR_DEFAULT = SCRIPT_DIR / "results"
HISTORY_NAME = "signals_history.csv"
REPORT_NAME = "performance_report.txt"

# フォワードリターンの計測ホライズン (営業日)
HORIZONS = (5, 10, 20)

HISTORY_COLUMNS = ["signal_date", "ticker", "strategy", "signal", "close", "rsi", "deviation"]

_TXT_LINE_RE = re.compile(
    r"^\s+([A-Z]+)\s+(🟢 BUY|🔴 SELL|⚪ HOLD)\s+([\d.]+)\s+([\d.]+)\s+([+\-][\d.]+)%"
)
_SIGNAL_MAP = {"🟢 BUY": "BUY", "🔴 SELL": "SELL", "⚪ HOLD": "HOLD"}


# ---------------------------------------------------------------------------
# 履歴 CSV の読み書き
# ---------------------------------------------------------------------------
def load_history(results_dir: Path = RESULTS_DIR_DEFAULT) -> pd.DataFrame:
    path = results_dir / HISTORY_NAME
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path, dtype={"signal_date": str, "ticker": str, "strategy": str, "signal": str})
    return df


def save_history(df: pd.DataFrame, results_dir: Path = RESULTS_DIR_DEFAULT) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["signal_date", "strategy", "ticker"]).reset_index(drop=True)
    df.to_csv(results_dir / HISTORY_NAME, index=False)


def ingest(rows: list[dict], results_dir: Path = RESULTS_DIR_DEFAULT) -> int:
    """batch_runner が計算した当日分のシグナルを履歴に追記する。

    rows は batch_runner.run() が組み立てる dict のリスト
    (signal_date / ticker / strategy / close / composite_signal / rsi / deviation を含む)。
    戻り値は追加された行数。
    """
    if not rows:
        return 0
    new = pd.DataFrame([
        {
            "signal_date": r["signal_date"],
            "ticker": r["ticker"],
            "strategy": r["strategy"],
            "signal": r["composite_signal"],
            "close": r["close"],
            "rsi": r.get("rsi", np.nan),
            "deviation": r.get("deviation", np.nan),
        }
        for r in rows
    ])
    hist = load_history(results_dir)
    before = len(hist)
    merged = pd.concat([hist, new], ignore_index=True)
    # 同一キーは後勝ち (再実行時に最新の計算結果を採用)
    merged = merged.drop_duplicates(subset=["signal_date", "ticker", "strategy"], keep="last")
    save_history(merged, results_dir)
    added = len(merged) - before
    logger.info(f"シグナル履歴に {added} 行追加 (合計 {len(merged)} 行)")
    return added


# ---------------------------------------------------------------------------
# 既存テキストレポートからのバックフィル
# ---------------------------------------------------------------------------
def backfill_from_text(results_dir: Path = RESULTS_DIR_DEFAULT) -> pd.DataFrame:
    """results/signals_*.txt を解析して履歴 CSV を再構築する。

    旧フォーマットには signal_date (データの日付) が無く実行日のみのため、
    - 全銘柄の終値が前回と同一の実行 (休場日の再実行) を除外し、
    - signal_date = 実行日の前日 (米国市場の終値確定日) として近似する。
    """
    rows = []
    for path in sorted(results_dir.glob("signals_*.txt")):
        m = re.search(r"signals_(\d{4}-\d{2}-\d{2})", path.name)
        if not m:
            continue
        run_date = m.group(1)
        strategy = None
        for line in path.read_text(encoding="utf-8").splitlines():
            sm = re.match(r"【(.+)】", line.strip())
            if sm:
                strategy = sm.group(1)
                continue
            lm = _TXT_LINE_RE.match(line)
            if lm and strategy:
                rows.append({
                    "run_date": run_date,
                    "ticker": lm.group(1),
                    "strategy": strategy,
                    "signal": _SIGNAL_MAP[lm.group(2)],
                    "close": float(lm.group(3)),
                    "rsi": float(lm.group(4)),
                    "deviation": float(lm.group(5)),
                })

    if not rows:
        logger.warning("バックフィル対象のテキストレポートが見つかりません")
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    df = pd.DataFrame(rows)

    # 休場日 (土日・祝日) の実行を除外: 終値パネルが前回保持分と完全一致なら新データなし
    kept_dates = []
    prev_panel = None
    for d in sorted(df["run_date"].unique()):
        panel = (df[df["run_date"] == d]
                 .pivot_table(index="ticker", columns="strategy", values="close")
                 .sort_index())
        if prev_panel is not None and panel.equals(prev_panel):
            continue
        kept_dates.append(d)
        prev_panel = panel

    df = df[df["run_date"].isin(kept_dates)].copy()
    # データの日付 ≒ 実行日の前日 (JST 朝の実行時点で確定している米国市場の終値)
    df["signal_date"] = (pd.to_datetime(df["run_date"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    df = df[HISTORY_COLUMNS]

    # 既存履歴とマージ (実測の signal_date を持つ新形式の行を優先)
    hist = load_history(results_dir)
    merged = pd.concat([df, hist], ignore_index=True)
    merged = merged.drop_duplicates(subset=["signal_date", "ticker", "strategy"], keep="last")
    save_history(merged, results_dir)
    logger.info(f"バックフィル完了: {len(kept_dates)} 営業日 / {len(merged)} 行")
    return merged


# ---------------------------------------------------------------------------
# フォワードリターン計測
# ---------------------------------------------------------------------------
def compute_onset_performance(hist: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """新規シグナル発生ごとのフォワードリターンを計算する。

    戻り値: 1行 = 1つの新規シグナル発生。fwd_{n}d 列は営業日ベースの
    その後のリターン (SELL は符号反転済み = プラスなら「正解」)。
    """
    if hist.empty:
        return pd.DataFrame()

    hist = hist.copy()
    hist["signal_date"] = pd.to_datetime(hist["signal_date"])

    # 終値パネル (戦略間で終値は同一なので任意の戦略から構築)
    px = (hist.pivot_table(index="signal_date", columns="ticker", values="close", aggfunc="last")
          .sort_index())

    onsets = []
    for (strategy, ticker), g in hist.groupby(["strategy", "ticker"]):
        g = g.sort_values("signal_date")
        prev_signal = None
        for _, r in g.iterrows():
            if r["signal"] != "HOLD" and r["signal"] != prev_signal:
                onsets.append(r)
            prev_signal = r["signal"]

    if not onsets:
        return pd.DataFrame()

    out = pd.DataFrame(onsets)
    series_cache = {t: px[t].dropna() for t in out["ticker"].unique()}
    for n in horizons:
        vals = []
        for r in out.itertuples():
            s = series_cache[r.ticker]
            i = s.index.searchsorted(r.signal_date, side="right") - 1
            if i < 0 or i + n >= len(s):
                vals.append(np.nan)
                continue
            ret = s.iloc[i + n] / s.iloc[i] - 1
            vals.append(ret if r.signal == "BUY" else -ret)
        out[f"fwd_{n}d"] = vals
    return out.reset_index(drop=True)


def summarize(onsets: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """戦略×シグナル×ホライズンごとの 件数 / 勝率 / 平均 / 中央値"""
    if onsets.empty:
        return pd.DataFrame()
    recs = []
    for (strategy, signal), g in onsets.groupby(["strategy", "signal"]):
        for n in horizons:
            vals = g[f"fwd_{n}d"].dropna()
            if vals.empty:
                continue
            recs.append({
                "strategy": strategy,
                "signal": signal,
                "horizon": n,
                "n": len(vals),
                "win_rate": 100.0 * (vals > 0).mean(),
                "avg_pct": 100.0 * vals.mean(),
                "median_pct": 100.0 * vals.median(),
            })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# レポート生成
# ---------------------------------------------------------------------------
def write_report(results_dir: Path = RESULTS_DIR_DEFAULT, recent_days: int = 30) -> Path | None:
    hist = load_history(results_dir)
    if hist.empty:
        logger.warning("履歴が空のためレポートをスキップ")
        return None

    onsets = compute_onset_performance(hist)
    summary = summarize(onsets)

    dates = pd.to_datetime(hist["signal_date"]).sort_values().unique()
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  シグナル実績レポート (新規シグナル発生ベース / 営業日換算)")
    lines.append(f"  対象期間: {pd.Timestamp(dates[0]).date()} 〜 {pd.Timestamp(dates[-1]).date()}"
                 f"  ({len(dates)} 営業日, {hist['ticker'].nunique()} 銘柄)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("※ SELL の勝率は「シグナル後に価格が下落した割合」")
    lines.append("※ 直近のシグナルはホライズン分の将来データが揃うまで集計対象外")

    if summary.empty:
        lines.append("\n(集計可能な新規シグナルがまだありません)")
    else:
        for strategy in summary["strategy"].unique():
            lines.append(f"\n【{strategy}】")
            lines.append(f"  {'シグナル':<6} {'日数':>4} {'件数':>4} {'勝率':>7} {'平均':>8} {'中央値':>8}")
            lines.append("  " + "-" * 50)
            sub = summary[summary["strategy"] == strategy]
            for signal in ["BUY", "SELL"]:
                for r in sub[sub["signal"] == signal].itertuples():
                    lines.append(
                        f"  {signal:<8} {r.horizon:>3}d {r.n:>4} {r.win_rate:>6.1f}% "
                        f"{r.avg_pct:>+7.2f}% {r.median_pct:>+7.2f}%"
                    )

    # 直近の新規シグナルと途中経過 (シグナルロジック変更後の検証用)
    if not onsets.empty:
        cutoff = pd.Timestamp(dates[-1]) - pd.Timedelta(days=recent_days)
        recent = onsets[onsets["signal_date"] >= cutoff].sort_values("signal_date", ascending=False)
        if not recent.empty:
            lines.append(f"\n【直近 {recent_days} 日の新規シグナル】")
            lines.append(f"  {'日付':<12} {'戦略':<10} {'銘柄':<7} {'シグナル':<5}"
                         f" {'5d':>8} {'10d':>8} {'20d':>8}")
            lines.append("  " + "-" * 62)
            for r in recent.itertuples():
                def fmt(v):
                    return f"{100*v:+7.2f}%" if pd.notna(v) else "      - "
                lines.append(
                    f"  {r.signal_date.date()}  {r.strategy:<10} {r.ticker:<7} {r.signal:<6}"
                    f" {fmt(r.fwd_5d)} {fmt(r.fwd_10d)} {fmt(r.fwd_20d)}"
                )

    lines.append("")
    path = results_dir / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"実績レポート生成 → {path}")
    return path


def update(rows: list[dict], results_dir: Path = RESULTS_DIR_DEFAULT) -> None:
    """batch_runner から呼ばれるエントリポイント: 追記 + レポート再生成"""
    ingest(rows, results_dir)
    write_report(results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "backfill":
        backfill_from_text()
        write_report()
    elif cmd == "report":
        write_report()
    else:
        print(__doc__)
        sys.exit(1)
