#!/usr/bin/env python3
"""
バッチ実行スクリプト
定時実行でシグナルを計算し、SQLite に蓄積する。
Streamlit に依存しない独立スクリプトとして動作する。

使い方:
    python batch_runner.py              # 通常実行
    python batch_runner.py --dry-run    # DB 保存せず標準出力のみ
"""
import sys
import argparse
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import toml
import yfinance as yf

# --- Streamlit をモックして非 UI 環境で動作させる ---
# @st.cache_data だけはパススルーデコレータとして差し替え、
# それ以外の st.error / st.info 等は MagicMock で無害化する
_mock_st = MagicMock()

def _passthrough_cache(func=None, **kwargs):
    """@st.cache_data を「何もしないデコレータ」として振る舞わせる"""
    if func is not None:
        return func                    # @st.cache_data (引数なし)
    return lambda f: f                 # @st.cache_data(ttl=...) (引数あり)

_mock_st.cache_data = _passthrough_cache
sys.modules["streamlit"] = _mock_st

from backtester import calculate_indicators_and_signals, resolve_ticker_class  # noqa: E402
import signal_tracker  # noqa: E402

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR_DEFAULT = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# ロギング設定 (ファイル + 標準出力)
# ---------------------------------------------------------------------------
def setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "batch.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データ取得 (Streamlit なし)
# ---------------------------------------------------------------------------
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        data.dropna(inplace=True)
        return data
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# シグナル計算
# ---------------------------------------------------------------------------
def compute_signals(raw_data: pd.DataFrame, params: dict, strategy: str) -> pd.DataFrame:
    """シグナルを計算する (cache_data はパススルー済みなので直接呼び出し可)"""
    return calculate_indicators_and_signals(None, raw_data.copy(), params, strategy)


# ---------------------------------------------------------------------------
# DB 保存
# ---------------------------------------------------------------------------
def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date      TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            signal_date   TEXT NOT NULL,
            strategy      TEXT NOT NULL,
            close         REAL,
            composite_signal TEXT,
            rsi           REAL,
            deviation     REAL,
            score         REAL,
            adx           REAL,
            ret_5d        REAL,
            ticker_class  TEXT,
            UNIQUE(run_date, ticker, strategy)
        )
    """)
    # 既存 DB へのカラム追加 (無ければ足す)
    existing = {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
    for col, coltype in [("score", "REAL"), ("adx", "REAL"),
                         ("ret_5d", "REAL"), ("ticker_class", "TEXT")]:
        if col not in existing:
            conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {coltype}")
    conn.commit()


def save_to_db(db_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with sqlite3.connect(db_path) as conn:
        ensure_table(conn)
        conn.executemany("""
            INSERT OR REPLACE INTO signals
                (run_date, ticker, signal_date, strategy, close, composite_signal, rsi, deviation,
                 score, adx, ret_5d, ticker_class)
            VALUES
                (:run_date, :ticker, :signal_date, :strategy, :close, :composite_signal, :rsi, :deviation,
                 :score, :adx, :ret_5d, :ticker_class)
        """, rows)
        conn.commit()


# ---------------------------------------------------------------------------
# テキスト書き出し
# ---------------------------------------------------------------------------
def save_to_text(results_dir: Path, run_date: str, rows: list[dict], errors: list[str]) -> None:
    """日付ごとのテキストファイルに結果を書き出す"""
    text_path = results_dir / f"signals_{run_date}.txt"
    df = pd.DataFrame(rows)

    SIGNAL_LABEL = {"BUY": "🟢 BUY ", "SELL": "🔴 SELL", "HOLD": "⚪ HOLD"}

    lines: list[str] = []
    lines.append(f"{'=' * 60}")
    lines.append(f"  株式シグナルレポート  {run_date}")
    lines.append(f"{'=' * 60}")

    for strategy in ["トレンドフォロー", "逆張り"]:
        sub = df[df["strategy"] == strategy].copy()
        lines.append(f"\n【{strategy}】")
        lines.append(f"  {'銘柄':<8} {'シグナル':<10} {'終値':>10}  {'RSI':>6}  {'乖離率':>8}")
        lines.append(f"  {'-' * 50}")

        # BUY → SELL → HOLD の順に表示
        for sig_key in ["BUY", "SELL", "HOLD"]:
            for _, r in sub[sub["composite_signal"] == sig_key].iterrows():
                label = SIGNAL_LABEL.get(sig_key, sig_key)
                dev = f"{r['deviation']:+.2f}%" if not np.isnan(r["deviation"]) else "   N/A"
                lines.append(
                    f"  {r['ticker']:<8} {label:<10} {r['close']:>10.2f}  {r['rsi']:>6.1f}  {dev:>8}"
                )

    if errors:
        lines.append(f"\n⚠️  取得失敗: {', '.join(errors)}")

    lines.append(f"\n{'=' * 60}\n")
    text = "\n".join(lines)

    text_path.write_text(text, encoding="utf-8")
    logging.getLogger(__name__).info(f"テキスト保存完了 → {text_path}")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def run(dry_run: bool = False, no_db: bool = False) -> None:
    config = toml.load(SCRIPT_DIR / "config.toml")
    batch_cfg = config.get("batch", {})

    tickers = batch_cfg.get("tickers", config.get("tickers", {}).get("default_tickers", []))
    lookback_days = int(batch_cfg.get("lookback_days", 365))
    results_dir = SCRIPT_DIR / batch_cfg.get("results_dir", "results")

    logger = setup_logging(results_dir)
    db_path = results_dir / "signals.db"

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=lookback_days)
    run_date = end_date.isoformat()

    # デフォルト分析パラメータ (スイングトレード設定)
    params = {
        "short_window": 10, "long_window": 40, "rsi_period": 10,
        "macd_fast": 10, "macd_slow": 20, "macd_signal": 7,
        "bb_length": 20, "bb_std": 2.0, "stoch_k": 14, "stoch_d": 3,
        "dev_upper": 10, "dev_lower": -10,
        "rsi_upper": 70, "rsi_lower": 30,
        "stoch_upper": 80, "stoch_lower": 20,
        "score_smooth_period": 3, "ema_slope_period": 5,
    }
    strategies = ["トレンドフォロー", "逆張り"]

    logger.info(f"=== バッチ開始: {run_date} | {len(tickers)} 銘柄 ===")
    rows: list[dict] = []
    errors: list[str] = []

    for ticker in tickers:
        raw_data = load_data(ticker, start_date.isoformat(), end_date.isoformat())
        if raw_data is None:
            logger.warning(f"  {ticker}: データ取得失敗、スキップ")
            errors.append(ticker)
            continue

        ticker_class = resolve_ticker_class(ticker, config)
        ticker_params = {**params, "ticker_class": ticker_class}

        for strategy in strategies:
            try:
                data = compute_signals(raw_data, ticker_params, strategy)
                if data.empty:
                    continue
                latest = data.iloc[-1]
                score_col = "counter_score" if strategy == "逆張り" else "trend_score"
                ret_5d = data["close"].pct_change(5).iloc[-1] if len(data) > 5 else np.nan
                row = {
                    "run_date":         run_date,
                    "ticker":           ticker,
                    "signal_date":      str(data.index[-1].date()),
                    "strategy":         strategy,
                    "close":            round(float(latest["close"]), 4),
                    "composite_signal": str(latest["composite_signal"]),
                    "rsi":              round(float(latest.get("rsi", np.nan)), 2),
                    "deviation":        round(float(latest.get("deviation", np.nan)), 4),
                    # 検証用特徴量 (シグナル精度のバケット分析に使う)
                    "score":            round(float(latest.get(score_col, np.nan)), 3),
                    "adx":              round(float(latest.get("ADX_14", np.nan)), 2),
                    "ret_5d":           round(float(ret_5d), 4) if pd.notna(ret_5d) else np.nan,
                    "ticker_class":     ticker_class,
                }
                rows.append(row)
                sig = row["composite_signal"]
                logger.info(f"  {ticker:6s} [{strategy}] {sig}")
            except Exception as e:
                logger.error(f"  {ticker} / {strategy} 計算エラー: {e}", exc_info=True)
                errors.append(f"{ticker}/{strategy}")

    if dry_run:
        logger.info("--- dry-run: DB・テキスト保存をスキップ ---")
        print(pd.DataFrame(rows).to_string(index=False))
    elif no_db:
        save_to_text(results_dir, run_date, rows, errors)
    else:
        save_to_db(db_path, rows)
        logger.info(f"DB 保存完了: {len(rows)} 件 → {db_path}")
        save_to_text(results_dir, run_date, rows, errors)

    # シグナル実績トラッキング (履歴 CSV 追記 + 精度レポート再生成)
    # 失敗してもバッチ本体は成功扱いにする
    if not dry_run:
        try:
            signal_tracker.update(rows, results_dir)
        except Exception as e:
            logger.error(f"実績トラッキング更新エラー: {e}", exc_info=True)

    logger.info(f"=== バッチ完了: 成功 {len(rows)} 件 / エラー {len(errors)} 件 ===")
    if errors:
        logger.warning(f"エラー銘柄: {errors}")


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="株式シグナルバッチ実行")
    parser.add_argument("--dry-run", action="store_true", help="DB・テキスト保存せず結果を標準出力のみ")
    parser.add_argument("--no-db",   action="store_true", help="DB 保存をスキップしてテキストのみ保存 (CI 環境向け)")
    args = parser.parse_args()
    run(dry_run=args.dry_run, no_db=args.no_db)
