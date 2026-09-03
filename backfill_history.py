#!/usr/bin/env python3
"""過去データからシグナル履歴を再構成する (統計的検証力を上げるための基盤)。

背景:
日次バッチの実績は 2026-04 開始で、2026-09 時点でも 98 営業日・約450 onset しかない。
銘柄が同一セクターで同時に動くため実質的な独立試行はさらに少なく、
「逆張りSELL は有効か」のような問いに答えられないまま時間だけが過ぎる
(1ヶ月待っても増えるのは 10-20 件)。待つのではなくサンプルを増やす。

このスクリプトは数年分の OHLCV を取得し、**現在のシグナルロジック**を
そのまま適用して onset 履歴を再構成する。出力は signals_history.csv と
同一スキーマなので、signal_tracker の集計関数がそのまま使える。

    python backfill_history.py --years 4
    → results/signals_history_backfill.csv

重要な制約 (解釈時に必ず考慮する):
- **本番履歴を汚染しない**: 出力は専用ファイル。signal_tracker.load_history()
  が読む signals_history.csv には一切触れない
- **生存者バイアス**: 銘柄リストは 2026 年時点で選ばれたもので、過去に遡ると
  「その後生き残った銘柄」だけを見ることになる。ただしベースレート比較
  (compute_base_rates) は同じユニバース・同じ期間の単純保有と比べるため、
  バイアスはシグナル側とベース側の両方に乗る = エッジへの影響は限定的
- **現行ロジックでの遡及適用**: 過去にこのシグナルが実際に出ていたわけではない。
  「今のルールが過去の相場でどう振る舞ったか」を測るものであり、
  当時のリアルタイム運用実績ではない
- 銘柄の上場前・ETF 設定前の期間は自動的に欠損として除外される
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import toml
import yfinance as yf

# --- Streamlit をモックして非 UI 環境で動作させる (batch_runner.py と同一手法) ---
_mock_st = MagicMock()


def _passthrough_cache(func=None, **kwargs):
    if func is not None:
        return func
    return lambda f: f


_mock_st.cache_data = _passthrough_cache
sys.modules["streamlit"] = _mock_st

from backtester import calculate_indicators_and_signals, resolve_ticker_class  # noqa: E402
import signal_tracker  # noqa: E402

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
BACKFILL_NAME = "signals_history_backfill.csv"

# 指標の計算に必要な助走期間 (EMA200 等が安定するまで)。
# この期間のシグナルは信頼できないため出力から除外する。
WARMUP_TRADING_DAYS = 250

STRATEGIES = ["トレンドフォロー", "逆張り"]

# batch_runner.run() と同一のデフォルトパラメータ (本番と同じ挙動を再現する)
PARAMS = {
    "short_window": 10, "long_window": 40, "rsi_period": 10,
    "macd_fast": 10, "macd_slow": 20, "macd_signal": 7,
    "bb_length": 20, "bb_std": 2.0, "stoch_k": 14, "stoch_d": 3,
    "dev_upper": 10, "dev_lower": -10,
    "rsi_upper": 70, "rsi_lower": 30,
    "stoch_upper": 80, "stoch_lower": 20,
    "score_smooth_period": 3, "ema_slope_period": 5,
}


def fetch(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """yfinance から日足を取得する (batch_runner.load_data と同一の正規化)"""
    try:
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).lower() for col in data.columns]
        data.dropna(inplace=True)
        return data if not data.empty else None
    except Exception as e:
        logger.warning(f"  {ticker}: 取得失敗 ({e})")
        return None


def rows_for_ticker(ticker: str, data: pd.DataFrame, ticker_class: str) -> list[dict]:
    """1銘柄 × 全戦略 × 全営業日のシグナル行を作る (助走期間は除外)"""
    out: list[dict] = []
    for strategy in STRATEGIES:
        params = {**PARAMS, "ticker_class": ticker_class}
        try:
            sig = calculate_indicators_and_signals(
                f"backfill-{ticker}-{strategy}", data.copy(), params, strategy
            )
        except Exception as e:
            logger.warning(f"  {ticker}/{strategy}: 計算失敗 ({e})")
            continue
        if sig is None or sig.empty:
            continue

        # ret_5d は切り出し前の系列で計算する (切り出し後だと先頭5本が NaN になる)
        score_col = "counter_score" if strategy == "逆張り" else "trend_score"
        ret5 = sig["close"].pct_change(5)

        # 助走期間の除外は日付で行う。calculate_indicators_and_signals が
        # 内部で先頭を落とすことがあるため、位置スライスだと二重に切られる。
        warmup_end = data.index[WARMUP_TRADING_DAYS]
        sig = sig[sig.index >= warmup_end]
        if sig.empty:
            continue

        for ts, r in sig.iterrows():
            out.append({
                "signal_date": ts.date().isoformat(),
                "ticker": ticker,
                "strategy": strategy,
                "signal": str(r["composite_signal"]),
                "close": round(float(r["close"]), 4),
                "rsi": round(float(r.get("rsi", np.nan)), 2),
                "deviation": round(float(r.get("deviation", np.nan)), 4),
                "score": round(float(r.get(score_col, np.nan)), 3),
                "adx": round(float(r.get("ADX_14", np.nan)), 2),
                "ret_5d": round(float(ret5.loc[ts]), 4) if pd.notna(ret5.loc[ts]) else np.nan,
                "ticker_class": ticker_class,
                "raw_signal": str(r.get("raw_signal", r["composite_signal"])),
            })
    return out


def run(years: int, results_dir: Path) -> Path | None:
    config = toml.load(SCRIPT_DIR / "config.toml")
    batch_cfg = config.get("batch", {})
    tickers = batch_cfg.get("tickers", config.get("tickers", {}).get("default_tickers", []))

    end_date = datetime.now().date()
    # 助走期間ぶん余分に取得する (暦日換算で約1.5倍)
    start_date = end_date - timedelta(days=int(years * 365 + WARMUP_TRADING_DAYS * 1.5))

    logger.info(f"=== バックフィル開始: {len(tickers)} 銘柄 / {start_date} 〜 {end_date} ===")
    rows: list[dict] = []
    failed: list[str] = []

    for ticker in tickers:
        data = fetch(ticker, start_date.isoformat(), end_date.isoformat())
        if data is None or len(data) <= WARMUP_TRADING_DAYS:
            n = 0 if data is None else len(data)
            logger.warning(f"  {ticker}: データ不足 ({n} 本) — スキップ")
            failed.append(ticker)
            continue
        ticker_class = resolve_ticker_class(ticker, config)
        got = rows_for_ticker(ticker, data, ticker_class)
        rows.extend(got)
        logger.info(f"  {ticker:6s} [{ticker_class:11s}] {len(got):5d} 行 "
                    f"({data.index[WARMUP_TRADING_DAYS].date()} 〜 {data.index[-1].date()})")

    if not rows:
        logger.error("出力できる行がありません")
        return None

    df = pd.DataFrame(rows).reindex(columns=signal_tracker.HISTORY_COLUMNS)
    df = df.sort_values(["signal_date", "strategy", "ticker"]).reset_index(drop=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / BACKFILL_NAME
    df.to_csv(path, index=False)

    onsets = signal_tracker.compute_onset_performance(df)
    logger.info(f"=== 完了: {len(df)} 行 / {df['ticker'].nunique()} 銘柄 / "
                f"{df['signal_date'].nunique()} 営業日 → {path}")
    logger.info(f"    onset {len(onsets)} 件 "
                f"(本番履歴の約 {len(onsets) / 450:.1f} 倍)")
    if failed:
        logger.warning(f"    取得できなかった銘柄: {failed}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="過去データからシグナル履歴を再構成する")
    ap.add_argument("--years", type=int, default=4, help="遡る年数 (既定: 4)")
    ap.add_argument("--results-dir", default=None, help="出力先 (既定: config.toml の results_dir)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    config = toml.load(SCRIPT_DIR / "config.toml")
    results_dir = (Path(args.results_dir) if args.results_dir
                   else SCRIPT_DIR / config.get("batch", {}).get("results_dir", "results"))
    if run(args.years, results_dir) is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
