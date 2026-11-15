"""Publish public market data from the daily analysis for the portfolio Site."""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import yfinance as yf


def write_snapshot(path, rows, tickers, strategies, errors):
    expected = {(ticker, strategy) for ticker in tickers for strategy in strategies}
    actual = {(row["ticker"], row["strategy"]) for row in rows}
    if not expected or errors or actual != expected or len(rows) != len(expected):
        raise ValueError("分析結果が未完了です")

    market_dates = {}
    for row in rows:
        previous = market_dates.setdefault(row["ticker"], row["signal_date"])
        if previous != row["signal_date"]:
            raise ValueError("戦略間で価格基準日が一致しません")

    history = yf.download(
        list(tickers), period="10d", auto_adjust=False, progress=False, timeout=15
    )
    prices = {}
    for ticker, date in market_dates.items():
        closes = history["Close"][ticker]
        matching = closes[closes.index.strftime("%Y-%m-%d") == date].dropna()
        if len(matching) != 1 or not math.isfinite(matching.iloc[0]) or matching.iloc[0] <= 0:
            raise ValueError(f"{ticker}: {date} の評価用終値がありません")
        price = format(Decimal(str(float(matching.iloc[0]))), ".6f")
        if Decimal(price) <= 0:
            raise ValueError(f"{ticker}: 評価用終値が精度範囲外です")
        prices[ticker] = price

    signals = []
    for row in rows:
        if row["composite_signal"] not in {"BUY", "HOLD", "SELL"}:
            raise ValueError("未対応のシグナルです")
        signals.append({
            "ticker": row["ticker"],
            "strategy": row["strategy"],
            "price_as_of": row["signal_date"],
            "close_usd": prices[row["ticker"]],
            "signal": row["composite_signal"],
            **{
                key: row[key] if math.isfinite(row[key]) else None
                for key in ("rsi", "deviation", "score", "adx", "ret_5d")
            },
        })

    snapshot = {
        "schema_version": 1,
        "run_id": os.environ["GITHUB_RUN_ID"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "signals": signals,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    staging.replace(path)
