#!/usr/bin/env python3
"""スイング用の売買プランが過去に成り立っていたかの検証。

各戦略の BUY 新規発生について「翌営業日の寄り付きで買い、ATR×k の損切りか
N 営業日目の引けで売る」を swing.simulate_plan で再現し、同じ銘柄を同じルールで
毎日買った場合 (ベースライン) との差を測る。銘柄ごとにベースラインを引くことで、
「上がり続けた銘柄を持っていただけ」の分を除く。

差の 95% 区間は日付単位のブートストラップ (同じ日に複数銘柄で出たシグナルは
独立でないため、日付ごとにまとめて再標本化する)。

使い方:
    python swing_study.py                 # Yahoo から取得して検証
    python swing_study.py --years 5 --cache /tmp/prices.pkl   # 取得結果を保存・再利用

結果の読み方と結論は docs/signal_accuracy_2026-07.md (2026-11-11) を参照。
"""
import argparse
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import toml
import yfinance as yf

# --- Streamlit をモックして非 UI 環境で動作させる (batch_runner.py と同一手法) ---
_mock_st = MagicMock()
_mock_st.cache_data = lambda func=None, **kwargs: func if func is not None else (lambda f: f)
sys.modules["streamlit"] = _mock_st

from backtester import calculate_indicators_and_signals, resolve_ticker_class  # noqa: E402
import swing  # noqa: E402

SCRIPT_DIR = Path(__file__).parent
STRATEGIES = ["押し目候補", "トレンドフォロー", "逆張り"]
# (損切り ATR 倍率, 保有日数, 利確 R, トレイリング)。先頭 2 つが画面のプランとその 20 日版
VARIANTS = [
    (2.0, 10, None, False), (2.0, 20, None, False),
    (1.5, 10, None, False), (3.0, 10, None, False),
    (2.0, 10, 2.0, False), (2.0, 10, None, True),
    (2.0, 20, 2.0, False), (2.0, 20, None, True),
]


def fetch_all(tickers, years, cache: Path | None):
    if cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    end = datetime.now().date()
    start = end - timedelta(days=int(365.25 * years) + 30)
    out = {}
    for ticker in tickers:
        for attempt in range(3):
            try:
                data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                                   auto_adjust=True, progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.columns = [str(c).lower() for c in data.columns]
                data = data.dropna()
                if len(data):
                    out[ticker] = data
                    break
            except Exception as e:
                print(f"  {ticker}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    if cache:
        cache.write_bytes(pickle.dumps(out))
    return out


def collect(prices: dict, config: dict) -> pd.DataFrame:
    """全銘柄・全ルールのトレードを 1 つの表にする (strategy='毎日買い' がベースライン)。"""
    parts = []
    for ticker, raw in prices.items():
        ticker_class = resolve_ticker_class(ticker, config)
        if ticker_class == "inverse_lev":
            continue  # インバース型は BUY 禁止 (G1)
        params = {**swing.SIGNAL_PARAMS, "ticker_class": ticker_class}
        signals = {s: calculate_indicators_and_signals(None, raw.copy(), params, s)["composite_signal"]
                   for s in STRATEGIES}
        # 3 戦略すべての指標が揃う (= 200 日平均が計算できる) 日以降だけを比較対象にする
        first = raw.index.get_loc(max(sig.index[0] for sig in signals.values()))
        entries = {"毎日買い": range(first, len(raw))}
        entries.update({s: [i for i in swing.onset_bars(sig, raw) if i >= first] for s, sig in signals.items()})
        for k, n, target_r, trail in VARIANTS:
            for strategy, bars in entries.items():
                t = swing.simulate_plan(raw, bars, k, n, target_r=target_r, trail=trail)
                if t.empty:
                    continue
                parts.append(t.assign(ticker=ticker, ticker_class=ticker_class, strategy=strategy,
                                      k=k, n=n, target_r=target_r or 0.0, trail=trail))
    return pd.concat(parts, ignore_index=True)


def report(trades: pd.DataFrame, reps: int = 2000) -> pd.DataFrame:
    key = ["k", "n", "target_r", "trail"]
    base = (trades[trades["strategy"] == "毎日買い"].groupby(key + ["ticker"])["ret_pct"].mean()
            .rename("base_ret"))
    d = trades[trades["strategy"] != "毎日買い"].join(base, on=key + ["ticker"])
    d["excess"] = d["ret_pct"] - d["base_ret"]
    rng = np.random.default_rng(0)
    rows = []
    for (strategy, k, n, target_r, trail), g in d.groupby(["strategy"] + key):
        by_day = g.groupby("signal_date")["excess"].agg(["sum", "size"])
        sums, sizes = by_day["sum"].to_numpy(), by_day["size"].to_numpy()
        idx = rng.integers(0, len(by_day), (reps, len(by_day)))
        boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
        rows.append({
            "戦略": strategy, "損切りATR": k, "保有日数": n, "利確R": target_r or "-", "トレイル": "有" if trail else "-",
            "件数": len(g), "発生日数": len(by_day),
            "勝率%": (g["ret_pct"] > 0).mean() * 100, "平均%": g["ret_pct"].mean(),
            "差pt": g["excess"].mean(), "差95%下限": np.percentile(boot, 2.5), "差95%上限": np.percentile(boot, 97.5),
            "平均R": g["r_multiple"].mean(), "損切り率%": (g["reason"] == "stop").mean() * 100,
        })
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(STRATEGIES)}
    variant_order = {v: i for i, v in enumerate((k, n, t or 0.0, tr) for k, n, t, tr in VARIANTS)}
    out["_s"] = out["戦略"].map(order)
    out["_v"] = [variant_order[(r["損切りATR"], r["保有日数"], 0.0 if r["利確R"] == "-" else r["利確R"], r["トレイル"] == "有")]
                 for _, r in out.iterrows()]
    return out.sort_values(["_s", "_v"]).drop(columns=["_s", "_v"]).reset_index(drop=True)


def by_year(trades: pd.DataFrame, k=2.0, n=10) -> pd.DataFrame:
    sel = trades[(trades["k"] == k) & (trades["n"] == n) & (trades["target_r"] == 0) & (~trades["trail"])]
    base = sel[sel["strategy"] == "毎日買い"].groupby("ticker")["ret_pct"].mean().rename("base_ret")
    d = sel[sel["strategy"] != "毎日買い"].join(base, on="ticker")
    d["excess"] = d["ret_pct"] - d["base_ret"]
    d["year"] = pd.to_datetime(d["signal_date"]).dt.year
    table = d.pivot_table(index="strategy", columns="year", values="excess", aggfunc=["mean", "size"])
    return table.reindex(STRATEGIES)


def main() -> None:
    parser = argparse.ArgumentParser(description="スイング用売買プランの検証")
    parser.add_argument("--years", type=float, default=5.5, help="取得する年数 (先頭の約 200 営業日は指標の助走に使う)")
    parser.add_argument("--cache", type=Path, default=None, help="価格データの保存・再利用先 (pickle)")
    args = parser.parse_args()

    config = toml.load(SCRIPT_DIR / "config.toml")
    prices = fetch_all(config["batch"]["tickers"], args.years, args.cache)
    trades = collect(prices, config)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    span = trades["signal_date"].min().date(), trades["signal_date"].max().date()
    print(f"対象: {trades['ticker'].nunique()} 銘柄 / シグナル日 {span[0]} 〜 {span[1]}")
    print("\n■ ルール別 (差pt = 同じ銘柄を同じルールで毎日買った場合との差)")
    print(report(trades).round(2).to_string(index=False))
    print("\n■ 年別の差pt と件数 (損切り ATR×2 / 10 日)")
    print(by_year(trades).round(2).to_string())
    base = trades[(trades["strategy"] == "毎日買い") & (trades["target_r"] == 0) & (~trades["trail"])]
    print("\n■ 毎日買い (ベースライン)")
    print(base.groupby(["k", "n"]).agg(勝率=("ret_pct", lambda x: (x > 0).mean() * 100), 平均=("ret_pct", "mean"),
                                       平均R=("r_multiple", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
