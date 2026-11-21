"""手数料・スリッページを入れたときに、シグナルの優位性がどれだけ残るかの検証。

BUY 新規発生 (onset) の終値で買い、N 営業日後の終値で売る。比較対象は「同じ銘柄を
売買せずに持ち続けた場合」(その銘柄の N 日リターンの全期間平均。手数料なし)。
取引した分だけ往復の手数料がかかるので、純超過 = 超過 − 往復コスト。
95% 区間は日付単位のブートストラップ (同じ日の複数銘柄のシグナルは独立でないため)。

使い方: python3 cost_study.py  (results/ の履歴 CSV 2 本を使う。ネットワーク不要)
結果の読み方は docs/signal_accuracy_2026-07.md (2026-11-21) を参照。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from signal_tracker import adjust_for_splits  # noqa: E402
from swing import net_ret_pct as net_ret  # noqa: E402

# 片道コスト (%)。事前に決めた水準 (結果を見てから変えない)
COST_LEVELS = {
    "0.05% (米国の手数料無料ブローカーで大型ETF)": 0.05,
    "0.1% (標準)": 0.1,
    "0.2% (保守: 小型株・レバETF・寄り付きの滑り)": 0.2,
    "0.5% (日本のネット証券の米国株 標準手数料 0.495% 相当)": 0.5,
}
STRATEGIES = ["トレンドフォロー", "逆張り", "押し目候補"]
DESIGN_END = pd.Timestamp("2026-09-02")   # 押し目候補の設計に使ったデータの終わり


def load():
    a = pd.read_csv("results/signals_history_backfill.csv", parse_dates=["signal_date"])
    live = pd.read_csv("results/signals_history.csv", parse_dates=["signal_date"])
    allr = pd.concat([a, live[live.signal_date > a.signal_date.max()]])
    cls = a.drop_duplicates("ticker").set_index("ticker").ticker_class
    tickers = [t for t in cls.index if cls[t] != "inverse_lev"]   # インバース型は BUY 禁止 (G1)
    prices = (allr.pivot_table(index="signal_date", columns="ticker", values="close", aggfunc="last")
              .sort_index().apply(adjust_for_splits)[tickers])
    signals = {s: allr[allr.strategy == s].pivot_table(index="signal_date", columns="ticker", values="signal",
                                                        aggfunc="last").reindex(prices.index)[tickers]
               for s in STRATEGIES}
    return prices, signals


def trades_for(prices, sig, h):
    buy = sig == "BUY"
    onset = buy & ~buy.shift(1, fill_value=False)
    fwd = (prices.shift(-h) / prices - 1) * 100
    passive = fwd.mean()                       # 持ち続けた場合の N 日平均 (銘柄ごと)
    gross = fwd[onset].stack().dropna()
    df = gross.rename("gross").reset_index()
    df.columns = ["date", "ticker", "gross"]
    df["passive"] = df["ticker"].map(passive)
    return df


def boot_ci(df, col, reps=2000, seed=0):
    by_day = df.groupby("date")[col].agg(["sum", "size"])
    sums, sizes = by_day["sum"].to_numpy(), by_day["size"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(by_day), (reps, len(by_day)))
    boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    prices, signals = load()
    print(f"対象: {prices.shape[1]} 銘柄 (plain + long_lev) / {prices.index.min().date()} 〜 {prices.index.max().date()}")
    for h in (10, 20):
        print(f"\n■ 保有 {h} 営業日 (純超過 = コスト込みの損益 − 持ち続けた場合。単位 pt、[ ] は 95% 区間)")
        for s in STRATEGIES:
            df = trades_for(prices, signals[s], h)
            periods = (("全期間", df.date == df.date, prices.index == prices.index),
                       ("〜09-02", df.date <= DESIGN_END, prices.index <= DESIGN_END),
                       ("09-03〜", df.date > DESIGN_END, prices.index > DESIGN_END))
            for label, mk, days in periods:
                d = df.loc[mk].copy()
                if len(d) < 5:
                    continue
                d["ex0"] = d.gross - d.passive
                lo, hi = boot_ci(d, "ex0")
                parts = [f"{s:<8} {label:<6} n={len(d):3d} 週{len(d)/days.sum()*5:4.1f}件  コストなし {d.ex0.mean():+5.2f} [{lo:+.2f},{hi:+.2f}]"]
                for name, c in COST_LEVELS.items():
                    d["exc"] = net_ret(d.gross, c) - d.passive
                    lo, hi = boot_ci(d, "exc")
                    parts.append(f"片道{c:g}% {d.exc.mean():+5.2f} [{lo:+.2f},{hi:+.2f}]")
                # 損益分岐: 純超過が 0 になる片道コスト (超過 ≒ 往復コスト)
                be = d.ex0.mean() / 2
                parts.append(f"損益分岐 片道 {be:.2f}%" if be > 0 else "損益分岐 なし (コスト前から負)")
                print("  " + " | ".join(parts))
    print("\n※ コスト水準:", " / ".join(COST_LEVELS))


if __name__ == "__main__":
    main()
