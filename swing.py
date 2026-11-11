"""スイングトレード用の売買プランと保有ポジション管理 (Streamlit 非依存)。

シグナル (BUY/HOLD) を「実際に発注できる形」にするための計算をまとめる:

- 売買プラン: 損切り価格 (ATR の倍数)・株数 (1 トレードの許容損失から逆算)・手仕舞い予定日
- スイング候補: 監視銘柄のうち最新バーが BUY のものを、プラン付きで一覧化
- 保有ポジションの判定: 損切りライン到達 / 期間満了 / 保有継続

前提にしている運用 (swing_study.py で検証した形):
    シグナル確定日の翌営業日の寄り付きで買い、損切りは 買値 − ATR×倍率、
    損切りに掛からなければ N 営業日目の引けで手仕舞う。
利確目標とトレイリングストップは置かない。2022〜2026 年・27 銘柄の検証でどちらも成績を
改善しなかったため (docs/signal_accuracy_2026-07.md 2026-11-11)。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# 日次バッチ・バックフィル・アプリの候補一覧で共通のシグナル計算パラメータ。
# 実績レポート (results/performance_report.txt) はこの設定のシグナルを計測している
SIGNAL_PARAMS = {
    "short_window": 10, "long_window": 40, "rsi_period": 10,
    "macd_fast": 10, "macd_slow": 20, "macd_signal": 7,
    "bb_length": 20, "bb_std": 2.0, "stoch_k": 14, "stoch_d": 3,
    "dev_upper": 10, "dev_lower": -10,
    "rsi_upper": 70, "rsi_lower": 30,
    "stoch_upper": 80, "stoch_lower": 20,
    "score_smooth_period": 3, "ema_slope_period": 5,
}

# 候補一覧に出す戦略 (表示順)。押し目候補を先頭にしているのは、検証で
# 銘柄補正後もプラスが残った唯一の戦略のため
SWING_STRATEGIES = ["押し目候補", "トレンドフォロー", "逆張り"]

DEFAULT_SETTINGS = {
    "account_size": 10000.0,   # 資金 ($)
    "risk_pct": 1.0,           # 1 トレードで失ってよい資金の割合 (%)
    "stop_atr_mult": 2.0,      # 損切り幅 = ATR(14) × この倍率
    "max_hold_days": 10,       # 買った日を 1 日目として、この営業日目の引けで手仕舞う
    "max_position_pct": 25.0,  # 1 銘柄の建玉上限 (資金に対する %)
}

POSITION_COLUMNS = ["ticker", "entry_date", "entry_price", "shares", "stop_price", "note"]
POSITIONS_FILE = "positions.csv"

STATUS_STOP = "損切りライン到達"
STATUS_TIME = "期間満了"
STATUS_HOLD = "保有継続"
STATUS_NO_DATA = "データなし"


def swing_settings(config: dict | None) -> dict:
    """config.toml の [swing] を既定値に重ねて返す。"""
    merged = dict(DEFAULT_SETTINGS)
    for key, value in ((config or {}).get("swing", {}) or {}).items():
        if key in merged:
            merged[key] = type(DEFAULT_SETTINGS[key])(value)
    return merged


def atr_series(data: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR (Wilder)。backtester._atr と同じ式 (あちらは Streamlit を import するため再実装)。"""
    prev_close = data["close"].shift(1)
    true_range = pd.concat([
        data["high"] - data["low"],
        (data["high"] - prev_close).abs(),
        (data["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def buy_streak(signals: pd.Series) -> int:
    """末尾から連続している BUY の本数。最新バーが BUY でなければ 0、1 なら当日が新規発生。"""
    values = signals.to_numpy()
    n = 0
    for value in values[::-1]:
        if value != "BUY":
            break
        n += 1
    return n


def add_trading_days(date, n: int) -> pd.Timestamp:
    """n 営業日後の日付 (土日のみ考慮。米国の祝日は見ていないので 1 日前後する目安)。"""
    return pd.Timestamp(date) + pd.tseries.offsets.BDay(n)


def trade_plan(entry: float, atr: float, settings: dict) -> dict | None:
    """買値と ATR から損切り価格と株数を決める。

    株数は「損切りに掛かったときの損失 = 資金 × risk_pct」になるよう逆算し、
    建玉が資金の max_position_pct を超える場合はそこで頭打ちにする。
    entry / atr が不正なら None。
    """
    if not (entry and atr) or not (np.isfinite(entry) and np.isfinite(atr)) or entry <= 0 or atr <= 0:
        return None
    risk_per_share = atr * settings["stop_atr_mult"]
    stop = entry - risk_per_share
    if stop <= 0:
        return None
    account = settings["account_size"]
    by_risk = account * settings["risk_pct"] / 100.0 / risk_per_share
    by_cap = account * settings["max_position_pct"] / 100.0 / entry
    shares = int(math.floor(min(by_risk, by_cap)))
    return {
        "entry": entry,
        "stop": stop,
        "stop_pct": -100.0 * risk_per_share / entry,
        "risk_per_share": risk_per_share,
        "shares": shares,
        "position_value": shares * entry,
        "max_loss": shares * risk_per_share,
        # 建玉上限で株数が削られた = 損切り時の損失は risk_pct より小さくなる
        "capped": by_cap < by_risk,
        # 資金額に依らない目安: 許容損失 risk_pct に対応する建玉比率 (%)
        "position_pct": min(100.0 * settings["risk_pct"] / (100.0 * risk_per_share / entry),
                            settings["max_position_pct"]),
    }


def scan_candidates(frames: dict, ticker_classes: dict, settings: dict) -> list[dict]:
    """シグナル計算済みデータから、最新バーが BUY の銘柄をプラン付きで列挙する。

    frames: {(ticker, strategy): calculate_indicators_and_signals の戻り値}
    戻り値は 戦略の表示順 → 新規発生が先 → 銘柄名 の順。
    """
    out = []
    for (ticker, strategy), data in frames.items():
        if data is None or data.empty or "composite_signal" not in data.columns:
            continue
        streak = buy_streak(data["composite_signal"])
        if streak == 0:
            continue
        latest = data.iloc[-1]
        close = float(latest["close"])
        atr = float(latest.get("atr", np.nan))
        plan = trade_plan(close, atr, settings)
        if plan is None:
            continue
        signal_date = data.index[-1]
        out.append({
            "ticker": ticker,
            "strategy": strategy,
            "ticker_class": ticker_classes.get(ticker, "plain"),
            "signal_date": pd.Timestamp(signal_date).date(),
            "streak": streak,
            "close": close,
            "rsi": float(latest.get("rsi", np.nan)),
            "atr": atr,
            "atr_pct": 100.0 * atr / close,
            # 買うのは翌営業日の寄り付き。その日を 1 日目として max_hold_days 日目に手仕舞う
            "exit_date": add_trading_days(signal_date, settings["max_hold_days"]).date(),
            **plan,
        })
    order = {s: i for i, s in enumerate(SWING_STRATEGIES)}
    out.sort(key=lambda c: (order.get(c["strategy"], len(order)), c["streak"], c["ticker"]))
    return out


def market_context(prices: dict, tickers=("SPY", "QQQ")) -> list[dict]:
    """指数 ETF の終値と 200 日単純平均の位置関係 (地合いの参考表示用)。"""
    out = []
    for ticker in tickers:
        data = prices.get(ticker)
        if data is None or len(data) < 200:
            continue
        close = float(data["close"].iloc[-1])
        sma200 = float(data["close"].rolling(200).mean().iloc[-1])
        out.append({
            "ticker": ticker,
            "close": close,
            "sma200": sma200,
            "gap_pct": 100.0 * (close / sma200 - 1),
            "above": close > sma200,
        })
    return out


# ---------------------------------------------------------------------------
# 保有ポジション
# ---------------------------------------------------------------------------
def load_positions(path: Path | str = POSITIONS_FILE) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=POSITION_COLUMNS)
    df = pd.read_csv(path, dtype={"ticker": str, "note": str}).reindex(columns=POSITION_COLUMNS)
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.date
    return df


def clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    """編集画面の入力を保存できる形に整える (銘柄・買付日・買値が揃わない行は捨てる)。"""
    df = df.reindex(columns=POSITION_COLUMNS).copy()
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce").dt.date
    for col in ("entry_price", "shares", "stop_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["ticker"].notna() & (df["ticker"] != "") & df["entry_date"].notna()
            & (df["entry_price"] > 0)]
    return df.reset_index(drop=True)


def save_positions(df: pd.DataFrame, path: Path | str = POSITIONS_FILE) -> pd.DataFrame:
    cleaned = clean_positions(df)
    cleaned.to_csv(Path(path), index=False)
    return cleaned


def evaluate_position(prices: pd.DataFrame | None, position: dict, settings: dict) -> dict:
    """保有ポジション 1 件の現状と、今日やることを判定する。

    prices: その銘柄の日足 OHLC (index = 日付)。position: POSITION_COLUMNS の dict。
    stop_price が空なら 買値 − 買付日前営業日の ATR × stop_atr_mult を使う。
    判定の優先順は 損切りライン到達 > 期間満了 > 保有継続。
    """
    ticker = position["ticker"]
    entry_price = float(position["entry_price"])
    shares = position.get("shares")
    shares = float(shares) if shares is not None and pd.notna(shares) else 0.0
    entry_date = pd.Timestamp(position["entry_date"])
    result = {
        "ticker": ticker, "entry_date": entry_date.date(), "entry_price": entry_price, "shares": shares,
        "note": position.get("note") if pd.notna(position.get("note")) else "",
        "status": STATUS_NO_DATA, "detail": "価格データを取得できませんでした",
    }
    if prices is None or prices.empty:
        return result

    held = prices[prices.index >= entry_date]
    if held.empty:
        result["detail"] = "買付日以降の価格データがまだありません"
        return result

    stop = position.get("stop_price")
    stop_is_auto = stop is None or pd.isna(stop) or float(stop) <= 0
    if stop_is_auto:
        before = atr_series(prices)[prices.index < entry_date].dropna()
        if before.empty:
            result["detail"] = "買付日より前のデータが足りず、損切り価格を自動計算できません"
            return result
        stop = entry_price - float(before.iloc[-1]) * settings["stop_atr_mult"]
    stop = float(stop)

    last_close = float(held["close"].iloc[-1])
    days_in_trade = len(held)  # 買付日を 1 日目とする営業日数
    max_hold = int(settings["max_hold_days"])
    risk_per_share = entry_price - stop

    # 損切り判定: 買付日当日はザラ場のどの時点で買ったか分からないため終値で、
    # 翌日以降は安値で判定する
    hit = held["low"] <= stop
    hit.iloc[0] = held["close"].iloc[0] <= stop
    stop_hit_date = hit.index[hit.to_numpy().argmax()].date() if hit.any() else None

    if stop_hit_date is not None:
        status = STATUS_STOP
        detail = f"{stop_hit_date} に損切り価格 ${stop:.2f} を割りました"
    elif days_in_trade >= max_hold:
        status = STATUS_TIME
        detail = f"保有 {days_in_trade} 営業日目。予定の {max_hold} 日に達したので手仕舞い"
    else:
        status = STATUS_HOLD
        detail = f"保有 {days_in_trade} 営業日目 (残り {max_hold - days_in_trade} 営業日)"

    result.update({
        "status": status,
        "detail": detail,
        "last_date": held.index[-1].date(),
        "last_close": last_close,
        "stop": stop,
        "stop_is_auto": bool(stop_is_auto),
        "days_in_trade": days_in_trade,
        "exit_date": add_trading_days(held.index[0], max_hold - 1).date(),
        "pnl_pct": 100.0 * (last_close / entry_price - 1),
        "pnl": (last_close - entry_price) * shares,
        "r_multiple": (last_close - entry_price) / risk_per_share if risk_per_share > 0 else np.nan,
        "stop_hit_date": stop_hit_date,
    })
    return result


# ---------------------------------------------------------------------------
# 売買プランの過去シミュレーション
# ---------------------------------------------------------------------------
TRADE_COLUMNS = ["signal_date", "entry_date", "entry", "stop", "exit_date", "exit", "reason",
                 "ret_pct", "r_multiple", "days"]


def simulate_plan(prices: pd.DataFrame, entry_bars, stop_atr_mult: float, max_hold_days: int,
                  target_r: float | None = None, trail: bool = False) -> pd.DataFrame:
    """売買プランどおりに売買した場合の 1 トレードごとの結果。

    prices: 日足 OHLC。entry_bars: シグナル確定バーの位置 (prices の行番号) の並び。
    各シグナルについて 翌バーの始値で買い → 安値が損切り価格に触れたらその価格
    (窓を開けて下で寄った日は始値) で売り → 触れなければ max_hold_days 日目の終値で売る。
    手仕舞い日までデータが揃っていないシグナルは結果が未確定なので含めない。
    手数料・スリッページは含まない。

    target_r / trail は比較検証用 (swing_study.py)。target_r は 買値 + リスク × target_r の
    指値利確、trail は 終値の最高値 − ATR × 倍率 への損切り引き上げ。どちらも成績を
    改善しなかったので、画面の売買プランでは使っていない。同じ日に損切りと利確の
    両方に届いた場合は損切りを先に適用する。
    """
    o, h, l, c = (prices[x].to_numpy(dtype=float) for x in ("open", "high", "low", "close"))
    atr = atr_series(prices).to_numpy()
    index = prices.index
    trades = []
    for i in entry_bars:
        last = i + max_hold_days
        if last >= len(c) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        entry = o[i + 1]
        risk = atr[i] * stop_atr_mult
        initial_stop = stop = entry - risk
        target = entry + risk * target_r if target_r else np.inf
        highest = entry
        exit_price, exit_bar, reason = c[last], last, "time"
        for d in range(i + 1, last + 1):
            if l[d] <= stop:
                exit_price, exit_bar, reason = min(o[d], stop), d, "stop"
                break
            if h[d] >= target:
                exit_price, exit_bar, reason = max(o[d], target), d, "target"
                break
            if trail and np.isfinite(atr[d]):
                highest = max(highest, c[d])
                stop = max(stop, highest - atr[d] * stop_atr_mult)
        trades.append({
            "signal_date": index[i], "entry_date": index[i + 1], "entry": entry, "stop": initial_stop,
            "exit_date": index[exit_bar], "exit": exit_price, "reason": reason,
            "ret_pct": 100.0 * (exit_price / entry - 1),
            "r_multiple": (exit_price - entry) / risk,
            "days": exit_bar - i,
        })
    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


def onset_bars(signals: pd.Series, prices: pd.DataFrame) -> list[int]:
    """BUY が新規発生したバー (前バーが BUY でない) の、prices 上の行番号。"""
    is_buy = (signals == "BUY")
    onset = is_buy & ~is_buy.shift(1, fill_value=False)
    positions = prices.index.get_indexer(signals.index[onset.to_numpy()])
    return [int(p) for p in positions if p >= 0]


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    return {
        "n": len(trades),
        "win_rate": 100.0 * (trades["ret_pct"] > 0).mean(),
        "avg_ret_pct": trades["ret_pct"].mean(),
        "avg_r": trades["r_multiple"].mean(),
        "stop_rate": 100.0 * (trades["reason"] == "stop").mean(),
        "worst_pct": trades["ret_pct"].min(),
    }


# ---------------------------------------------------------------------------
# 日次テキストレポート用
# ---------------------------------------------------------------------------
def format_candidates_text(candidates: list[dict], settings: dict) -> list[str]:
    """日次レポートに追記するスイング候補の行。

    レポートは git に公開コミットされるため資金額・株数は載せず、
    資金に対する建玉比率 (%) で表す。行頭は signal_tracker の解析用正規表現
    (銘柄 + 絵文字付きシグナル) に一致しない形にしてある。
    """
    lines = [
        "",
        "■ スイング候補 (翌営業日の寄り付きで買う前提)",
        f"  損切り = 終値 − ATR×{settings['stop_atr_mult']:g} / "
        f"建玉 = 損切り時の損失が資金の {settings['risk_pct']:g}% になる比率 "
        f"(上限 {settings['max_position_pct']:g}%) / 手仕舞い = {settings['max_hold_days']} 営業日目の引け",
    ]
    if not candidates:
        lines.append("  該当なし")
        return lines
    lines.append(f"  {'戦略':<10} {'銘柄':<6} {'状態':<8} {'終値':>9} {'損切り':>9} {'幅':>7} {'建玉':>6}  手仕舞い予定")
    lines.append(f"  {'-' * 72}")
    for c in candidates:
        state = "新規" if c["streak"] == 1 else f"継続{c['streak']}日目"
        lines.append(
            f"  ・{c['strategy']:<9} {c['ticker']:<6} {state:<8} {c['close']:>9.2f} {c['stop']:>9.2f} "
            f"{c['stop_pct']:>6.1f}% {c['position_pct']:>5.0f}%  {c['exit_date']}"
        )
    return lines
