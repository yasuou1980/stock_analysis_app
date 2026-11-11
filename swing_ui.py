"""スイングトレード用の画面 (今日の候補 / 保有ポジション / プラン検証)。

計算は swing.py、ここは Streamlit の表示だけを担当する。
"""
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

import signal_tracker
import swing
from backtester import calculate_indicators_and_signals, resolve_ticker_class
from data_loader import load_data

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
POSITIONS_PATH = SCRIPT_DIR / swing.POSITIONS_FILE
RESULTS_DIR = SCRIPT_DIR / "results"
# 日次バッチ (batch_runner.py) と同じ取得期間。期間を変えると指数平滑系の指標が
# わずかにずれ、日次レポートと画面でシグナルが食い違う
LOOKBACK_DAYS = 365
CLASS_LABELS = {"plain": "現物・ETF", "long_lev": "レバレッジ(ロング)", "inverse_lev": "レバレッジ(インバース)"}


def universe(config: dict) -> list[str]:
    """候補を探す銘柄 = バッチの監視銘柄 + アプリの既定銘柄 (重複なし・順序維持)。"""
    tickers = list(config.get("batch", {}).get("tickers", []))
    tickers += config.get("tickers", {}).get("default_tickers", [])
    return list(dict.fromkeys(tickers))


def settings_inputs(config: dict) -> dict:
    """資金・リスク設定の入力欄。値はセッション内で候補タブと保有タブが共有する。"""
    defaults = swing.swing_settings(config)
    with st.expander("⚙️ 売買プランの設定 (資金・損切り幅・保有日数)", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        settings = {
            "account_size": c1.number_input("資金 ($)", 100.0, 100_000_000.0, float(defaults["account_size"]),
                                            step=1000.0, key="swing_account"),
            "risk_pct": c2.number_input("1回の許容損失 (%)", 0.1, 10.0, float(defaults["risk_pct"]), step=0.1,
                                        key="swing_risk", help="損切りに掛かったとき、資金の何%を失ってよいか"),
            "stop_atr_mult": c3.number_input("損切り幅 (ATR倍率)", 0.5, 6.0, float(defaults["stop_atr_mult"]),
                                             step=0.5, key="swing_stop", help="損切り価格 = 買値 − ATR(14) × この倍率"),
            "max_hold_days": int(c4.number_input("保有日数 (営業日)", 2, 60, int(defaults["max_hold_days"]), step=1,
                                                 key="swing_hold", help="買った日を1日目として、この日目の引けで手仕舞う")),
            "max_position_pct": c5.number_input("1銘柄の上限 (%)", 1.0, 100.0, float(defaults["max_position_pct"]),
                                                step=5.0, key="swing_cap", help="資金に対する建玉の上限"),
        }
        st.caption("既定値は config.toml の [swing]。ここでの変更はこの画面を開いている間だけ有効です。")
    return settings


def _date_range():
    end = datetime.now().date()
    return (end - timedelta(days=LOOKBACK_DAYS)).isoformat(), end.isoformat()


def load_universe(tickers: list[str], config: dict):
    """全銘柄の価格とシグナルを計算する。戻り値: (prices, frames, 取得失敗銘柄)。"""
    start, end = _date_range()
    prices, frames, failed = {}, {}, []
    progress = st.progress(0.0, text="監視銘柄のデータを取得中...")
    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"取得・計算中: {ticker} ({i + 1}/{len(tickers)})")
        raw = load_data(ticker, start, end)
        if raw is None or raw.empty:
            failed.append(ticker)
            continue
        prices[ticker] = raw
        params = {**swing.SIGNAL_PARAMS, "ticker_class": resolve_ticker_class(ticker, config)}
        for strategy in swing.SWING_STRATEGIES:
            # 1 銘柄の計算エラーで画面全体を落とさない
            try:
                key = hashlib.sha256((str(raw.values.tobytes()) + str(params) + strategy).encode()).hexdigest()
                frames[(ticker, strategy)] = calculate_indicators_and_signals(key, raw, params, strategy)
            except Exception as e:
                logger.error(f"シグナル計算に失敗しました ({ticker}/{strategy}): {e}")
    progress.empty()
    return prices, frames, failed


@st.cache_data(show_spinner=False)
def _track_record(mtimes: tuple) -> pd.DataFrame:
    """戦略別 BUY の 10 営業日後実績。mtimes は CSV 更新でキャッシュを無効化するためのキー。"""
    rows = []
    for name, label in ((signal_tracker.HISTORY_NAME, "ライブ"), ("signals_history_backfill.csv", "過去検証")):
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        hist = pd.read_csv(path, dtype={"signal_date": str, "ticker": str, "strategy": str, "signal": str})
        hist = hist.reindex(columns=signal_tracker.HISTORY_COLUMNS)
        summary = signal_tracker.summarize(signal_tracker.compute_onset_performance(hist),
                                           base_rates=signal_tracker.compute_base_rates(hist))
        if summary.empty:
            continue
        summary = summary[(summary["signal"] == "BUY") & (summary["horizon"] == 10)]
        for r in summary.itertuples():
            rows.append({"source": label, "strategy": r.strategy, "n": r.n, "win_rate": r.win_rate,
                         "base": r.base_win_rate, "edge": r.edge_win_rate, "avg": r.avg_pct,
                         "period": f"{hist['signal_date'].min()} 〜 {hist['signal_date'].max()}"})
    return pd.DataFrame(rows)


def _render_track_record():
    mtimes = tuple((RESULTS_DIR / n).stat().st_mtime if (RESULTS_DIR / n).exists() else 0
                   for n in (signal_tracker.HISTORY_NAME, "signals_history_backfill.csv"))
    try:
        record = _track_record(mtimes)
    except Exception as e:
        logger.error(f"実績の集計に失敗しました: {e}")
        return
    if record.empty:
        return
    st.subheader("この戦略の BUY は当たってきたか")
    order = {s: i for i, s in enumerate(swing.SWING_STRATEGIES)}
    record = (record.assign(_src=record["source"].map({"過去検証": 0, "ライブ": 1}), _st=record["strategy"].map(order))
              .sort_values(["_src", "_st"]).drop(columns=["_src", "_st"]))
    st.dataframe(
        record.rename(columns={"source": "データ", "strategy": "戦略", "n": "件数", "win_rate": "10日後に上昇",
                               "base": "何もしない場合", "edge": "差", "avg": "平均リターン", "period": "期間"}),
        column_config={
            "10日後に上昇": st.column_config.NumberColumn(format="%.1f%%"),
            "何もしない場合": st.column_config.NumberColumn(
                format="%.1f%%", help="同じ期間に監視銘柄をただ持っていた場合に、10営業日後に上がっていた割合"),
            "差": st.column_config.NumberColumn(format="%+.1f pt", help="プラスで初めてシグナルに意味がある"),
            "平均リターン": st.column_config.NumberColumn(format="%+.2f%%"),
        },
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "「過去検証」は現在のロジックを過去 4 年に当てはめた結果、「ライブ」は実際に日次バッチが出したシグナルの結果 "
        "(古いロジックの時期を含む)。シグナル当日の終値で買い 10 営業日後の終値で売った場合で、損切りは入っていません。"
    )


def render_candidates(config: dict, settings: dict, prices: dict, frames: dict, failed: list[str]):
    tickers = universe(config)
    classes = {t: resolve_ticker_class(t, config) for t in tickers}
    candidates = swing.scan_candidates(frames, classes, settings)

    latest_dates = [df.index[-1] for df in prices.values() if len(df)]
    as_of = max(latest_dates).date() if latest_dates else None
    context = swing.market_context(prices)
    cols = st.columns(len(context) + 2)
    cols[0].metric("データの日付 (米国市場の終値)", str(as_of) if as_of else "-")
    cols[1].metric("今日の候補", f"{len(candidates)} 件", help=f"監視 {len(tickers)} 銘柄 × {len(swing.SWING_STRATEGIES)} 戦略")
    for col, m in zip(cols[2:], context):
        col.metric(f"{m['ticker']} (200日平均との差)", f"${m['close']:.2f}", f"{m['gap_pct']:+.1f}%")

    if failed:
        st.warning(f"データを取得できなかった銘柄: {', '.join(failed)} (時間を置いて再読み込みしてください)")

    st.subheader("買い候補と売買プラン")
    if not candidates:
        st.info("最新の終値で BUY シグナルが出ている銘柄はありません。")
    else:
        table = pd.DataFrame([{
            "銘柄": c["ticker"],
            "戦略": c["strategy"],
            "状態": "新規" if c["streak"] == 1 else f"継続 {c['streak']}日目",
            "終値": c["close"],
            "RSI": c["rsi"],
            "損切り価格": c["stop"],
            "損切り幅": c["stop_pct"],
            "株数": c["shares"],
            "建玉額": c["position_value"],
            "最大損失": c["max_loss"],
            "手仕舞い予定": c["exit_date"],
            "種別": CLASS_LABELS.get(c["ticker_class"], c["ticker_class"]),
        } for c in candidates])
        st.dataframe(
            table,
            column_config={
                "状態": st.column_config.TextColumn(help="新規 = 最新の終値で初めて BUY になった。過去の検証は新規発生日だけを対象にしている"),
                "終値": st.column_config.NumberColumn(format="$%.2f"),
                "RSI": st.column_config.NumberColumn(format="%.1f"),
                "損切り価格": st.column_config.NumberColumn(format="$%.2f", help="終値 − ATR × 倍率。実際は約定した買値から同じ幅を引く"),
                "損切り幅": st.column_config.NumberColumn(format="%.1f%%"),
                "株数": st.column_config.NumberColumn(format="%d", help="損切りに掛かったときの損失が許容損失に収まる株数"),
                "建玉額": st.column_config.NumberColumn(format="$%.0f"),
                "最大損失": st.column_config.NumberColumn(format="$%.0f", help="損切り価格で売れた場合。窓を開けて下で寄るとこれより大きくなる"),
            },
            hide_index=True, use_container_width=True,
        )
        if any(c["shares"] == 0 for c in candidates):
            st.caption("株数 0 の銘柄は、設定した資金と許容損失では 1 株も買えません。")
    st.caption(
        f"プランの前提: 翌営業日の寄り付きで買う / 損切り = 買値 − ATR×{settings['stop_atr_mult']:g} / "
        f"損切りに掛からなければ {settings['max_hold_days']} 営業日目の引けで売る。"
        "利確目標とトレイリングストップは、2022〜2026 年の検証で成績を改善しなかったため置いていません。"
        "手仕舞い予定日は米国の祝日を考慮していない目安です。"
    )

    _render_track_record()

    with st.expander("監視銘柄すべての状況"):
        rows = []
        for ticker in tickers:
            raw = prices.get(ticker)
            if raw is None:
                continue
            row = {"銘柄": ticker, "種別": CLASS_LABELS.get(classes[ticker], classes[ticker]),
                   "終値": float(raw["close"].iloc[-1])}
            sma200 = raw["close"].rolling(200).mean().iloc[-1]
            row["200日平均との差"] = 100.0 * (row["終値"] / sma200 - 1) if pd.notna(sma200) else None
            row["RSI"] = None
            for strategy in swing.SWING_STRATEGIES:
                df = frames.get((ticker, strategy))
                row[strategy] = df["composite_signal"].iloc[-1] if df is not None and not df.empty else "-"
                if df is not None and not df.empty:
                    row["RSI"] = float(df["rsi"].iloc[-1])
            rows.append(row)
        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                column_config={
                    "終値": st.column_config.NumberColumn(format="$%.2f"),
                    "200日平均との差": st.column_config.NumberColumn(format="%+.1f%%"),
                    "RSI": st.column_config.NumberColumn(format="%.1f"),
                },
                hide_index=True, use_container_width=True,
            )
        st.caption("SELL はインバース型レバレッジ ETF にしか出ません (それ以外は検証で優位性が無かったため)。")


def _editor_frame(positions: pd.DataFrame) -> pd.DataFrame:
    df = positions.reindex(columns=swing.POSITION_COLUMNS).copy()
    df["ticker"] = df["ticker"].astype("object")
    df["note"] = df["note"].astype("object")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    for col in ("entry_price", "shares", "stop_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def render_positions(config: dict, settings: dict, prices: dict):
    st.subheader("保有ポジション")
    saved = swing.load_positions(POSITIONS_PATH)

    if not saved.empty:
        start, end = _date_range()
        results = []
        for pos in saved.to_dict("records"):
            raw = prices.get(pos["ticker"])
            if raw is None:
                raw = load_data(pos["ticker"], start, end)
            results.append(swing.evaluate_position(raw, pos, settings))

        for r in results:
            if r["status"] == swing.STATUS_STOP:
                st.error(f"**{r['ticker']}** 損切り: {r['detail']}")
            elif r["status"] == swing.STATUS_TIME:
                st.warning(f"**{r['ticker']}** 手仕舞い: {r['detail']}")
            elif r["status"] == swing.STATUS_NO_DATA:
                st.warning(f"**{r['ticker']}** {r['detail']}")

        table = pd.DataFrame([{
            "銘柄": r["ticker"],
            "判定": r["status"],
            "買付日": r["entry_date"],
            "買値": r["entry_price"],
            "現在値": r.get("last_close"),
            "損益": r.get("pnl_pct"),
            "損益額": r.get("pnl"),
            "R": r.get("r_multiple"),
            "損切り価格": r.get("stop"),
            "保有日数": r.get("days_in_trade"),
            "手仕舞い予定": r.get("exit_date"),
            "メモ": r.get("note", ""),
        } for r in results])
        st.dataframe(
            table,
            column_config={
                "買値": st.column_config.NumberColumn(format="$%.2f"),
                "現在値": st.column_config.NumberColumn(format="$%.2f"),
                "損益": st.column_config.NumberColumn(format="%+.1f%%"),
                "損益額": st.column_config.NumberColumn(format="$%+.0f"),
                "R": st.column_config.NumberColumn(format="%+.2f", help="含み損益 ÷ 最初に取ったリスク (買値 − 損切り価格)。−1 で損切り水準"),
                "損切り価格": st.column_config.NumberColumn(format="$%.2f"),
                "保有日数": st.column_config.NumberColumn(format="%d 日目", help="買付日を 1 日目とする営業日数"),
            },
            hide_index=True, use_container_width=True,
        )
        st.caption(
            f"判定は 損切り価格を安値が割ったか → {settings['max_hold_days']} 営業日目に達したか の順。"
            "損切り価格は買ったときの水準で固定し、引き上げません (引き上げは過去検証で成績を下げたため)。"
        )
    else:
        st.info("保有ポジションはまだ登録されていません。下の表に入力して保存してください。")

    st.markdown("##### ポジションの登録・編集")
    edited = st.data_editor(
        _editor_frame(saved),
        num_rows="dynamic",
        column_config={
            "ticker": st.column_config.TextColumn("銘柄", required=True),
            "entry_date": st.column_config.DateColumn("買付日", required=True, format="YYYY-MM-DD"),
            "entry_price": st.column_config.NumberColumn("買値 ($)", required=True, min_value=0.0, format="%.2f"),
            "shares": st.column_config.NumberColumn("株数", min_value=0.0),
            "stop_price": st.column_config.NumberColumn(
                "損切り価格 ($)", min_value=0.0, format="%.2f",
                help=f"空欄なら 買値 − 買付前日の ATR × {settings['stop_atr_mult']:g} で自動計算"),
            "note": st.column_config.TextColumn("メモ"),
        },
        hide_index=True, use_container_width=True, key="positions_editor",
    )
    if st.button("保存", type="primary"):
        cleaned = swing.save_positions(edited, POSITIONS_PATH)
        dropped = len(edited) - len(cleaned)
        st.session_state["positions_saved_msg"] = (
            f"{len(cleaned)} 件を保存しました。"
            + (f" 銘柄・買付日・買値が揃っていない {dropped} 行は保存していません。" if dropped else ""))
        st.rerun()
    if msg := st.session_state.pop("positions_saved_msg", None):
        st.success(msg)
    st.caption(
        f"保存先はこのアプリのフォルダの {swing.POSITIONS_FILE} (git の管理対象外)。"
        "Streamlit Cloud で動かしている場合は再起動で消えるため、手元の PC で動かすときに使ってください。"
    )


def render_plan_check(raw: pd.DataFrame, data: pd.DataFrame, settings: dict, ticker: str, strategy: str):
    """選択中の銘柄・戦略で、売買プランどおりに売買していた場合の過去成績。"""
    st.subheader("この銘柄を売買プランどおりに売買していたら")
    k, n = settings["stop_atr_mult"], settings["max_hold_days"]
    trades = swing.simulate_plan(raw, swing.onset_bars(data["composite_signal"], raw), k, n)
    first = raw.index.get_loc(data.index[0])
    baseline = swing.simulate_plan(raw, range(first, len(raw)), k, n)
    s, b = swing.summarize_trades(trades), swing.summarize_trades(baseline)
    if s["n"] == 0:
        st.info(f"この期間に {strategy} の BUY が新規発生して手仕舞いまで完了した例がありません。")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("回数", f"{s['n']}")
    c2.metric("勝率", f"{s['win_rate']:.0f}%", f"{s['win_rate'] - b['win_rate']:+.0f}pt 対 毎日買い" if b["n"] else None)
    c3.metric("平均損益", f"{s['avg_ret_pct']:+.2f}%",
              f"{s['avg_ret_pct'] - b['avg_ret_pct']:+.2f}pt 対 毎日買い" if b["n"] else None)
    c4.metric("平均 R", f"{s['avg_r']:+.2f}")
    c5.metric("損切りになった割合", f"{s['stop_rate']:.0f}%")
    st.caption(
        f"BUY が新規発生した翌営業日の寄り付きで買い、ATR×{k:g} の損切りか {n} 営業日目の引けで売った場合 "
        "(手数料・スリッページ抜き)。「対 毎日買い」は同じ銘柄を同じルールで毎日買った場合との差で、"
        "これがプラスでないとシグナルを待つ意味がありません。1 銘柄だけでは回数が少なく、偶然の影響が大きい点に注意してください。"
    )
    with st.expander("トレード一覧"):
        show = trades.copy()
        for col in ("signal_date", "entry_date", "exit_date"):
            show[col] = pd.to_datetime(show[col]).dt.date
        show["reason"] = show["reason"].map({"stop": "損切り", "time": "期間満了"})
        st.dataframe(
            show.rename(columns={"signal_date": "シグナル日", "entry_date": "買付日", "entry": "買値", "stop": "損切り価格",
                                 "exit_date": "売却日", "exit": "売値", "reason": "理由", "ret_pct": "損益",
                                 "r_multiple": "R", "days": "日数"}),
            column_config={
                "買値": st.column_config.NumberColumn(format="$%.2f"),
                "損切り価格": st.column_config.NumberColumn(format="$%.2f"),
                "売値": st.column_config.NumberColumn(format="$%.2f"),
                "損益": st.column_config.NumberColumn(format="%+.2f%%"),
                "R": st.column_config.NumberColumn(format="%+.2f"),
            },
            hide_index=True, use_container_width=True,
        )
