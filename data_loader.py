import streamlit as st
import pandas as pd
import yfinance as yf
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_WAIT_SEC = 2
_DOWNLOAD_TIMEOUT_SEC = 120

# yfinance の HTTP バックエンド (curl_cffi) はスレッドセーフではなく、
# 複数スレッドから同じセッションに触れるとセグメンテーションフォルトで
# プロセスごと落ちる。Streamlit は再実行 (rerun) のたびに新しいスレッドで
# スクリプトを実行するため、yf.download を直接呼ぶと2回目以降の操作
# (銘柄切替等) でクラッシュする。すべてのダウンロードをこの専用スレッド
# 1本に閉じ込めることで、yfinance を常に同一スレッドから使う。
_YF_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yf-download")


def _download_in_worker(ticker, start, end):
    future = _YF_EXECUTOR.submit(
        yf.download, ticker, start=start, end=end, auto_adjust=True, progress=False
    )
    return future.result(timeout=_DOWNLOAD_TIMEOUT_SEC)


class DataFetchError(Exception):
    """データ取得の失敗。st.cache_data は例外を結果としてキャッシュしないため、
    失敗を None として1時間キャッシュしてしまう問題を避けるために使う。"""


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_data(ticker, start_date_str, end_date_str):
    """Yahoo Financeから株価データを取得する（成功した結果のみキャッシュされる）"""
    start = datetime.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str
    end = datetime.fromisoformat(end_date_str) if isinstance(end_date_str, str) else end_date_str

    last_error = None
    for attempt in range(_MAX_RETRIES):
        try:
            data = _download_in_worker(ticker, start, end)
        except Exception as e:
            last_error = e
            data = None

        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [str(col).lower() for col in data.columns]
            data.dropna(inplace=True)
            if not data.empty:
                return data

        # レート制限や一時的な障害の可能性があるため、間隔を空けて再試行
        if attempt < _MAX_RETRIES - 1:
            time.sleep(_RETRY_WAIT_SEC * (attempt + 1))

    raise DataFetchError(
        f"No data returned for {ticker} ({start_date_str} - {end_date_str})"
        + (f": {last_error}" if last_error else "")
    )


def load_data(ticker, start_date_str, end_date_str):
    """Yahoo Financeから株価データを読み込む。失敗時は None を返す。

    失敗（レート制限・一時的な通信エラー等）は例外として伝播させることで
    キャッシュに残さない。次回の呼び出しで再度取得を試みる。
    """
    try:
        return _fetch_data(ticker, start_date_str, end_date_str)
    except Exception as e:
        logger.error(f"Data loading error for {ticker}: {e}")
        return None
