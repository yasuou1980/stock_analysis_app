import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def load_data(ticker, start_date_str, end_date_str):
    """Yahoo Financeから株価データを読み込む"""
    try:
        start = datetime.fromisoformat(start_date_str) if isinstance(start_date_str, str) else start_date_str
        end = datetime.fromisoformat(end_date_str) if isinstance(end_date_str, str) else end_date_str

        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if data.empty:
            logger.warning(f"No data returned for {ticker} ({start_date_str} - {end_date_str})")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.columns = [str(col).lower() for col in data.columns]
        data.dropna(inplace=True)
        return data

    except Exception as e:
        logger.error(f"Data loading error for {ticker}: {e}")
        return None
