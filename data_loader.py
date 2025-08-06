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
        
        st.info(f"📊 {ticker} のデータを取得中... ({start.strftime('%Y-%m-%d')} ～ {end.strftime('%Y-%m-%d')})")
        
        data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if data.empty: 
            st.error(f"❌ ティッカー {ticker} のデータが見つかりません。")
            return None
            
        if isinstance(data.columns, pd.MultiIndex): 
            data.columns = data.columns.get_level_values(0)
        
        data.columns = [str(col).lower() for col in data.columns]
        data.dropna(inplace=True)
        
        st.success(f"✅ {ticker} のデータを正常に取得しました ({len(data)} 日分)")
        return data
        
    except Exception as e:
        st.error(f"❌ データの読み込み中にエラーが発生しました: {e}")
        logger.error(f"Data loading error for {ticker}: {e}")
        return None
