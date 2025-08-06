import streamlit as st
from datetime import datetime
import toml
import tomli_w
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

@st.cache_data
def load_config(filename="config.toml"):
    """設定ファイルを読み込む"""
    try:
        with open(filename, 'r') as f:
            return toml.load(f)
    except (FileNotFoundError, toml.TomlDecodeError) as e:
        st.error(f"設定ファイル {filename} の読み込みに失敗しました: {e}")
        return {}

def save_settings(settings, filename="config.toml"):
    """現在の設定をTOMLファイルに保存する"""
    config = load_config(filename)
    settings_serializable = {k: v.isoformat() if isinstance(v, datetime.date) else v for k, v in settings.items()}
    config['user_settings'] = settings_serializable
    with open(filename, "wb") as f:
        tomli_w.dump(config, f)
    st.sidebar.success(f"設定を {filename} に保存しました。")

def load_settings(filename="config.toml"):
    """TOMLファイルから設定を読み込む"""
    config = load_config(filename)
    settings = config.get('user_settings', {})
    for key, value in settings.items():
        if key in ['start_date', 'end_date'] and isinstance(value, str):
            settings[key] = datetime.fromisoformat(value).date()
    return settings

def validate_date_range(start_date, end_date):
    """日付範囲の入力を検証する"""
    if start_date >= end_date:
        st.error("開始日は終了日より前の日付を選択してください。")
        return False
    return True

def validate_data_quality(data):
    """データ品質の包括的チェック"""
    issues = []
    if (data['high'] < data['low']).any(): 
        issues.append("⚠️ 価格データに矛盾があります（高値 < 安値）")
    if 'volume' in data and (data['volume'] == 0).sum() > len(data) * 0.1: 
        issues.append("⚠️ 出来高0の日が10%以上あります")
    price_changes = data['close'].pct_change().abs()
    if extreme_gaps := (price_changes > 0.2).sum(): 
        issues.append(f"⚠️ {extreme_gaps}日に20%以上の価格ギャップがあります")
    return issues

def safe_calculate_signal_strength(row):
    try:
        if pd.isna(row['sma_long']) or row['sma_long'] == 0: return 50
        strength = 50
        sma_divergence = (row['sma_short'] - row['sma_long']) / row['sma_long'] * 100
        strength += min(max(sma_divergence * 2, -25), 25)
        if not pd.isna(row['rsi']):
            if row['rsi'] < 30: strength += (30 - row['rsi']) * 0.5
            elif row['rsi'] > 70: strength -= (row['rsi'] - 70) * 0.5
        return max(0, min(100, strength))
    except Exception as e:
        logger.warning(f"Signal strength calculation error: {e}")
        return 50
