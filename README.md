# 🏆 プロフェッショナル株式戦略分析アプリケーション

Streamlit で構築したインタラクティブな株価分析ツールです。テクニカル指標の計算・バックテスト・パラメータ最適化に加え、毎日自動でシグナルを生成・蓄積するバッチ実行基盤を備えています。

## ✨ 主な機能

### Streamlit アプリ
- **株価データ取得**: `yfinance` で任意の銘柄・期間のデータを取得
- **2種類の売買戦略**
  - **トレンドフォロー**: EMA クロス・MACD・ADX・RSI・ブレイクアウトを複合スコアリング
  - **逆張り**: ストキャスティクス反転・RSI・ボリンジャーバンド・乖離率を複合スコアリング
- **バックテスト**: ポジションサイジング（ボラティリティ調整 / 固定比率）・手数料・スリッページを考慮
- **パラメータ最適化**: グリッドサーチで最高シャープレシオ・最高リターンの設定値を探索
- **インタラクティブチャート**: ローソク足・指標・シグナル・資産推移を Plotly で可視化
- **設定の永続化**: `config.toml` へのパラメータ保存・読み込み

### 自動バッチ実行（毎日）
- **GitHub Actions** により毎日 JST 8:00 に自動実行
- 30 銘柄 × 2 戦略のシグナルを計算し `results/signals_YYYY-MM-DD.txt` として自動 commit
- **ローカル Mac** でも `launchd` による定時実行に対応（`setup_schedule.sh`）
- SQLite（`results/signals.db`）への蓄積もサポート

## 📊 監視銘柄（バッチ対象）

| カテゴリ | 銘柄 |
|----------|------|
| 半導体・ハイテク | NVDA, AMD, TSM, INTC, AVGO, QCOM, MU, AMAT |
| 主要テック | AAPL, MSFT, GOOGL, META, AMZN |
| レバレッジ ETF（半導体） | SOXL, SOXS |
| レバレッジ ETF（ナスダック・S&P） | TQQQ, SQQQ, UPRO, SPXS |
| テーマ型 ETF | FNGG, TECL, TSLL, NUGT |
| 個別グロース | SOFI, CLSK, MSTR, COIN |
| 主要インデックス ETF | SPY, QQQ, IWM |

## 🚀 セットアップ

**Python 3.10 以上が必要です**

```bash
git clone https://github.com/yasuou1980/stock_analysis_app.git
cd stock_analysis_app

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Streamlit アプリを起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開きます。

### バッチ実行

```bash
# 動作確認（DB・ファイル保存なし）
python batch_runner.py --dry-run

# 本番実行（SQLite + テキスト保存）
python batch_runner.py

# テキストのみ保存（DB なし・CI 環境向け）
python batch_runner.py --no-db
```

### macOS でのローカル定時実行

```bash
# 毎日 9:00 に自動実行するよう launchd に登録
bash setup_schedule.sh

# 登録解除
bash setup_schedule.sh --uninstall
```

## ⚙️ 設定ファイル（`config.toml`）

```toml
[tickers]
# Streamlit アプリのサイドバーに表示する銘柄
default_tickers = [...]

[batch]
# バッチ実行で分析する銘柄リスト
tickers = [...]
# 分析に使う過去データの日数
lookback_days = 365
```

## 📁 ファイル構成

```
stock_analysis_app/
├── app.py                  # Streamlit アプリ エントリポイント
├── backtester.py           # テクニカル指標計算・バックテスト・パフォーマンス計算
├── batch_runner.py         # Streamlit 非依存のバッチ実行スクリプト
├── data_loader.py          # yfinance データ取得
├── optimizer_ui.py         # パラメータ最適化 UI
├── plotting.py             # Plotly チャート描画
├── ui_components.py        # サイドバー UI コンポーネント
├── utils.py                # バリデーション・設定管理ユーティリティ
├── config.toml             # アプリ・バッチ設定
├── setup_schedule.sh       # macOS launchd 登録スクリプト
├── requirements.txt        # 依存パッケージ
├── results/
│   ├── signals_YYYY-MM-DD.txt  # 日次シグナルレポート（git 管理）
│   └── signals.db              # SQLite 蓄積データ（ローカルのみ）
└── .github/workflows/
    └── daily_run.yml       # GitHub Actions 定時実行ワークフロー
```

## ⚠️ 免責事項

このツールは教育および情報提供のみを目的としています。提供される情報や分析結果は投資助言を構成するものではありません。実際の投資判断はご自身の責任と判断において行ってください。
