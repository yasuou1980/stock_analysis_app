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

### シグナル実績トラッキングと精度改善
- 日次シグナルを `results/signals_history.csv` に蓄積し、N 営業日後のフォワードリターンで
  勝率を自動計測（`results/performance_report.txt`）。ロジック版別の集計で改善効果を検証できる
- 実測データの検証で特定した負けパターンを商品クラス別ゲートで遮断
  （インバース型レバETFの BUY 禁止・急落直後の SELL 禁止など）。
  逆に「拾えていなかった底」は押し目リバウンド BUY（20日高値から-8%超の押し目を
  長期MA上で反発した日に買う）で捕捉。
  診断と根拠の詳細は [docs/signal_accuracy_2026-07.md](docs/signal_accuracy_2026-07.md) を参照

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

[ticker_classes]
# 商品クラス定義 (シグナルゲートで使用。未記載のティッカーは plain = 現物扱い)
inverse_lev = ["SOXS", "SQQQ", "SPXS"]      # BUY シグナル禁止 (構造的減価)
long_lev = ["SOXL", "TQQQ", "UPRO", ...]    # SELL シグナル制限 (反発リスク)

[batch]
# バッチ実行で分析する銘柄リスト
tickers = [...]
# 分析に使う過去データの日数
lookback_days = 365
```

銘柄を追加するときは、レバレッジ型 ETF であれば `[ticker_classes]` にも分類を追記してください。

## 📁 ファイル構成

```
stock_analysis_app/
├── app.py                  # Streamlit アプリ エントリポイント
├── backtester.py           # テクニカル指標計算・シグナルゲート・バックテスト
├── batch_runner.py         # Streamlit 非依存のバッチ実行スクリプト
├── signal_tracker.py       # シグナル実績トラッキング・勝率レポート生成
├── data_loader.py          # yfinance データ取得
├── optimizer_ui.py         # パラメータ最適化 UI
├── plotting.py             # Plotly チャート描画
├── ui_components.py        # サイドバー UI コンポーネント
├── utils.py                # バリデーション・設定管理ユーティリティ
├── config.toml             # アプリ・バッチ設定・商品クラス定義
├── setup_schedule.sh       # macOS launchd 登録スクリプト
├── requirements.txt        # 依存パッケージ
├── tests/                  # 回帰テスト (python -m pytest tests/)
├── docs/
│   └── signal_accuracy_2026-07.md  # シグナル精度の診断・改善記録
├── scriptable/
│   └── StockSignal.js      # iOS Scriptable 用シグナル確認スクリプト
├── results/
│   ├── signals_YYYY-MM-DD.txt   # 日次シグナルレポート（git 管理）
│   ├── signals_history.csv      # シグナル履歴（実績計測用・git 管理）
│   ├── performance_report.txt   # 勝率レポート（自動生成・git 管理）
│   └── signals.db               # SQLite 蓄積データ（ローカルのみ）
└── .github/workflows/
    └── daily_run.yml       # GitHub Actions 定時実行ワークフロー
```

## 🧪 テスト

```bash
python -m pytest tests/
```

シグナルゲートの単体テストと 3 戦略（トレンドフォロー / 逆張り / レジーム切替）の
統合テストが含まれます。シグナルロジックを変更する際は必ず実行してください。

## 📱 Scriptable (iOS) 対応

`scriptable/StockSignal.js` に、日次シグナル判定ロジック（トレンドフォロー・逆張り）を JavaScript に移植したスクリプトを用意しています。Streamlit のフル機能（バックテスト・パラメータ最適化・チャート）は含みませんが、iPhone 単体で好きな銘柄のシグナルをその場で確認できます。

- Yahoo Finance の chart API から直接データ取得（`yfinance` 相当、pandas 非依存）
- `backtester.py` の `calculate_indicators_and_signals`（トレンドフォロー / 逆張り）と同じパラメータ・ロジックで最新シグナルを計算
- ティッカーは実行のたびにダイアログで入力可能（前回入力値を記憶するので気軽に差し替えて確認できる）
- ホーム画面ウィジェットとしても利用可能（ウィジェットパラメータにティッカーを指定）

**セットアップ:**
1. [Scriptable](https://apps.apple.com/app/scriptable/id1405459188) アプリを iPhone にインストール
2. `scriptable/StockSignal.js` の中身をコピーし、Scriptable 内で新規スクリプトとして貼り付け・保存
3. スクリプトを実行するとティッカー入力ダイアログが表示される

## ⚠️ 免責事項

このツールは教育および情報提供のみを目的としています。提供される情報や分析結果は投資助言を構成するものではありません。実際の投資判断はご自身の責任と判断において行ってください。
