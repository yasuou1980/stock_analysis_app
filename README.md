# 🏆 プロフェッショナル株式戦略分析アプリケーション

Streamlit で構築したインタラクティブな株価分析ツールです。テクニカル指標の計算・バックテスト・パラメータ最適化に加え、毎日自動でシグナルを生成・蓄積するバッチ実行基盤を備えています。

## ✨ 主な機能

### スイングトレード用の画面 (Streamlit アプリ)
- **今日のスイング候補**: 監視 30 銘柄のうち最新の終値で BUY が出ている銘柄を、売買プラン付きで一覧表示
  - 損切り価格 (終値 − ATR×2)、株数 (損切り時の損失が資金の 1% になるよう逆算、1 銘柄 25% まで)、手仕舞い予定日 (10 営業日目)
  - 資金・許容損失・損切り幅・保有日数は画面上部で変更可。既定値は `config.toml` の `[swing]`
  - 戦略ごとの BUY 実績 (過去 4 年の検証とライブ) を同じ画面に表示
- **保有ポジション**: 買付日・買値・株数を登録すると、損切りライン到達 / 期間満了 / 保有継続 を判定
  - 保存先は `positions.csv` (git 管理外)。Streamlit Cloud では再起動で消えるためローカル実行向け
- **個別分析**の末尾に、その銘柄を売買プランどおりに売買した場合の過去成績 (毎日買った場合との比較つき)

売買プランは「翌営業日の寄り付きで買い、損切りに掛からなければ 10 営業日目の引けで売る」前提です。
利確目標とトレイリングストップは検証で成績を改善しなかったため置いていません。
**どの戦略も統計的に確かな優位性は確認できていません** (押し目候補が弱いプラス)。数字と根拠は
[docs/signal_accuracy_2026-07.md](docs/signal_accuracy_2026-07.md) の 2026-11-11 の節を参照してください。

### 個別分析・バックテスト (Streamlit アプリ)
- **株価データ取得**: `yfinance` で任意の銘柄・期間のデータを取得
- **売買戦略**
  - **押し目候補**: 終値が 200 日平均より上で RSI(10) < 35 (BUY 候補のみ)
  - **トレンドフォロー**: EMA クロス・MACD・ADX・RSI・ブレイクアウトを複合スコアリング
  - **逆張り**: ストキャスティクス反転・RSI・ボリンジャーバンド・乖離率を複合スコアリング
- **バックテスト**: ポジションサイジング（ボラティリティ調整 / 固定比率）・手数料・スリッページを考慮
- **パラメータ最適化**: グリッドサーチで最高シャープレシオ・最高リターンの設定値を探索
- **インタラクティブチャート**: ローソク足・指標・シグナル・資産推移を Plotly で可視化
- **設定の永続化**: `config.toml` へのパラメータ保存・読み込み

### 自動バッチ実行（毎日）
- **GitHub Actions** により毎日 JST 8:00 に自動実行
- 30 銘柄 × 3 戦略のシグナルを計算し `results/signals_YYYY-MM-DD.txt` として自動 commit
- レポート末尾に「スイング候補」(損切り価格・損切り幅・資金に対する建玉比率・手仕舞い予定日) を追記
- **ローカル Mac** でも `launchd` による定時実行に対応（`setup_schedule.sh`）
- SQLite（`results/signals.db`）への蓄積もサポート

### シグナル実績トラッキングと精度改善
- 日次シグナルを `results/signals_history.csv` に蓄積し、N 営業日後のフォワードリターンで
  勝率を自動計測（`results/performance_report.txt`）。ロジック版別の集計で改善効果を検証できる
- 実測データの検証で特定した負けパターンを商品クラス別ゲートで遮断
  （インバース型レバETFの BUY 禁止・トレンド SELL はインバース型のみ許可など）。
  一度は押し目リバウンド BUY も追加したが、2ヶ月のフォワード検証で
  予測 (10日勝率78.9%) を大きく下回った (実測20.0%) ため撤回した。
  ゲートが止めたシグナルは `raw_signal` としてシャドー計測し、ゲートの妥当性を
  フォワードで検証し続ける。診断と根拠、および撤回に至った検証の詳細は [docs/signal_accuracy_2026-07.md](docs/signal_accuracy_2026-07.md) を参照

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
├── swing.py                # スイング用の売買プラン・候補スキャン・保有ポジション判定
├── swing_ui.py             # スイング候補 / 保有ポジションの画面
├── swing_study.py          # 売買プランの過去検証 (python swing_study.py)
├── data_loader.py          # yfinance データ取得
├── optimizer_ui.py         # パラメータ最適化 UI
├── plotting.py             # Plotly チャート描画
├── ui_components.py        # サイドバー UI コンポーネント
├── utils.py                # バリデーション・設定管理ユーティリティ
├── config.toml             # アプリ・バッチ設定・商品クラス定義・スイング設定 ([swing])
├── positions.csv           # 保有ポジション (アプリが作成。git 管理外)
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
