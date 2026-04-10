#!/bin/bash
# macOS launchd に毎日定時実行を登録するセットアップスクリプト
# 実行: bash setup_schedule.sh
# 削除: bash setup_schedule.sh --uninstall

set -e

LABEL="com.stockanalysis.daily"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# venv があればそちらを優先、なければ system python3 を使用
if [[ -f "${SCRIPT_DIR}/venv/bin/python3" ]]; then
  PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python3"
else
  PYTHON_BIN="$(which python3)"
fi
LOG_DIR="${SCRIPT_DIR}/results"

# 実行時刻: 平日・休日問わず 毎日 9:00 (米国市場の前日終値が確定した後)
RUN_HOUR=9
RUN_MINUTE=0

# ---------------------------------------------------------------------------
# アンインストール
# ---------------------------------------------------------------------------
if [[ "$1" == "--uninstall" ]]; then
  if launchctl list | grep -q "$LABEL"; then
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    echo "✅ launchd ジョブを停止しました"
  fi
  rm -f "$PLIST_PATH"
  echo "✅ plist を削除しました: $PLIST_PATH"
  exit 0
fi

# ---------------------------------------------------------------------------
# インストール
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON_BIN}</string>
    <string>${SCRIPT_DIR}/batch_runner.py</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${SCRIPT_DIR}</string>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${RUN_HOUR}</integer>
    <key>Minute</key>
    <integer>${RUN_MINUTE}</integer>
  </dict>

  <!-- PC がスリープ中だった場合、起動後に即実行する -->
  <key>RunAtLoad</key>
  <false/>

  <key>StandardOutPath</key>
  <string>${LOG_DIR}/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/launchd_stderr.log</string>

  <!-- クラッシュしても再起動しない (毎日定時に起動するため不要) -->
  <key>KeepAlive</key>
  <false/>
</dict>
</plist>
EOF

# 既存ジョブを一旦アンロード
launchctl unload "$PLIST_PATH" 2>/dev/null || true

# 登録
launchctl load "$PLIST_PATH"

echo ""
echo "✅ 登録完了"
echo "   スクリプト : ${SCRIPT_DIR}/batch_runner.py"
echo "   Python     : ${PYTHON_BIN}"
echo "   実行時刻   : 毎日 ${RUN_HOUR}:$(printf '%02d' $RUN_MINUTE)"
echo "   plist      : ${PLIST_PATH}"
echo "   ログ       : ${LOG_DIR}/batch.log"
echo ""
echo "今すぐ手動実行してテストする場合:"
echo "   python batch_runner.py --dry-run"
echo ""
echo "登録解除する場合:"
echo "   bash setup_schedule.sh --uninstall"
