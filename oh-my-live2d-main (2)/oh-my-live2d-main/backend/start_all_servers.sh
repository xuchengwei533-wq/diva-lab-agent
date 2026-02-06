#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-"$(cd "$SCRIPT_DIR/.." && pwd)"}"
BT_ENV_NAME="${BT_ENV_NAME:-lesson_diva-ai_cn}"
LOG_DIR="${LOG_DIR:-/www/wwwlogs/${BT_ENV_NAME}}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"

load_env_kv_file() {
  local file="$1"
  [[ -f "$file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%$'\r'}"
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" =~ ^[[:space:]]*export[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
      continue
    fi

    if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
      continue
    fi
  done < "$file"
}

if [[ -f /www/server/panel/script/btpyprojectenv.sh ]]; then
  unset _BT_PROJECT_ENV || true
  if ! source /www/server/panel/script/btpyprojectenv.sh "$BT_ENV_NAME"; then
    echo "warning: btpyprojectenv activation failed for ${BT_ENV_NAME}; continue"
  fi
fi

ENV_FILE="/www/server/python_project/vhost/env/${BT_ENV_NAME}.env"
load_env_kv_file "$ENV_FILE"
load_env_kv_file "$BASE/backend/.env"

pids=()
names=()
start_bg() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  mkdir -p "$LOG_DIR"
  : > "$log_file"
  "$@" >>"$log_file" 2>&1 &
  local pid="$!"
  pids+=("$pid")
  names+=("$name")
  echo "started ${name} pid=${pid} log=${log_file}"
}
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python not found in PATH; bt env activation may have failed"
    exit 1
  fi
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "warning: DASHSCOPE_API_KEY is NOT set; llm/tts/asr may exit immediately"
else
  echo "DASHSCOPE_API_KEY is set"
fi

cd "$BASE"
echo "BASE=${BASE}"
echo "BT_ENV_NAME=${BT_ENV_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "logs: ${LOG_DIR}/web_8000.log ${LOG_DIR}/web_8010.log ${LOG_DIR}/web_8001.log ${LOG_DIR}/asr_legacy_8002.log ${LOG_DIR}/llm_8003.log ${LOG_DIR}/tts_8004.log ${LOG_DIR}/scoring_8005.log ${LOG_DIR}/asr_new_8006.log"
start_bg web_8000 env WEB_PORT=8000 WEB_MODE=page "$PYTHON_BIN" mao_demo_server.py
start_bg web_8010 env WEB_PORT=8010 WEB_MODE=assets "$PYTHON_BIN" mao_demo_server.py
start_bg web_8001 "$PYTHON_BIN" -m http.server 8001 --bind 127.0.0.1

cd "$BASE/backend"
start_bg asr_legacy_8002 "$PYTHON_BIN" main.py
start_bg llm_8003 "$PYTHON_BIN" qwen_chat_server.py
start_bg tts_8004 "$PYTHON_BIN" tts_ws_server.py
start_bg scoring_8005 "$PYTHON_BIN" asr_server.py
start_bg asr_new_8006 "$PYTHON_BIN" asr_new.py

echo "urls:"
echo "  - http://127.0.0.1:8000/mao_demo.html?live2dPort=8010"
echo "  - http://127.0.0.1:8000/chat_interface.html?live2dPort=8010"
echo "  - http://127.0.0.1:8001/chat_interface.html?live2dPort=8010"
echo "  - http://127.0.0.1:8010/"
wait
