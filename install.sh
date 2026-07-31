#!/usr/bin/env bash
set -Eeuo pipefail

APP_NAME="lite-ai-chat"
SERVICE_NAME="lite-ai-chat.service"
SEARCH_CONTAINER="lite-ai-search"
DEFAULT_INSTALL_DIR="/opt/lite-ai-chat"
DEFAULT_APP_PORT="8000"
SEARCH_PORT="8888"
SEARXNG_IMAGE="searxng/searxng@sha256:5d6d903ab82afa56ee32792d477f36bc63d3e5ca04fcb6947e28a5cfd987fad3"

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"
APP_PORT="${APP_PORT:-$DEFAULT_APP_PORT}"
TLS_HOST="${TLS_HOST:-}"
GROQ_API_KEY="${GROQ_API_KEY:-}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

log() {
  printf '[%s] %s\n' "$APP_NAME" "$*"
}

fail() {
  printf '[%s] 错误：%s\n' "$APP_NAME" "$*" >&2
  exit 1
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    fail "请使用 sudo bash install.sh，或以 root 身份运行。"
  fi
}

validate_inputs() {
  [[ "$INSTALL_DIR" == /* ]] || fail "INSTALL_DIR 必须是绝对路径。"
  [[ "$INSTALL_DIR" != "/" && "$INSTALL_DIR" != "/opt" ]] || fail "拒绝使用过宽的安装目录：$INSTALL_DIR"
  [[ "$INSTALL_DIR" != *$'\n'* && "$INSTALL_DIR" != *" "* ]] || fail "安装目录不能包含空格或换行。"
  [[ "$APP_PORT" =~ ^[0-9]+$ ]] || fail "APP_PORT 必须是数字。"
  (( APP_PORT >= 1 && APP_PORT <= 65535 )) || fail "APP_PORT 必须在 1-65535 之间。"
  if [[ -n "$TLS_HOST" ]]; then
    [[ "$TLS_HOST" =~ ^[A-Za-z0-9.-]+$ ]] ||
      fail "TLS_HOST 只能是 IP 地址或普通域名。"
  fi
  [[ -f "$SOURCE_DIR/app/main.py" ]] || fail "安装包不完整：缺少 app/main.py"
  [[ -f "$SOURCE_DIR/deploy/searxng/settings.yml" ]] || fail "安装包不完整：缺少 SearXNG 配置。"
}

install_dependencies() {
  local packages=(ca-certificates curl openssl python3 python3-pip python3-venv)
  local missing=()
  local package

  command -v apt-get >/dev/null 2>&1 || fail "当前脚本支持 Debian/Ubuntu（需要 apt-get）。"
  for package in "${packages[@]}"; do
    dpkg -s "$package" >/dev/null 2>&1 || missing+=("$package")
  done
  if ! command -v docker >/dev/null 2>&1; then
    missing+=(docker.io)
  fi
  if ((${#missing[@]})); then
    log "安装系统依赖：${missing[*]}"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "${missing[@]}"
  fi

  systemctl enable --now docker >/dev/null
}

read_secret_if_needed() {
  if [[ -f "$INSTALL_DIR/.env" && -z "$GROQ_API_KEY" && -z "$DEEPSEEK_API_KEY" ]]; then
    log "检测到现有 .env，将保留原模型密钥和配置。"
    return
  fi

  if [[ -t 0 ]]; then
    if [[ -z "$GROQ_API_KEY" ]]; then
      read -r -s -p "Groq API Key（可留空）: " GROQ_API_KEY
      printf '\n'
    fi
    if [[ -z "$DEEPSEEK_API_KEY" ]]; then
      read -r -s -p "DeepSeek API Key（可留空）: " DEEPSEEK_API_KEY
      printf '\n'
    fi
  fi

  if [[ -z "$GROQ_API_KEY" && -z "$DEEPSEEK_API_KEY" ]]; then
    log "警告：未提供模型 API Key。安装会继续，但需编辑 $INSTALL_DIR/.env 后才能聊天。"
  fi
}

install_application_files() {
  log "安装应用文件到 $INSTALL_DIR"
  install -d -m 0755 "$INSTALL_DIR/app" "$INSTALL_DIR/static" "$INSTALL_DIR/deploy/searxng"
  install -d -m 0700 "$INSTALL_DIR/data"
  if [[ "$(realpath "$SOURCE_DIR")" != "$(realpath -m "$INSTALL_DIR")" ]]; then
    cp -a "$SOURCE_DIR/app/." "$INSTALL_DIR/app/"
    cp -a "$SOURCE_DIR/static/." "$INSTALL_DIR/static/"
    cp -a "$SOURCE_DIR/deploy/searxng/." "$INSTALL_DIR/deploy/searxng/"
    install -m 0644 "$SOURCE_DIR/requirements.txt" "$INSTALL_DIR/requirements.txt"
  fi

  python3 -m venv "$INSTALL_DIR/.venv"
  "$INSTALL_DIR/.venv/bin/python" -m pip install --disable-pip-version-check --upgrade pip
  "$INSTALL_DIR/.venv/bin/pip" install --disable-pip-version-check -r "$INSTALL_DIR/requirements.txt"
}

generate_tls_certificate() {
  local tls_dir="$INSTALL_DIR/data/tls"
  local tls_host="$TLS_HOST"
  local san_type="DNS"
  local check_option="-checkhost"

  if [[ -z "$tls_host" ]]; then
    tls_host="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  tls_host="${tls_host:-127.0.0.1}"
  [[ "$tls_host" =~ ^[A-Za-z0-9.-]+$ ]] ||
    fail "无法安全地生成 TLS 证书：主机名格式无效。"
  if [[ "$tls_host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    san_type="IP"
    check_option="-checkip"
  fi

  install -d -m 0700 "$tls_dir"
  if [[ ! -s "$tls_dir/ca.crt" || ! -s "$tls_dir/ca.key" ]]; then
    log "生成 Lite AI Chat 本机 CA。"
    openssl req -x509 -new -nodes -newkey rsa:2048 -sha256 -days 3650 \
      -keyout "$tls_dir/ca.key" \
      -out "$tls_dir/ca.crt" \
      -subj "/CN=Lite AI Chat Local CA" \
      -extensions v3_ca >/dev/null 2>&1
  fi

  if [[
    ! -s "$tls_dir/server.crt" ||
    ! -s "$tls_dir/server.key"
  ]] || ! openssl x509 -checkend 2592000 -noout \
    -in "$tls_dir/server.crt" >/dev/null 2>&1 ||
    ! openssl x509 "$check_option" "$tls_host" -noout \
      -in "$tls_dir/server.crt" >/dev/null 2>&1; then
    log "为 $tls_host 生成 HTTPS 服务器证书。"
    openssl req -new -nodes -newkey rsa:2048 -sha256 \
      -keyout "$tls_dir/server.key" \
      -out "$tls_dir/server.csr" \
      -subj "/CN=$tls_host" >/dev/null 2>&1
    {
      printf 'subjectKeyIdentifier=hash\n'
      printf 'authorityKeyIdentifier=keyid,issuer\n'
      printf 'basicConstraints=critical,CA:FALSE\n'
      printf 'keyUsage=critical,digitalSignature,keyEncipherment\n'
      printf 'extendedKeyUsage=serverAuth\n'
      printf 'subjectAltName=%s:%s,IP:127.0.0.1,DNS:localhost\n' "$san_type" "$tls_host"
    } >"$tls_dir/server.ext"
    openssl x509 -req \
      -in "$tls_dir/server.csr" \
      -CA "$tls_dir/ca.crt" \
      -CAkey "$tls_dir/ca.key" \
      -CAcreateserial \
      -out "$tls_dir/server.crt" \
      -days 825 \
      -sha256 \
      -extfile "$tls_dir/server.ext" >/dev/null 2>&1
    rm -f "$tls_dir/server.csr" "$tls_dir/server.ext" "$tls_dir/ca.srl"
  else
    log "保留现有 HTTPS 服务器证书。"
  fi
  chmod 0600 "$tls_dir/ca.key" "$tls_dir/server.key"
  chmod 0644 "$tls_dir/ca.crt" "$tls_dir/server.crt"
}

set_env_value() {
  local file="$1"
  local key="$2"
  local value="$3"
  if grep -q "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >>"$file"
  fi
}

write_environment() {
  if [[ -f "$INSTALL_DIR/.env" && -z "$GROQ_API_KEY" && -z "$DEEPSEEK_API_KEY" ]]; then
    set_env_value "$INSTALL_DIR/.env" MAX_TOOL_ROUNDS 10
    set_env_value "$INSTALL_DIR/.env" MAX_SEARCH_RESULTS 10
    set_env_value "$INSTALL_DIR/.env" LLM_TIMEOUT 1200
    set_env_value "$INSTALL_DIR/.env" TLS_CERT_FILE "$INSTALL_DIR/data/tls/server.crt"
    set_env_value "$INSTALL_DIR/.env" TLS_KEY_FILE "$INSTALL_DIR/data/tls/server.key"
    chmod 0600 "$INSTALL_DIR/.env"
    return
  fi

  local jwt_secret
  jwt_secret="$(openssl rand -hex 32)"
  umask 077
  {
    printf 'OPENAI_API_BASE=https://api.groq.com/openai/v1\n'
    printf 'OPENAI_API_KEY=%s\n' "$GROQ_API_KEY"
    printf 'MODEL_NAME=llama-3.3-70b-versatile\n'
    printf 'DEEPSEEK_API_BASE=https://api.deepseek.com\n'
    printf 'DEEPSEEK_API_KEY=%s\n' "$DEEPSEEK_API_KEY"
    printf 'SEARXNG_URL=http://127.0.0.1:%s\n' "$SEARCH_PORT"
    printf 'SCRAPER_URL=http://127.0.0.1:3002\n'
    printf 'MAX_TOOL_ROUNDS=10\n'
    printf 'MAX_SEARCH_RESULTS=10\n'
    printf 'HTTP_TIMEOUT=8\n'
    printf 'LLM_TIMEOUT=1200\n'
    printf 'JWT_EXPIRE_DAYS=60\n'
    printf 'JWT_SECRET=%s\n' "$jwt_secret"
    printf 'HOST=0.0.0.0\n'
    printf 'PORT=%s\n' "$APP_PORT"
    printf 'DATA_DIR=%s/data\n' "$INSTALL_DIR"
    printf 'TLS_CERT_FILE=%s/data/tls/server.crt\n' "$INSTALL_DIR"
    printf 'TLS_KEY_FILE=%s/data/tls/server.key\n' "$INSTALL_DIR"
  } >"$INSTALL_DIR/.env"
}

install_search_service() {
  local search_secret
  search_secret="$(openssl rand -hex 32)"
  sed -i "s/CHANGE_ME_AT_INSTALL/$search_secret/" "$INSTALL_DIR/deploy/searxng/settings.yml"
  chmod 0600 "$INSTALL_DIR/deploy/searxng/settings.yml"

  if docker container inspect "$SEARCH_CONTAINER" >/dev/null 2>&1; then
    log "替换现有外部搜索容器。"
    docker rm -f "$SEARCH_CONTAINER" >/dev/null
  fi

  log "部署独立 SearXNG 搜索服务。"
  docker pull "$SEARXNG_IMAGE"
  docker run -d \
    --name "$SEARCH_CONTAINER" \
    --restart unless-stopped \
    --network host \
    --memory 160m \
    --memory-swap 512m \
    --cpus 0.75 \
    -e GRANIAN_HOST=127.0.0.1 \
    -e GRANIAN_PORT="$SEARCH_PORT" \
    -e GRANIAN_BLOCKING_THREADS=2 \
    -v "$INSTALL_DIR/deploy/searxng/settings.yml:/etc/searxng/settings.yml:ro" \
    "$SEARXNG_IMAGE" >/dev/null
}

install_systemd_service() {
  log "创建 systemd 服务。"
  cat >"/etc/systemd/system/$SERVICE_NAME" <<EOF
[Unit]
Description=Lite AI Chat
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/.venv/bin/python -m app.main
Restart=on-failure
RestartSec=3
MemoryMax=150M

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable "$SERVICE_NAME" >/dev/null
  systemctl restart "$SERVICE_NAME"
}

verify_installation() {
  local attempt
  for attempt in {1..20}; do
    if curl -kfsS --max-time 2 "https://127.0.0.1:$APP_PORT/health" >/dev/null; then
      break
    fi
    sleep 1
  done
  curl -kfsS --max-time 5 "https://127.0.0.1:$APP_PORT/health" >/dev/null ||
    fail "应用健康检查失败。请运行：journalctl -u $SERVICE_NAME -n 100"

  # SearXNG 首次启动会更新 CA 证书并初始化引擎，通常比应用更慢。
  for attempt in {1..30}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$SEARCH_PORT/" >/dev/null; then
      break
    fi
    sleep 1
  done
  curl -fsS --max-time 5 "http://127.0.0.1:$SEARCH_PORT/" >/dev/null ||
    fail "外部搜索服务未就绪。请运行：docker logs $SEARCH_CONTAINER"
  curl -fsS --max-time 20 \
    --get "http://127.0.0.1:$SEARCH_PORT/search" \
    --data-urlencode "q=OpenAI" \
    --data "format=json" >/dev/null ||
    fail "外部搜索服务健康检查失败。请运行：docker logs $SEARCH_CONTAINER"
}

main() {
  require_root
  validate_inputs
  install_dependencies
  read_secret_if_needed
  install_application_files
  generate_tls_certificate
  write_environment
  install_search_service
  install_systemd_service
  verify_installation

  local server_ip
  server_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  log "安装完成。"
  printf '访问地址：https://%s:%s\n' "${TLS_HOST:-${server_ip:-服务器IP}}" "$APP_PORT"
  printf '首次访问会出现自签证书警告；可将 %s/data/tls/ca.crt 安装为受信任 CA。\n' "$INSTALL_DIR"
  printf '首次打开页面时创建管理员账号。\n'
}

main "$@"
