#!/usr/bin/env bash
set -Eeuo pipefail

APP_NAME="lite-ai-chat"
SERVICE_NAME="lite-ai-chat.service"
SEARCH_CONTAINER="lite-ai-search"
DEFAULT_INSTALL_DIR="/opt/lite-ai-chat"
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"
PURGE=false

if [[ "${1:-}" == "--purge" ]]; then
  PURGE=true
elif [[ -n "${1:-}" ]]; then
  printf '用法：sudo bash uninstall.sh [--purge]\n' >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$APP_NAME" "$*"
}

fail() {
  printf '[%s] 错误：%s\n' "$APP_NAME" "$*" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  fail "请使用 sudo bash uninstall.sh，或以 root 身份运行。"
fi
[[ "$INSTALL_DIR" == /* ]] || fail "INSTALL_DIR 必须是绝对路径。"
[[ "$INSTALL_DIR" != "/" && "$INSTALL_DIR" != "/opt" ]] || fail "拒绝删除过宽目录：$INSTALL_DIR"

if systemctl list-unit-files "$SERVICE_NAME" >/dev/null 2>&1; then
  systemctl disable --now "$SERVICE_NAME" >/dev/null 2>&1 || true
fi
if [[ -f "/etc/systemd/system/$SERVICE_NAME" ]]; then
  rm -f "/etc/systemd/system/$SERVICE_NAME"
  systemctl daemon-reload
fi

if command -v docker >/dev/null 2>&1 &&
  docker container inspect "$SEARCH_CONTAINER" >/dev/null 2>&1; then
  docker rm -f "$SEARCH_CONTAINER" >/dev/null
fi

if [[ -d "$INSTALL_DIR" ]]; then
  if [[ "$PURGE" == false ]]; then
    backup_dir="/var/backups/lite-ai-chat-$(date +%Y%m%d-%H%M%S)"
    install -d -m 0700 "$backup_dir"
    [[ -f "$INSTALL_DIR/.env" ]] && cp -a "$INSTALL_DIR/.env" "$backup_dir/"
    [[ -d "$INSTALL_DIR/data" ]] && cp -a "$INSTALL_DIR/data" "$backup_dir/"
    log "配置与用户数据已备份到 $backup_dir"
  fi
  rm -rf -- "$INSTALL_DIR"
fi

if [[ "$PURGE" == true ]]; then
  log "已彻底卸载，应用目录和数据均已删除。Docker 与系统 Python 未删除。"
else
  log "已卸载应用和搜索容器；配置与数据备份已保留。"
fi
