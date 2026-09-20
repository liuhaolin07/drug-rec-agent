#!/usr/bin/env bash
# 无头浏览器冒烟测试: 校验网页版页面在真实浏览器里的 JS 渲染是否正常, 并生成截图。
#
# 用法:  bash scripts/check_webapp_headless.sh [端口,默认8000]
# 要求:  先启动网页版服务(webapp/server.py), 且本机装有 Chrome 或 Edge。
set -u
PORT="${1:-8000}"
CH=""
for p in "C:/Program Files/Google/Chrome/Application/chrome.exe" \
         "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
         "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
         "/usr/bin/google-chrome" "/usr/bin/chromium" "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; do
  if [ -f "$p" ]; then CH="$p"; break; fi
done
if [ -z "$CH" ]; then echo "未找到 Chrome/Edge"; exit 1; fi
echo "browser: $CH"

if [ -n "${LOCALAPPDATA:-}" ]; then  # Windows(MSYS): 给原生浏览器传 Windows 可读路径
  TMP="${LOCALAPPDATA}/Temp/drugrec_web_check"
  mkdir -p "$TMP"
else
  TMP="$(mktemp -d)"
fi
URL="http://127.0.0.1:${PORT}/?demo=1"
OUT="$TMP/dom.html"

"$CH" --headless=new --disable-gpu --no-first-run --user-data-dir="$TMP/profile" \
      --virtual-time-budget=15000 --dump-dom "$URL" > "$OUT" 2> "$TMP/chrome.err"

c() { grep -o "$1" "$OUT" | wc -l; }
echo "下拉选项数(期望>=68): $(c 'option value=')"
echo "快捷标签数(期望>=32): $(c 'class=\"chip\"')"
echo "渲染出的分数单元格(期望>=5): $(grep -oE '>[0-9]\.[0-9][0-9][0-9]<' "$OUT" | wc -l)"
echo "渲染出的Major徽章(期望>=1): $(c 'badge b-major')"
echo "推荐清单出现次数(>1=已渲染): $(c '推荐清单')"

"$CH" --headless=new --disable-gpu --no-first-run --user-data-dir="$TMP/profile" \
      --window-size=1500,2100 --hide-scrollbars --virtual-time-budget=15000 \
      --screenshot="$TMP/webui.png" "$URL" 2>> "$TMP/chrome.err"
echo "截图: $TMP/webui.png"
cp "$TMP/webui.png" "$(dirname "$0")/../docs/webui.png" 2>/dev/null && echo "已更新 docs/webui.png"
