#!/usr/bin/env bash
# 启动 API 服务：先杀掉占用 8066 端口的旧进程，再起 uvicorn。
# 跨平台：Linux 部署机走 conda + lsof；Windows 本地(Git Bash)走 .venv + netstat/taskkill。

# ---------- 环境激活（仅 Linux 部署机；Windows 本地靠已激活的 .venv） ----------
if [ -f /root/anaconda3/etc/profile.d/conda.sh ]; then
    source /root/anaconda3/etc/profile.d/conda.sh
    conda activate dev-new
fi

# ---------- 杀掉占用 8066 端口的旧进程 ----------
PORT=8066

if command -v lsof >/dev/null 2>&1; then
    # Linux / macOS
    PID=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "发现占用 ${PORT} 端口的进程: $PID，正在杀掉..."
        kill -9 $PID
    else
        echo "${PORT} 端口未被占用"
    fi
elif command -v netstat >/dev/null 2>&1 && command -v taskkill >/dev/null 2>&1; then
    # Windows (Git Bash)：netstat 找 LISTENING 的 PID → taskkill 强杀
    PID=$(netstat -ano | grep -i listening | grep ":${PORT} " | awk '{print $NF}' | head -n1)
    if [ -n "$PID" ]; then
        echo "发现占用 ${PORT} 端口的进程: $PID，正在杀掉..."
        taskkill //F //PID "$PID"
    else
        echo "${PORT} 端口未被占用"
    fi
else
    echo "警告：未找到 lsof / netstat+taskkill，跳过端口清理"
fi
sleep 1

# ---------- 启动 uvicorn ----------
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        # Windows 本地开发：切到项目根目录下的 API，绑定本机回环
        cd "$(dirname "$0")/API" || exit 1
        python -m uvicorn main:app --host 127.0.0.1 --port "$PORT"
        ;;
    *)
        # Linux 部署机
        cd /home/kylin/14229/back/code-0913/API || exit 1
        python -m uvicorn main:app --host 7.81.21.52 --port "$PORT"
        ;;
esac
