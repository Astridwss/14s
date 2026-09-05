#!/bin/bash

# Deactivate conda if active
conda deactivate 2>/dev/null

# Activate virtual environment
source ./.venv/Scripts/activate

# 奖励分项诊断：每 50 局打印一次分项表（0 = 关闭，训练零额外开销）
export REWARD_BREAKDOWN_INTERVAL=50

PID_MOCK_SERVER=""
PID_MOCK_ZMQ=""

cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    [ -n "$PID_MOCK_SERVER" ] && kill "$PID_MOCK_SERVER" 2>/dev/null
    [ -n "$PID_MOCK_ZMQ" ] && kill "$PID_MOCK_ZMQ" 2>/dev/null
    wait 2>/dev/null
}
# 无论正常退出还是 Ctrl+C / 收到 TERM，都清理后台进程
trap cleanup EXIT INT TERM

# Start mock_server.py in background
echo "Starting mock_server.py..."
python test/mock_server.py &
PID_MOCK_SERVER=$!

# Start mock_zmq.py in background
echo "Starting mock_zmq.py..."
python test/mock_zmq.py &
PID_MOCK_ZMQ=$!

# Start uvicorn in foreground (with --reload for hot-reload)
echo "Starting uvicorn..."
cd API && python -m uvicorn main:app 
