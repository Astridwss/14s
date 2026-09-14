source /root/anaconda3/etc/profile.d/conda.sh
conda activate dev-new

# 按 8066 端口查进程，有就杀掉
PID=$(lsof -ti :8066)
if [ -n "$PID" ]; then
    echo "发现占用 8066 端口的进程: $PID，正在杀掉..."
    kill -9 $PID
    sleep 1
    echo "已杀掉"
else
    echo "8066 端口未被占用"
fi

cd /home/kylin/14229/back/code-0908/API || exit 1
python -m uvicorn main:app --host 7.81.21.52 --port 8066