import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

app = FastAPI(title="本地测试伪装平台")

# =====================================================================
# 🌟 新增：伪装文件服务器，提供 scene.json 下载
# =====================================================================
# =====================================================================
# 🌟 新增：伪装文件服务器，提供 scene.json 或 scene.xml 下载
# =====================================================================
@app.get("/scene.json")
@app.get("/scene.xml")  # 💡 增加一个路由，无论算法请求 json 还是 xml 都能命中
async def download_scene():
    """
    提供场景文件下载。
    请确保你的 JSON 预案文件（建议改名为 scene.json）与本 mock_server.py 放在同一个目录下
    """
    # 由 generate_mock_scene.py 生成，位于仓库根目录（本文件的上一级目录）
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mock_scene_100r_25s.json",
    )
    if os.path.exists(file_path):
        print(f"\n[伪装平台] 收到下载请求，正在下发文件: {file_path}")
        # 💡 修复 media_type 为 application/json
        return FileResponse(file_path, media_type="application/json", filename="scene.json")
    else:
        print(f"\n[伪装平台] 下载失败：找不到 {file_path}")
        # 注意：不要 return 字典，抛出标准的 HTTP 404 异常，这样 requests 库才能正确捕捉
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="scene.json not found")

# =====================================================================
# 原有逻辑：接收推送的回调
# =====================================================================
# 1. 伪装：接收训练指标推送
@app.post("/prod-api/agent/train/record")
async def receive_metrics(request: Request):
    data = await request.json()
    print(f"\n🟢 [伪装平台] 收到打点推送 (Metrics):")
    print(data)
    return {"code": 200, "msg": "Metrics Received"}

# 2. 伪装：接收模型路径推送
@app.post("/prod-api/agent/model/record")
async def receive_weights(request: Request):
    data = await request.json()
    print(f"\n🔵 [伪装平台] 收到模型路径 (Weights):")
    print(data)
    return {"code": 200, "msg": "Weights Received"}

# 3. 伪装：接收推演战报推送
@app.post("/api/receive/eval_result")
async def receive_eval_result(request: Request):
    data = await request.json()
    print(f"\n🟣 [伪装平台] 收到推演战报 (Eval Result):")
    print(data)
    return {"code": 200, "msg": "Eval Result Received"}

# 3.1 伪装：接收基线对比战报推送（专家预案基线，用于 20% 提升指标）
@app.post("/api/receive/compare_eval_result")
async def receive_baseline_eval_result(request: Request):
    data = await request.json()
    print(f"\n🟠 [伪装平台] 收到基线战报 (Baseline Eval Result):")
    print(data)
    return {"code": 200, "msg": "Baseline Eval Result Received"}

# 4. 伪装：接收训练报错
@app.post("/prod-api/agent/train/status")
async def receive_train_status(request: Request):
    data = await request.json()
    print(f"\n🟢 [伪装平台] 收到训练异常推送:")
    print(data)
    return {"code": 200, "msg": "Metrics Received"}


@app.post("/prod-api/agent/train/status")
async def receive_train_status(request: Request):
    """
    接收来自训练脚本的报错/状态更新
    """
    data = await request.json()
    import time
    # 获取当前时间
    receive_time = time.strftime("%H:%M:%S", time.localtime())
    
    print(f"\n" + "="*50)
    print(f"🟢 [伪装平台] 收到任务消息 ({receive_time})")
    print(f"任务 ID: {data.get('task_id')}")
    print(f"当前状态: {data.get('status')}")
    print(f"报错详情: {data.get('error_msg')}")
    print(f"原始数据: {data}")
    print("="*50)
    
    return {
        "code": 200, 
        "msg": "Platform received status successfully",
        "timestamp": int(time.time())
    }

if __name__ == "__main__":
    # 运行在 8080 端口，避免和你的主 API (8000端口) 冲突
    uvicorn.run(app, host="127.0.0.1", port=8080)