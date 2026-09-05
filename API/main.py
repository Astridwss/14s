"""
LD 协同 MARL 训练平台 — API 入口

职责：创建 FastAPI 实例、挂载子路由、启动 uvicorn。
业务逻辑 → API/handlers/，路由定义 → API/routers/，数据校验 → schema/。
"""
import sys
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
os.environ["PROJECT_ROOT"] = str(_project_root)

from API.routers.train import router as train_router
from API.routers.control import router as control_router

app = FastAPI(title="LD协同 MARL 训练平台 API", version="1.0.0")


app.include_router(train_router, prefix="/api/v1")
app.include_router(control_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
