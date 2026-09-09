"""
平台控制路由 —— 暂停、恢复、终止训练任务

前缀统一为 /api/v1，由 main.py 挂载时指定。
"""
from fastapi import APIRouter

from schema.request import TaskActionRequest, SpeedRequest
from API.handlers.control_handler import ControlHandler

router = APIRouter(tags=["任务控制"])


@router.post("/train/pause")
def pause_training_task(request: TaskActionRequest):
    return ControlHandler(request.task_id).pause()


@router.post("/train/resume")
def resume_training_task(request: TaskActionRequest):
    return ControlHandler(request.task_id).resume()


@router.post("/train/terminate")
def terminate_training_task(request: TaskActionRequest):
    return ControlHandler(request.task_id).terminate()


@router.post("/train/speed")
def set_speed(request: SpeedRequest):
    return ControlHandler(request.task_id).set_speed(request.speed)
