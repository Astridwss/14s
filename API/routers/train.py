"""
算法路由 —— RL 训练、IL 训练、推演评估
"""
from fastapi import APIRouter, BackgroundTasks

from schema.request import RLTrainRequest, ILTrainRequest, EvalRequest
from API.handlers import TrainHandler

router = APIRouter(tags=["训练与推演"])


@router.post("/train/rl")
async def start_rl_training(request: RLTrainRequest, background_tasks: BackgroundTasks):
    return TrainHandler(request, mode="train").run_rl(background_tasks)


@router.post("/train/il")
def start_il_training(request: ILTrainRequest, background_tasks: BackgroundTasks):
    return TrainHandler(request, mode="train").run_il(background_tasks)


@router.post("/eval/start")
def start_eval_task(request: EvalRequest):
    return TrainHandler(request, mode="eval").run_eval()
