import sys
import json
from pathlib import Path
import traceback
from pydantic import BaseModel
import os
import uvicorn
from fastapi import APIRouter
#from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_project_root  = Path(__file__).resolve().parent.parent
os.environ["PROJECT_ROOT"] = str(_project_root)

from config import Config
from runner import TrainRunner, ILTrainRunner, EvalRunner
from notifier import RLTrainRequest, ILTrainRequest, EvalRequest, PlatformNotifier, TaskActionRequest
from fastapi import FastAPI, BackgroundTasks
from API.utils.downloader import SceneDownloader
from env import ScenarioAdapter
from sim.plan_file_process import PlanFileProcess

def run_rl_train_task(task_id: str, request: RLTrainRequest):
    """后台守护进程，执行强化学习训练逻辑。"""
    try:
        # 完成：参数融合 + 文件夹创建 + 场景下载 + 维度解析
        my_train_config = prepare_task_context(task_id, request, mode="train")
        
        runner = TrainRunner(conf=my_train_config)
        notifier = PlatformNotifier(task_id)

        runner.run(
            metrics_callback=notifier.push_metrics,  
            weight_callback=notifier.push_weights     
        )
        
        if request.hyperparameters.show_log:
            print(f"[Task {task_id}] 强化学习训练任务完成")
            
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[Task {task_id}] 强化学习训练崩溃: {error_detail}")


def run_il_train_task(task_id: str, request: ILTrainRequest):
    """后台守护进程，执行模仿学习训练逻辑。"""
    try:
        #  一键完成前置准备（保证了 IL 和 RL 获取维度的逻辑绝对同源）
        my_train_config = prepare_task_context(task_id, request, mode="train")
        
        runner = ILTrainRunner(conf=my_train_config)
        notifier = PlatformNotifier(task_id)

        runner.run(
            metrics_callback=notifier.push_metrics,   
            weight_callback=notifier.push_weights     
        )
        
        print(f"[ILTrainTask {task_id}] 模仿学习训练任务完成")
            
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[ILTrainTask {task_id}] 模仿学习训练崩溃: {error_detail}")


def run_eval_task(task_id: str, request: EvalRequest):
    """后台守护进程，执行推演逻辑。"""
    try:
        my_eval_config = prepare_task_context(task_id, request, mode="eval")
        my_eval_config.task_id = task_id  # 推演特有的参数
        
        runner = EvalRunner(conf=my_eval_config)
        #notifier = PlatformNotifier(task_id)

        #runner.run(result_callback=notifier.push_eval_result)
        file_path = runner.run()
        
        print(f"[EvalTask {task_id}] 推演完成")
        return file_path
        
    except Exception as e:
        error_detail = traceback.format_exc()
        print(f"[EvalTask {task_id}] 推演崩溃: {error_detail}")
        raise e


def generate_entities_from_scene(conf: Config, plan_id: int):
    """内部绘制态势求业务函数：负责读取文件、翻译坐标、生成态势所需的实体列表"""
    scene_path = conf.local_scene_path
    processor = PlanFileProcess()
    battle_scene = processor.read_battle_scene_from_json(plan_id=plan_id, file_path=scene_path)
    
    entities = []
    for rid, radar in battle_scene.dict_radar_id_info.items():
        entities.append({"id": rid, "type": "Radar", "name": f"雷达 {rid}", "lon": radar.longitude, "lat": radar.latitude, "alt": radar.altitude, "color": "BLUE"})
    for tid, target in battle_scene.dict_target_id_info.items():
        entities.append({"id": tid, "type": "Target", "name": f"目标 {tid}", "lon": target.longitude, "lat": target.latitude, "alt": target.altitude, "color": "RED"})
        
    return entities


def _get_flag_dir() -> str:
    """内部通用函数：获取并确保标志位文件夹存在"""
    flag_dir = os.path.join(os.environ.get("PROJECT_ROOT", "."), "temp_flags")
    os.makedirs(flag_dir, exist_ok=True)
    return flag_dir


def get_pause_flag_path(task_id: str) -> str:
    """统一管理暂停标志文件的路径"""
    return os.path.join(_get_flag_dir(), f"{task_id}_pause.flag")


def get_terminate_flag_path(task_id: str) -> str:
    """统一管理终止标志文件的路径"""
    return os.path.join(_get_flag_dir(), f"{task_id}_terminate.flag")

#============================================================================================================================
def prepare_task_context(task_id: str, request_data, mode: str) -> Config:
    """整合场景下载器和维度解析器，生成唯一的 Config 对象"""
    conf = Config(mode=mode)
    conf.update_from_api(request_data)
    
    #创建文件夹, 注入conf
    if mode == "train":
        conf.model_dir = os.path.join(conf.model_dir, task_id)
        os.makedirs(conf.model_dir, exist_ok=True)
        conf.result_dir = os.path.join(conf.result_dir, task_id)
        os.makedirs(conf.result_dir, exist_ok=True)

    if mode == "eval":
        conf.eval_records_dir = os.path.join(conf.eval_records_dir, task_id)
        os.makedirs(conf.eval_records_dir, exist_ok=True)
    
    #暂停和终止的路径都注入conf
    conf.pause_flag_file = get_pause_flag_path(task_id)
    conf.terminate_flag_file = get_terminate_flag_path(task_id)

    if hasattr(conf, 'scene_url') and conf.scene_url:
        #下载场景文件，并注入conf
        local_scene_path = SceneDownloader.download_file(conf.scene_url, task_id)
        conf.local_scene_path = local_scene_path
        
        #解析场景文件，算好动作空间和状态空间，注入 conf
        ScenarioAdapter.parse_and_inject(conf)
    else:
        conf.local_scene_path = os.path.join(
            os.environ.get("PROJECT_ROOT", "."), "scenarios", task_id, "scene.json"
        )
        
    return conf

#==============================================================API接口==============================================================
app = FastAPI(title="LD协同 MARL 训练平台 API", version="1.0.0")
# static_dir = os.path.join(os.environ.get("PROJECT_ROOT", "."), "static")
# app.mount("/static", StaticFiles(directory=static_dir), name="static")

# @app.get("/map")
# def get_map_page():
#     html_path = os.path.join(os.environ.get("PROJECT_ROOT", "."), "static", "index.html")
#     print(f"index_path is {html_path}")
#     with open(html_path, "r", encoding="utf-8") as f:
#         return HTMLResponse(content=f.read())


# @app.get("/api/v1/scene/init_data")
# def get_scene_init_data(task_id: str, plan_id: int = 867):
#     """
#     注意：FastAPI 会自动把 URL 中的 ?task_id=xxx 解析并赋值给这里的 task_id 变量
#     """
#     try:
#         # 你的总管家函数
#         my_eval_config = prepare_task_context(task_id=task_id, request_data=None, mode="eval")
#         # 你的业务函数
#         entities_data = generate_entities_from_scene(my_eval_config, plan_id)
        
#         return {"code": 200, "data": entities_data}
#     except Exception as e:
#         return {"code": 500, "message": f"场景解析失败: {str(e)}"}


@app.post("/api/v1/train/rl")
async def start_training(request: RLTrainRequest, background_tasks: BackgroundTasks):
    """
    接收平台 JSON 配置，解析参数，并下发异步训练任务。
    """
    print(f"\n==========[收到 RL 训练请求]========== \n{request.model_dump_json(indent=2, ensure_ascii=False)}\n========== ")
    # 1. 生成全局唯一任务 ID
    task_id = request.task_id
    
    # 2. 将耗时的训练逻辑丢给 FastAPI 的后台任务队列
    background_tasks.add_task(run_rl_train_task, task_id, request)
    
    return {
        "code": 200,
        "message": "训练任务已成功下发至后台",
        "data": {
            "task_id": task_id,
            "algorithm": request.algorithm,
            "status": "starting"
        }
    }


@app.post("/api/v1/train/il")
def start_il_training(request: ILTrainRequest, background_tasks: BackgroundTasks):
    """
    接收平台 JSON 配置，解析参数，并下发异步训练任务。
    """
    print(f"\n==========[收到 RL 训练请求]========== \n{request.model_dump_json(indent=2, ensure_ascii=False)}\n========== ")
    # 1. 生成全局唯一任务 ID
    task_id = request.task_id

    background_tasks.add_task(run_il_train_task, task_id, request)
    
    return {
    "code": 200,
    "message": "训练任务已成功下发至后台",
    "data": {
        "task_id": task_id,
        "algorithm": request.algorithm,
        "status": "starting"
    }
}


@app.post("/api/v1/eval/start")
def start_eval_task(request: EvalRequest):
    """
    平台调用此接口启动推演评估
    """        
    print(f"\n==========[收到推演评估请求]========== \n{request.model_dump_json(indent=2, ensure_ascii=False)}\n========== ")
    file_path = None 
    
    try:
        # 同步拉起推演，并阻塞等待它跑完返回结果
        file_path = run_eval_task(request.task_id, request) 
        
        return {
            "code": 200,
            "message": "推演任务已成功完成",
            "data": {
                "taskId": request.task_id,
                "algo": request.algorithm,
                "status": "finished",
                "timeSeriesFile": file_path,  
                "evalFile": file_path
            }
        }
    except Exception as e:
        return {
            "code": 500,
            "message": f"推演任务失败: {str(e)}",
            "data": {
                "taskId": request.task_id, # 保持驼峰命名一致性
                "status": "failed",
                "eval_records_file": file_path # 如果第一步就崩溃了，这里会返回 None
            }
        }


@app.post("/api/v1/train/pause")
def pause_training_task(request: TaskActionRequest):
    flag_path = get_pause_flag_path(request.task_id)
    try:
        with open(flag_path, 'w') as f:
            f.write("PAUSED")
        return {"code": 200, "message": f"已暂停任务 {request.task_id}"}
    except Exception as e:
        return {"code": 500, "message": f"暂停失败: {str(e)}"}


@app.post("/api/v1/train/resume")
def resume_training_task(request: TaskActionRequest):
    flag_path = get_pause_flag_path(request.task_id)
    if os.path.exists(flag_path):
        os.remove(flag_path)
        return {"code": 200, "message": f"已继续任务 {request.task_id}"}
    return {"code": 200, "message": "任务未处于暂停状态"}


@app.post("/api/v1/train/terminate")
def terminate_training_task(request: TaskActionRequest):
    terminate_flag_path = get_terminate_flag_path(request.task_id)
    pause_flag_path = get_pause_flag_path(request.task_id) # 获取该任务的暂停信箱地址
    
    try:
        with open(terminate_flag_path, 'w') as f:
            f.write("TERMINATE")
        if os.path.exists(pause_flag_path):
            os.remove(pause_flag_path)
            
        return {"code": 200, "message": f"已向任务 {request.task_id} 发送终止指令"}
    except Exception as e:
        return {"code": 500, "message": f"终止指令下发失败: {str(e)}"}


if __name__ == "__main__":
    # host="0.0.0.0" 表示允许局域网/外网访问
    # port=8000 是默认端口，你可以改成任何你需要的端口 (如 8080)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)