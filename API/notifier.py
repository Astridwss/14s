import sys
import requests
import threading
import os
from pathlib import Path

from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_project_root  = Path(__file__).resolve().parent.parent
from config import Config

os.environ["PROJECT_ROOT"] = str(_project_root)


class PlatformNotifier:
    """
    统一的平台回调通信组件
    将所有与 Java/Go 后端的 HTTP 通信逻辑集中于此。
    """
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.conf = Config()

        self.PLATFORM_BASE_URL = self.conf.PLATFORM_BASE_URL
        self.PLATFORM_METRICS_URL = f"{self.PLATFORM_BASE_URL}/prod-api/agent/train/record"
        self.PLATFORM_WEIGHTS_URL = f"{self.PLATFORM_BASE_URL}/prod-api/agent/model/record"
        self.PLATFORM_EVAL_RESULT_URL = f"{self.PLATFORM_BASE_URL}/api/receive/eval_result"
        #self.PLATFORM_STATUS_URL = f"{self.PLATFORM_BASE_URL}/prod-api/agent/train/status"

    def _async_post(self, url: str, payload: dict, error_prefix: str = ""):
        """底层的通用异步发送方法"""
        def _send():
            try:
                # 可以在这里统一加 Header、鉴权 Token、重试逻辑等
                requests.post(url, json=payload, timeout=5)
            except Exception as e:
                if error_prefix:
                    print(f" [Task {self.task_id}] {error_prefix}: {e}")
        
        threading.Thread(target=_send).start()

    def push_metrics(self, metrics_data: dict):
        """推送训练指标 (RL / IL 共用)"""
        metrics_data["taskId"] = self.task_id
        self._async_post(self.PLATFORM_METRICS_URL, metrics_data)

    def push_weights(self, weight_data: dict):
        payload = {
            "taskId": self.task_id, 
            "planId": weight_data.get('planId'),
            "episode": weight_data.get('episode'),
            "absModelPath": weight_data.get('model_dir'),
            "modelType": 1 if weight_data.get('algo') == "qmix" else 0,
        }
        
        # 如果 key 不存在，这段代码跳过
        if "absCsvPath" in weight_data:
            payload["absCsvPath"] = weight_data["absCsvPath"]
            
        self._async_post(self.PLATFORM_WEIGHTS_URL, payload, "模型路径上传失败")

    def push_error(self, error_msg: str):
        """推送崩溃信息到平台"""
        payload = {
            "task_id": self.task_id,
            "status": "failed",
            "error_msg": error_msg
        }
        try:
            # 这里的 timeout 很重要，防止平台挂了导致训练线程也卡住
            response = requests.post(self.PLATFORM_STATUS_URL, json=payload, timeout=5)
            print(f"发送报错至平台成功: {response.status_code}")
        except Exception as e:
            print(f"无法联系平台发送错误信息: {e}")

    def push_eval_result(self, result_data: dict):
        """推送评估/推演结果"""
        result_data["taskId"] = self.task_id
        self._async_post(self.PLATFORM_EVAL_RESULT_URL, result_data, "采集数据推送失败")



# 1. 定义数据校验模型 (映射 JSON 要素)
class RLHyperparameters(BaseModel):
    # 1. 基础环境与日志控制
    show_log: Optional[bool] = Field(default=True, description="是否在控制台打印日志")
    seed: Optional[int] = Field(default=None, description="随机种子")
    device: Optional[str] = Field(default=None, description="运行设备: cpu, cuda, cuda:0")
        
    # 2. QMIX 训练参数
    learning_rate: Optional[float] = Field(default=None, description="QMIX 整体学习率")
    gamma: Optional[float] = Field(default=None, description="折扣因子")
    batch_size: Optional[int] = Field(default=None, description="每次训练采样的 Episode 数量")
    buffer_size: Optional[int] = Field(default=None, description="经验池最大容量 (Episode 数)")
    update_target_params: Optional[int] = Field(default=None, description="目标网络更新频率")
    grad_norm_clip: Optional[float] = Field(default=None, description="梯度裁剪阈值")
    
    # 3. 探索与利用 (Epsilon Greedy)
    epsilon_start: Optional[float] = Field(default=None, description="初始随机探索概率")
    epsilon_finish: Optional[float] = Field(default=None, description="最终随机探索概率")
    epsilon_anneal_time: Optional[int] = Field(default=None, description="探索率退火总步数")

    # 4. 神经网络容量 (高级设置)
    drqn_hidden_dim: Optional[int] = Field(default=None, description="单体网络 GRU 隐藏层维度")
    qmix_hidden_dim: Optional[int] = Field(default=None, description="QMIX 混频网络隐藏层维度")
    hyper_hidden_dim: Optional[int] = Field(default=None, description="生成混频权重的超网络维度")

class RLTrainRequest(BaseModel):
    task_id: str
    plan_id: str
    load_dir: str = Field(..., description="选择要继续训练的权重路径")
    scene_url: str = Field(..., description="平台下发的场景文件下载链接")
    max_episodes: int = Field(..., description="最大推演局数", gt=0)
    max_episode_steps: Optional[int] = Field(default=None, description="单个回合的最大步数") 
    algorithm: str = Field(..., description="选择算法: PPO, QMIX等")
    push_interval: int = Field(default=0, description="多少回合推送一次态势。0表示不推送 2表示推送结束。") # [新增] ZMQ 态势推送配置
    hyperparameters: RLHyperparameters

class ILHyperparameters(BaseModel):
    learning_rate: Optional[float] = Field(default=None)
    seq_len: Optional[int] = Field(default=None, description="RNN时序截断长度")
    batch_size: Optional[int] = Field(default=None, description="批次大小")
    device: Optional[str] = Field(default=None, description="运行设备: cpu, cuda, cuda:0")

class ILTrainRequest(BaseModel):
    task_id: str
    plan_id: str
    load_dir: str = Field(..., description="选择要继续训练的权重路径")
    scene_url: str = Field(..., description="平台下发的场景文件下载链接")
    algorithm: str = Field(..., description="选择算法: PPO, QMIX等")
    #dataset_path: str = Field(..., description="要模仿的专家数据集路径")
    epochs: int = Field(..., description="训练总轮数")
    hyperparameters: ILHyperparameters

class EvalRequest(BaseModel):
    task_id: str
    scene_url: str = Field(..., description="平台下发的场景文件下载链接")
    algorithm: str = "qmix"
    load_dir: str          # 平台下发的模型绝对路径
    max_episodes: int = 1
    max_episode_steps: Optional[int] = Field(default=None, description="推理一回合的最大步数")  

class TaskActionRequest(BaseModel):
    task_id: str