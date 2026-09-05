"""
API 请求数据校验模型

定义所有入参的 Pydantic 结构，用于 FastAPI 自动校验与文档生成。
"""
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 超参数子结构
# ============================================================

class RLHyperparameters(BaseModel):
    """强化学习超参数（全部可选，未传则使用 Config 默认值）"""

    # 基础环境与日志控制
    show_log: Optional[bool] = Field(default=True, description="是否在控制台打印日志")
    seed: Optional[int] = Field(default=None, description="随机种子")
    device: Optional[str] = Field(default=None, description="运行设备: cpu, cuda, cuda:0")

    # QMIX 训练参数
    learning_rate: Optional[float] = Field(default=None, description="QMIX 整体学习率")
    gamma: Optional[float] = Field(default=None, description="折扣因子")
    batch_size: Optional[int] = Field(default=None, description="每次训练采样的 Episode 数量")
    buffer_size: Optional[int] = Field(default=None, description="经验池最大容量 (Episode 数)")
    update_target_params: Optional[int] = Field(default=None, description="目标网络更新频率")
    grad_norm_clip: Optional[float] = Field(default=None, description="梯度裁剪阈值")

    # 探索与利用 (Epsilon Greedy)
    epsilon_start: Optional[float] = Field(default=None, description="初始随机探索概率")
    epsilon_finish: Optional[float] = Field(default=None, description="最终随机探索概率")
    epsilon_anneal_time: Optional[int] = Field(default=None, description="探索率退火总步数")

    # 神经网络容量 (高级设置)
    drqn_hidden_dim: Optional[int] = Field(default=None, description="单体网络 GRU 隐藏层维度")
    qmix_hidden_dim: Optional[int] = Field(default=None, description="QMIX 混频网络隐藏层维度")
    hyper_hidden_dim: Optional[int] = Field(default=None, description="生成混频权重的超网络维度")


class ILHyperparameters(BaseModel):
    """模仿学习超参数（全部可选）"""

    learning_rate: Optional[float] = Field(default=None, description="学习率")
    seq_len: Optional[int] = Field(default=None, description="RNN 时序截断长度")
    batch_size: Optional[int] = Field(default=None, description="批次大小")
    device: Optional[str] = Field(default=None, description="运行设备: cpu, cuda, cuda:0")


# ============================================================
# 训练 / 推演请求
# ============================================================

class RLTrainRequest(BaseModel):
    """强化学习训练请求"""

    task_id: str
    plan_id: int
    load_dir: str = Field(..., description="选择要继续训练的权重路径")
    scene_url: str = Field(..., description="平台下发的场景文件本地路径")
    max_episodes: int = Field(..., description="最大推演局数", gt=0)
    max_episode_steps: Optional[int] = Field(default=None, description="单个回合的最大步数")
    group_size: Optional[int] = Field(default=None, description="H-QMIX 每组雷达数量，0=标准 QMIX")
    algorithm: str = Field(..., description="选择算法: PPO, QMIX等")
    push_interval: Optional[int] = Field(default=None, description="每 N 回合推送一次态势，0 表示不推送")
    rl_num_workers: Optional[int] = Field(default=None, description="RL 并行环境采样 worker 数（>1 启用多进程 rollout；按部署机器 CPU 物理核数调，超核数会被自动 clamp）")
    hyperparameters: RLHyperparameters


class ILTrainRequest(BaseModel):
    """模仿学习训练请求"""

    task_id: str
    plan_id: int
    load_dir: str = Field(..., description="选择要继续训练的权重路径")
    scene_url: str = Field(..., description="平台下发的场景文件本地路径")
    algorithm: str = Field(
        default="drqn_il",
        description="IL 训练固定走 DRQN 预训练，必须为 'drqn_il' 才能命中 algo.yaml 的 drqn_il 段"
                    "（专属学习率 0.001 / seq_len 50 / 数据增强等超参）；传 qmix/DRQN 等其它值会"
                    "静默回退到 qmix 段，导致学习率与 seq_len 用错，影响 IL 表征与 RL 热启动质量",
    )
    epochs: int = Field(..., description="训练总轮数")
    il_num_workers: Optional[int] = Field(default=None, description="IL 数据加载并行进程数（DataLoader num_workers；Windows 开发机报错可设 0）")
    hyperparameters: ILHyperparameters


class EvalRequest(BaseModel):
    """推演评估请求"""

    task_id: str
    scene_url: str = Field(..., description="平台下发的场景文件本地路径")
    algorithm: str = "qmix"
    load_dir: str = Field(..., description="平台下发的模型绝对路径")
    max_episodes: int = 1
    max_episode_steps: Optional[int] = Field(default=None, description="推理一回合的最大步数")
    group_size: Optional[int] = Field(default=None, description="H-QMIX 每组雷达数量，需与训练时一致")
    push_interval: Optional[int] = Field(default=None, description="每 N 回合推送一次态势，0 表示不推送")


class TaskActionRequest(BaseModel):
    """任务控制请求（暂停 / 恢复 / 终止）"""

    task_id: str
