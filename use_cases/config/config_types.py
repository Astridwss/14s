"""
聚焦配置类型 —— 替代原来的 God Object RuntimeConfig。

将原来扁平 30+ 属性的 RuntimeConfig 拆分为 4 个关注点明确的子配置:
  - EnvConfig:    环境/场景相关（plan_id, scene_path, 雷达/目标 keys, 维度）
  - AlgorithmConfig: 算法/网络相关（device, learning_rate, drqn_hidden_dim, gamma 等）
  - TrainingConfig:  训练流程相关（max_episodes, batch_size, epsilon, buffer_size 等）
  - InfraConfig:     基础设施相关（task_id, 路径, ZMQ, 标志位文件, 推送）

每个子配置都是不可变的（frozen dataclass），构造后不可修改。
"""

import dataclasses
from typing import List, Tuple

# ---- 内部工具 ----

def _get(conf, key: str, default):
    """兼容 dict 和 attribute-based config 的值提取。

    ConfigAssembler 传入 merged dict，Runner fallback 传入 RuntimeConfig。
    """
    if isinstance(conf, dict):
        return conf.get(key, default)
    return getattr(conf, key, default)


def _ensure_tuple(v):
    """将 int 或 tuple 统一转为 tuple，防御 scene_parser 产出 int 的情况。"""
    if isinstance(v, int):
        return (v,)
    return tuple(v)


# ============================================================
# EnvConfig —— 环境 / 场景
# ============================================================

@dataclasses.dataclass(frozen=True)
class EnvConfig:
    """环境与场景维度的静态配置。"""

    plan_id: int
    local_scene_path: str
    max_episode_steps: int
    time_step: int                          # 仿真时间步长（秒），传给 sim 引擎 load_battle_scene

    # 实体 keys
    radar_keys: List[str]
    satellites_keys: List[str]
    target_keys: List[str]

    # 维度
    n_radars: int
    n_satellites: int
    n_targets: int
    n_agents: int
    n_ld: int
    n_wx: int
    n_actions: int
    obs_shape: Tuple[int, ...]
    state_shape: Tuple[int, ...]
    radar_obs_dim: int

    @property
    def agent_keys(self) -> List[str]:
        """全部智能体 keys = 雷达(LD) + 卫星(WX)，统一骨架 225 个。"""
        return list(self.radar_keys) + list(self.satellites_keys)

    @classmethod
    def from_config(cls, conf) -> "EnvConfig":
        """从扁平 RuntimeConfig 或 merged dict 构建。"""
        return cls(
            plan_id=_get(conf, "plan_id", None),
            local_scene_path=_get(conf, "local_scene_path", ""),
            max_episode_steps=_get(conf, "max_episode_steps", 600),
            time_step=_get(conf, "time_step", 1),
            radar_keys=list(_get(conf, "radar_keys", [])),
            satellites_keys=list(_get(conf, "satellites_keys", [])),
            target_keys=list(_get(conf, "target_keys", [])),
            n_radars=_get(conf, "n_radars", 0),
            n_satellites=_get(conf, "n_satellites", 0),
            n_targets=_get(conf, "n_targets", 0),
            n_agents=_get(conf, "n_agents", 1),
            n_ld=_get(conf, "n_ld", _get(conf, "n_radars", 0)),
            n_wx=_get(conf, "n_wx", _get(conf, "n_satellites", 0)),
            n_actions=_get(conf, "n_actions", 2),
            obs_shape=_ensure_tuple(_get(conf, "obs_shape", (25,))),
            state_shape=_ensure_tuple(_get(conf, "state_shape", (80,))),
            radar_obs_dim=_get(conf, "radar_obs_dim", 25),
        )


# ============================================================
# AlgorithmConfig —— 算法 / 网络
# ============================================================

@dataclasses.dataclass(frozen=True)
class AlgorithmConfig:
    """算法与神经网络的超参数配置。"""

    algorithm: str                          # "qmix" | "drqn"
    device: object                          # torch.device 或 "cpu"/"cuda"
    n_agents: int
    n_ld: int                               # 雷达(多标签)智能体数
    n_wx: int                               # 卫星(单选)智能体数
    n_actions: int
    ld_n_actions: int                       # LD 多标签动作空间（= n_targets）
    obs_shape: Tuple[int, ...]
    state_shape: Tuple[int, ...]

    # H-QMIX 分组
    group_size: int                         # 每组雷达数量, 0 或不传 = 标准 QMIX

    # DRQN / 网络结构
    drqn_hidden_dim: int
    qmix_hidden_dim: int
    hyper_hidden_dim: int
    last_action: bool
    reuse_network: bool

    # 优化器
    learning_rate: float

    # RL 特有
    gamma: float
    grad_norm_clip: float
    update_target_params: int

    @classmethod
    def from_config(cls, conf) -> "AlgorithmConfig":
        n_targets = _get(conf, "n_targets", 0)
        return cls(
            algorithm=_get(conf, "algorithm", "qmix"),
            device=_get(conf, "device", "cpu"),
            n_agents=_get(conf, "n_agents", 1),
            n_ld=_get(conf, "n_ld", _get(conf, "n_radars", 0)),
            n_wx=_get(conf, "n_wx", _get(conf, "n_satellites", 0)),
            n_actions=_get(conf, "n_actions", 2),
            ld_n_actions=_get(conf, "ld_n_actions", n_targets),
            obs_shape=_ensure_tuple(_get(conf, "obs_shape", (25,))),
            state_shape=_ensure_tuple(_get(conf, "state_shape", (80,))),
            group_size=_get(conf, "group_size", 0),
            drqn_hidden_dim=_get(conf, "drqn_hidden_dim", 128),
            qmix_hidden_dim=_get(conf, "qmix_hidden_dim", 32),
            hyper_hidden_dim=_get(conf, "hyper_hidden_dim", 64),
            last_action=_get(conf, "last_action", True),
            reuse_network=_get(conf, "reuse_network", True),
            learning_rate=_get(conf, "learning_rate", 0.001),
            gamma=_get(conf, "gamma", 0.99),
            grad_norm_clip=_get(conf, "grad_norm_clip", 10.0),
            update_target_params=_get(conf, "update_target_params", 200),
        )


# ============================================================
# TrainingConfig —— 训练流程
# ============================================================

@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """训练流程的控制参数。"""

    max_episodes: int
    batch_size: int
    train_seq_len: int   # RL 时间展开长度（子序列采样），过长会 GPU OOM
    buffer_size: int
    utd_ratio: int       # UTD 比：每新采 1 局做几次梯度更新（replay 复用）
    seed: int
    save_frequency: int

    # Epsilon 探索
    epsilon_start: float
    epsilon_finish: float
    epsilon_anneal_time: int

    # 两阶段冻结训练
    phase1_episodes: int                   # -1=纯 phase1 标定(永不切) / 0=直接联合 / N=第 N 局切 phase2
    wx_epsilon: float                      # phase2 卫星独立探索率（WX 从零学，给足探索）

    # IL 特有
    epochs: int
    seq_len: int
    train_val_split: float

    @classmethod
    def from_config(cls, conf) -> "TrainingConfig":
        return cls(
            max_episodes=_get(conf, "max_episodes", 5000),
            batch_size=_get(conf, "batch_size", 32),
            train_seq_len=_get(conf, "train_seq_len", 60),
            buffer_size=_get(conf, "buffer_size", 5000),
            utd_ratio=_get(conf, "utd_ratio", 4),
            seed=_get(conf, "seed", 42),
            save_frequency=_get(conf, "save_frequency", 100),
            epsilon_start=_get(conf, "epsilon_start", 1.0),
            epsilon_finish=_get(conf, "epsilon_finish", 0.05),
            epsilon_anneal_time=_get(conf, "epsilon_anneal_time", 50000),
            phase1_episodes=_get(conf, "phase1_episodes", -1),
            wx_epsilon=_get(conf, "wx_epsilon", 0.3),
            epochs=_get(conf, "epochs", 50),
            seq_len=_get(conf, "seq_len", 10),
            train_val_split=_get(conf, "train_val_split", 0.8),
        )


# ============================================================
# InfraConfig —— 基础设施
# ============================================================

@dataclasses.dataclass(frozen=True)
class InfraConfig:
    """基础设施与平台对接参数。"""

    task_id: str
    mode: str                                # "train" | "eval" | "baseline"

    # 路径
    model_dir: str
    result_dir: str
    load_dir: str
    load_type: str
    eval_records_dir: str

    # 任务控制
    pause_flag_file: str
    terminate_flag_file: str

    # 外部通信
    platform_base_url: str
    zmq_server_ip: str
    zmq_pub_port: int
    push_interval: int

    # 态势轨迹推送（并行模式发送线程，见 services/zmq/situation_sender.py）
    situation_base_interval: float      # 1× 倍速时的帧间隔（秒）
    situation_speed_refresh: float      # 前端倍速参数重读 TTL（秒）
    situation_warn_episodes: int        # 发送池高水位告警阈值（局）
    situation_pool_max: int             # 发送池容量上限（局，0=无上限，满则丢最旧）

    # 任务结束清理
    clean_scene: bool                   # 是否清理场景/预案文件（false=保留供排查）

    @classmethod
    def from_config(cls, conf) -> "InfraConfig":
        return cls(
            task_id=_get(conf, "task_id", ""),
            mode=_get(conf, "mode", "train"),
            model_dir=_get(conf, "model_dir", "./models"),
            result_dir=_get(conf, "result_dir", "./results"),
            load_dir=_get(conf, "load_dir", ""),
            load_type=_get(conf, "load_type", "il"),
            eval_records_dir=_get(conf, "eval_records_dir", "./eval_records"),
            pause_flag_file=_get(conf, "pause_flag_file", ""),
            terminate_flag_file=_get(conf, "terminate_flag_file", ""),
            platform_base_url=_get(conf, "PLATFORM_BASE_URL", ""),
            zmq_server_ip=_get(conf, "zmq_server_ip", "192.168.1.51"),
            zmq_pub_port=int(_get(conf, "zmq_pub_port", 5558)),
            push_interval=int(_get(conf, "push_interval", 0)),
            situation_base_interval=float(_get(conf, "situation_base_interval", 0.5)),
            situation_speed_refresh=float(_get(conf, "situation_speed_refresh", 0.5)),
            situation_warn_episodes=int(_get(conf, "situation_warn_episodes", 200)),
            situation_pool_max=int(_get(conf, "situation_pool_max", 0)),
            clean_scene=bool(_get(conf, "clean_scene", True)),
        )
