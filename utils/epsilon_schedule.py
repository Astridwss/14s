"""
探索率调度 —— 纯计算，无状态，零依赖。
"""


class EpsilonSchedule:
    """线性衰减探索率调度器。"""

    def __init__(self, start: float, finish: float, anneal_steps: int):
        self.start = start
        self.finish = finish
        self.anneal_steps = max(1, anneal_steps)

    def get(self, env_steps: int) -> float:
        """根据当前环境交互步数返回 epsilon。"""
        if env_steps >= self.anneal_steps:
            return self.finish
        return max(self.finish,
                   self.start - (self.start - self.finish) * (env_steps / self.anneal_steps))

    @classmethod
    def from_conf(cls, conf_or_train_cfg):
        """从 RuntimeConfig 或 TrainingConfig 构建。"""
        # 兼容扁平的 conf 和聚焦的 TrainingConfig
        if hasattr(conf_or_train_cfg, 'epsilon_start'):
            return cls(
                start=conf_or_train_cfg.epsilon_start,
                finish=conf_or_train_cfg.epsilon_finish,
                anneal_steps=conf_or_train_cfg.epsilon_anneal_time,
            )
        return cls(
            start=getattr(conf_or_train_cfg, 'epsilon_start', 1.0),
            finish=getattr(conf_or_train_cfg, 'epsilon_finish', 0.05),
            anneal_steps=getattr(conf_or_train_cfg, 'epsilon_anneal_time', 50000),
        )
