"""
BaseRunner —— 所有训练/推演 Runner 的抽象基类。

统一提供:
  - conf / push 注入
  - TaskController（暂停/终止标志位轮询）
  - _should_continue(step) → bool
"""

from utils.task_control import TaskController


class BaseRunner:
    """Runner 基类：接收配置与推送通道，管理任务生命周期。

    子类只需实现 ``run()``，其余由基类统一处理。
    """

    def __init__(self, conf, push=None):
        self.conf = conf
        self.push = push
        self._ctrl = TaskController(
            conf.pause_flag_file, conf.terminate_flag_file
        )

    def _should_continue(self, step: int) -> bool:
        """检查任务是否应继续（未被暂停或终止）。"""
        task_id = getattr(self.conf, "task_id", "")
        return self._ctrl.check(task_id, step)

    def run(self):
        """执行训练/推演主循环。子类必须实现。"""
        raise NotImplementedError
