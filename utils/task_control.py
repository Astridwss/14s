"""
任务控制 —— 基于文件标志位的暂停/终止。

单一来源：路径计算 + 写入（HTTP 控制）+ 读取（运行时轮询），全部在此。
"""

import os
import time


class TaskController:
    """文件标志位：运行时轮询 + HTTP 控制操作，统一入口。

    运行时轮询::

        ctrl = TaskController(pause_file="...", terminate_file="...")
        for episode in range(max_episodes):
            if not ctrl.check(task_id, episode):
                break

    HTTP 控制（类方法）::

        TaskController.pause(task_id)
        TaskController.resume(task_id)
        TaskController.terminate(task_id)
    """

    # ============================================================
    # 路径（单一来源 —— ConfigAssembler 也调用此处）
    # ============================================================

    @staticmethod
    def _flag_dir() -> str:
        d = os.path.join(os.environ.get("PROJECT_ROOT", "."), "temp_flags")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def pause_path(task_id: str) -> str:
        return os.path.join(TaskController._flag_dir(), f"{task_id}_pause.flag")

    @staticmethod
    def terminate_path(task_id: str) -> str:
        return os.path.join(
            TaskController._flag_dir(), f"{task_id}_terminate.flag"
        )

    # ============================================================
    # HTTP 控制 —— 写标志文件
    # ============================================================

    @classmethod
    def pause(cls, task_id: str) -> None:
        """下发暂停指令（创建标志文件）。"""
        with open(cls.pause_path(task_id), "w") as f:
            f.write("PAUSED")

    @classmethod
    def resume(cls, task_id: str) -> None:
        """下发恢复指令（删除暂停标志文件）。"""
        path = cls.pause_path(task_id)
        if os.path.exists(path):
            os.remove(path)

    @classmethod
    def terminate(cls, task_id: str) -> None:
        """下发终止指令（创建终止标志 + 清除暂停标志）。"""
        with open(cls.terminate_path(task_id), "w") as f:
            f.write("TERMINATE")
        # 同时清除暂停标志，避免被暂停阻塞
        cls.resume(task_id)

    # ============================================================
    # 运行时轮询
    # ============================================================

    def __init__(self, pause_file: str = '', terminate_file: str = ''):
        self._pause_file = pause_file
        self._terminate_file = terminate_file

    def check(self, task_id: str = '', episode_idx: int = 0) -> bool:
        """在每局开始前调用。返回 True 继续，False 终止。"""
        if self._terminate_file and os.path.exists(self._terminate_file):
            print(f"[TaskController] 检测到终止信号任务 {task_id}，强行退出...")
            os.remove(self._terminate_file)
            return False

        if self._pause_file and os.path.exists(self._pause_file):
            print(f"[TaskController] 检测到暂停信号，训练挂起 (Episode: {episode_idx})...")
            while os.path.exists(self._pause_file):
                time.sleep(2)
            print(f"[TaskController] 暂停信号解除，训练继续")

        return True
