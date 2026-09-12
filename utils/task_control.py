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
        """全部任务控制标志位的根目录（单一来源，ConfigAssembler 也调用此处）。"""
        d = os.path.join(os.environ.get("PROJECT_ROOT", "."), "tmp_flag")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _task_dir(task_id: str) -> str:
        """单个任务的标志位子目录 tmp_flag/{task_id}/（每个任务一个独立目录）。"""
        d = os.path.join(TaskController._flag_dir(), task_id)
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def pause_path(task_id: str) -> str:
        return os.path.join(TaskController._task_dir(task_id), "pause.flag")

    @staticmethod
    def terminate_path(task_id: str) -> str:
        return os.path.join(TaskController._task_dir(task_id), "terminate.flag")

    @staticmethod
    def speed_path(task_id: str) -> str:
        return os.path.join(TaskController._task_dir(task_id), "speed.flag")

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

    @classmethod
    def set_speed(cls, task_id: str, speed: float) -> None:
        """下发前端消费倍速（写 speed 标志文件，发送线程按 TTL 重读）。"""
        with open(cls.speed_path(task_id), "w") as f:
            f.write(str(float(speed)))

    @classmethod
    def read_speed(cls, task_id: str, default: float = 1.0) -> float:
        """读取前端消费倍速；文件不存在或非法时返回 default。"""
        try:
            with open(cls.speed_path(task_id)) as f:
                return float(f.read().strip())
        except (OSError, ValueError):
            return default

    @classmethod
    def clear(cls, task_id: str) -> None:
        """清理任务全部控制标志位（pause/terminate/speed），任务结束后调用。

        幂等：文件不存在或删除失败均静默忽略，避免影响任务主流程。
        删除标志位后，再尝试移除已空的 per-task 目录 tmp_flag/{task_id}/。
        """
        for path in (cls.pause_path(task_id),
                     cls.terminate_path(task_id),
                     cls.speed_path(task_id)):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

        # 清理已空的 per-task 子目录（目录非空或不存在时静默忽略）
        try:
            os.rmdir(cls._task_dir(task_id))
        except OSError:
            pass

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
