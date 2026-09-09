"""任务控制 handler —— 暂停/恢复/终止。"""
from utils.task_control import TaskController


class ControlHandler:
    """任务控制处理器。

    用法::

        ControlHandler(task_id).pause()
        ControlHandler(task_id).resume()
        ControlHandler(task_id).terminate()
    """

    def __init__(self, task_id: str):
        self._task_id = task_id

    def pause(self) -> dict:
        TaskController.pause(self._task_id)
        return {
            "code": 200,
            "message": f"已暂停任务 {self._task_id}",
        }

    def resume(self) -> dict:
        TaskController.resume(self._task_id)
        return {
            "code": 200,
            "message": f"已继续任务 {self._task_id}",
        }

    def terminate(self) -> dict:
        TaskController.terminate(self._task_id)
        return {
            "code": 200,
            "message": f"已向任务 {self._task_id} 发送终止指令",
        }

    def set_speed(self, speed: float) -> dict:
        TaskController.set_speed(self._task_id, speed)
        return {
            "code": 200,
            "message": f"已设置任务 {self._task_id} 消费倍速为 {speed}",
        }
