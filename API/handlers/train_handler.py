"""训练/推演 handler —— 装配配置、推送通道，调度 Runner 执行。"""
import os
import traceback
from typing import Callable

from fastapi import BackgroundTasks

from use_cases.config import ConfigAssembler
from use_cases.pusher import Pusher
from use_cases.runners import RLTrainRunner, ILTrainRunner, EvalRunner, BaselineEvalRunner


class TrainHandler:
    """训练/推演请求处理器。

    每个 HTTP 请求创建一个实例，在 ``__init__`` 中一次性装配配置和推送通道，
    各 ``run_xxx`` 方法直接调度对应的 Runner。

    用法::

        # RL 训练（异步）
        handler = TrainHandler(request, mode="train")
        handler.run_rl(background_tasks)

        # 推演评估（同步）
        handler = TrainHandler(request, mode="eval")
        result = handler.run_eval()
    """

    def __init__(self, request, mode: str):
        self._request = request
        self._task_id = request.task_id

        # ---- 一次性装配 ----
        self.conf = ConfigAssembler(self._task_id, request, mode).build()
        self.push = Pusher(
            task_id=self._task_id, base_url=self.conf.PLATFORM_BASE_URL
        )

    # ============================================================
    # 公开入口
    # ============================================================

    def run_rl(self, bg: BackgroundTasks) -> dict:
        """RL 训练（异步后台执行）。"""
        print(f"[RL训练] 收到请求: {self._task_id}")
        runner = RLTrainRunner(self.conf, self.push)
        bg.add_task(
            self._async_runner, "RL训练", runner.run
        )
        return self._ok_response(self._request.algorithm)

    def run_il(self, bg: BackgroundTasks) -> dict:
        """IL 训练（异步后台执行）。"""
        print(f"[IL训练] 收到请求: {self._task_id}")
        runner = ILTrainRunner(self.conf, self.push)
        bg.add_task(
            self._async_runner, "IL训练", runner.run
        )
        return self._ok_response(self._request.algorithm)

    def run_eval(self) -> dict:
        """推演评估（同步返回结果）。"""
        print(f"[推演] 收到请求: {self._task_id}")
        runner = EvalRunner(self.conf, self.push)
        try:
            result = runner.run()
            return {
                "code": 200,
                "message": "推演任务已成功完成",
                "data": {
                    "taskId": self._task_id,
                    "algo": self._request.algorithm,
                    "status": "finished",
                    **result,
                },
            }
        except Exception as e:
            print(f"[推演] 失败: {self._task_id} - {e}")
            self.push.push_error(str(e))
            return {
                "code": 500,
                "message": f"推演任务失败: {str(e)}",
                "data": {"taskId": self._task_id, "status": "failed"},
            }
        finally:
            self._cleanup_scene()

    def run_baseline_eval(self) -> dict:
        """基准推演评估（同步返回结果）—— 只生成专家预案基线，不加载模型权重。"""
        print(f"[基准推演] 收到请求: {self._task_id}")
        runner = BaselineEvalRunner(self.conf, self.push)
        try:
            result = runner.run()
            return {
                "code": 200,
                "message": "基准推演任务已成功完成",
                "data": {
                    "taskId": self._task_id,
                    "algo": self._request.algorithm,
                    "status": "finished",
                    **result,
                },
            }
        except Exception as e:
            print(f"[基准推演] 失败: {self._task_id} - {e}")
            self.push.push_error(str(e))
            return {
                "code": 500,
                "message": f"基准推演任务失败: {str(e)}",
                "data": {"taskId": self._task_id, "status": "failed"},
            }
        finally:
            self._cleanup_scene()

    # ============================================================
    # 私有
    # ============================================================

    def _ok_response(self, algorithm: str) -> dict:
        return {
            "code": 200,
            "message": "训练任务已成功下发至后台",
            "data": {
                "task_id": self._task_id,
                "algorithm": algorithm,
                "status": "starting",
            },
        }

    def _async_runner(self, label: str, fn: Callable[[], None]) -> None:
        """异步执行 fn()，异常时 print + push_error，并回收 CUDA 显存缓存。"""
        try:
            fn()
            print(f"[{label}] 训练完成: {self._task_id}")
        except Exception:
            error_detail = traceback.format_exc()
            print(f"[Task {self._task_id}] {label}崩溃: {error_detail}")
            try:
                self.push.push_error(error_detail)
            except Exception:
                print(f"[Task {self._task_id}] 无法通知平台训练崩溃")
        finally:
            self._release_cuda_cache()
            self._cleanup_scene()

    def _cleanup_scene(self) -> None:
        """任务结束后清理场景文件（幂等，由本服务统一回收，避免 task 目录堆积）。

        先删 conf.local_scene_path 指向的文件，再尝试删已空的父目录
        （os.rmdir 仅对空目录生效，非空或不存在会抛 OSError，静默忽略）。
        失败不抛出，避免影响任务主流程。
        """
        scene_path = getattr(self.conf, "local_scene_path", "")
        if not scene_path or not os.path.exists(scene_path):
            return
        try:
            os.remove(scene_path)
            print(f"[TrainHandler] 已清理场景文件: {scene_path}")
        except OSError as e:
            print(f"[TrainHandler] 清理场景文件失败（忽略）: {scene_path} - {e}")
            return

        # 文件删除成功后，清理已空的父目录（如 scenarios/{task_id}/）
        parent_dir = os.path.dirname(scene_path)
        try:
            os.rmdir(parent_dir)
            print(f"[TrainHandler] 已清理空目录: {parent_dir}")
        except OSError:
            pass  # 目录非空或不存在，忽略

    @staticmethod
    def _release_cuda_cache() -> None:
        """释放 PyTorch 缓存的显存（CUDA / NPU），避免多次训练任务累加占用导致 OOM。"""
        from utils.torch_device import empty_device_cache
        empty_device_cache()
