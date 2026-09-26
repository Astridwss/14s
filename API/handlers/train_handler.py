"""训练/推演 handler —— 装配配置、推送通道，调度 Runner 执行。"""
import os
import traceback
from typing import Callable

from fastapi import BackgroundTasks

from use_cases.config import ConfigAssembler
from use_cases.pusher import Pusher
from use_cases.runners import RLTrainRunner, ILTrainRunner, EvalRunner, BaselineEvalRunner
from utils.task_control import TaskController
from utils.config_printer import print_config_params


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
        # 打印 RL/IL 训练、RL 推理实际使用的参数及值（核对接口传参与最终生效值）

        print_config_params(self.conf)
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
            p1 = result.get("targetCoverCount")
            p2 = result.get("interruptCount")
            p3 = result.get("coverageMultiplicity")
            p4 = result.get("trackingCoverage")
            p = result.get("totalScore")
            print(f"[BaselineEvalRunner] 目标覆盖数量评分P11:{p1}")
            print(f"[BaselineEvalRunner] 航迹连续性评分P12:{p2}")
            print(f"[BaselineEvalRunner] 航迹可靠性评分P13:{p3}")
            print(f"[BaselineEvalRunner] 跟踪覆盖率评分P14:{p4}")
            print(f"[BaselineEvalRunner] 探测效能总分P1:{p}")
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
        # finally:
            # self._cleanup_scene()

    def run_baseline_eval(self) -> dict:
        """基准推演评估（同步返回结果）—— 只生成专家预案基线，不加载模型权重。"""
        print(f"[基准推演] 收到请求: {self._task_id}")
        runner = BaselineEvalRunner(self.conf, self.push)
        try:
            result = runner.run()
            p1 = result.get("targetCoverCount")
            p2 = result.get("interruptCount")
            p3 = result.get("coverageMultiplicity")
            p4 = result.get("trackingCoverage")
            p = result.get("totalScore")
            print(f"[BaselineEvalRunner] 目标覆盖数量评分P21:{p1}")
            print(f"[BaselineEvalRunner] 航迹连续性评分P22:{p2}")
            print(f"[BaselineEvalRunner] 航迹可靠性评分P23:{p3}")
            print(f"[BaselineEvalRunner] 跟踪覆盖率评分P24:{p4}")
            print(f"[BaselineEvalRunner] 探测效能总分P2:{p}")
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
        # finally:
        #     self._cleanup_scene()

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
        """任务结束后清理任务产物（控制标志位 + 场景文件），幂等，由本服务统一回收。

        1. 清理 tmp_flag/{task_id}/ 下的 pause/terminate/speed 标志文件（及空目录），
           避免控制标志位随任务堆积。
        2. 先删 conf.local_scene_path 指向的文件，再尝试删已空的父目录
           （os.rmdir 仅对空目录生效，非空或不存在会抛 OSError，静默忽略）。
        失败不抛出，避免影响任务主流程。

        场景文件删除受 algo.yaml 的 clean_scene 开关控制：置 false 时保留场景文件与
        父目录供排查（标志位清理不受影响，始终执行，避免残留控制标志影响后续任务）。
        """
        TaskController.clear(self._task_id)

        # 配置开关：clean_scene=false 时保留场景/预案文件（排查用）
        if not getattr(self.conf, "clean_scene", True):
            print("[TrainHandler] clean_scene=false，跳过场景文件清理（保留供排查）")
            return

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
