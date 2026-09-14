"""
推送器 —— 训练指标/模型权重/推演结果/错误上报的统一 HTTP 通道。

用法:
    push = Pusher(task_id="xxx", base_url="http://127.0.0.1:8080")
    push.push_metrics({"episode": 1, "reward": 1.5})
    push.push_weights({"episode": 10, "model_dir": "./models/xxx", "algo": "qmix"})
    push.push_error("训练崩溃，原因: ...")
"""

import threading

import requests


class Pusher:
    """统一推送器。

    所有向平台的 HTTP 推送经由此类，Handler/Runner 不需要关心 URL 拼接和异步线程细节。
    除 ``push_error`` 为同步（确保崩溃信息不丢失）外，其余方法均为异步非阻塞。
    """

    def __init__(self, task_id: str, base_url: str):
        self._task_id = task_id
        base = base_url.rstrip("/")

        self._metrics_url = f"{base}/prod-api/agent/train/record"
        self._weights_url = f"{base}/prod-api/agent/model/record"
        self._eval_result_url = f"{base}/api/receive/eval_result" 
        self._status_url = f"{base}/prod-api/agent/train/status"
        self._baseline_eval_result_url = f"{base}/api/receive/compare_eval_result" #基线数据甘特图 

    # ============================================================
    # 公开 API
    # ============================================================

    def push_metrics(self, metrics_data: dict) -> None:
        """推送训练指标（RL / IL 共用，异步）。"""
        metrics_data["taskId"] = self._task_id
        print(f"[任务结束推送信息]:{metrics_data}")
        self._async_post(self._metrics_url, metrics_data)

    def push_weights(self, weight_data: dict) -> None:
        """推送模型权重信息（异步）。"""
        payload = {
            "taskId": self._task_id,
            "planId": weight_data.get("planId"),
            "episode": weight_data.get("episode"),
            "absModelPath": weight_data.get("model_dir"),
            "modelType": 1 if weight_data.get("algo") == "qmix" else 0,
        }
        if "absCsvPath" in weight_data:
            payload["absCsvPath"] = weight_data["absCsvPath"]
        print(f"[任务结束推送模型权重信息]:{payload}")
        self._async_post(self._weights_url, payload, "模型路径上传失败")

    def push_eval_result(self, result_data: dict) -> None:
        """推送评估/推演结果（异步）。"""
        result_data["taskId"] = self._task_id
        result_data["taskName"] = "eval"
        self._async_post(self._eval_result_url, result_data, "推理预案数据推送失败")

    def push_baseline_eval_result(self, result_data: dict) -> None:
        """推送评估/推演基线结果（异步）。"""
        result_data["taskId"] = self._task_id
        result_data["taskName"] = "baseline"
        self._async_post(self._baseline_eval_result_url, result_data, "基线数据推送失败")

    def push_error(self, error_msg: str) -> None:
        """推送崩溃信息到平台（同步，确保异常终止时不丢失）。"""
        payload = {
            "task_id": self._task_id,
            "status": "failed",
            "error_msg": error_msg,
        }
        try:
            response = requests.post(self._status_url, json=payload, timeout=5)
            print(f"已推送错误至平台，状态码: {response.status_code}")
        except Exception as e:
            print(f"无法联系平台发送错误信息: {e}")

    # ============================================================
    # 内部
    # ============================================================

    def _async_post(
        self, url: str, payload: dict, error_prefix: str = ""
    ) -> None:
        """异步 POST，daemon 线程不阻塞主线程。"""

        def _send() -> None:
            try:
                requests.post(url, json=payload, timeout=5)
            except requests.RequestException as e:
                if error_prefix:
                    print(f"[Task {self._task_id}] {error_prefix}: {e}")

        threading.Thread(target=_send, daemon=True).start()
