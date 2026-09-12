"""态势发送线程 —— 从池取局，按前端倍速逐帧 ZMQ push。"""

import math
import queue
import threading
import time

from utils.task_control import TaskController
from services.zmq.situation_pool import SituationPool, SituationTrajectory


class SituationSender(threading.Thread):
    """独立发送线程：从池取局 → 按前端倍速逐帧 ZMQ push。

    base_interval 定义 1× 倍速的帧间隔（秒），speed 为前端倍速系数：
    帧间隔 = base_interval / speed。speed 越大消费越快、池子越空。
    """

    def __init__(
        self,
        publisher,
        task_id: str,
        pool: SituationPool,
        base_interval: float = 0.5,
        speed_refresh: float = 0.5,
    ):
        super().__init__(daemon=True, name=f"SituationSender-{task_id}")
        self._publisher = publisher
        self._task_id = task_id
        self._pool = pool
        self._base_interval = base_interval
        self._speed_refresh = speed_refresh
        # 注意：不能命名为 `self._stop` —— 那会覆盖 threading.Thread 的私有方法
        # `_stop()`（线程 run() 结束后由 _bootstrap_inner/_delete 调用，用于释放
        # _tstate_lock 让 join()/is_alive() 正确返回），导致线程退出时报
        # `TypeError: 'Event' object is not callable`。
        self._stop_event = threading.Event()
        self._speed = 1.0
        self._speed_ts = 0.0

    def _current_speed(self) -> float:
        """按 TTL 重读前端倍速参数（避免每帧读文件）。

        防御：flag 若被写成 '0'/'负数'/'nan'/'inf' 或任何非有限数，下游
        `0.5 / max(speed, 1e-3)` 会得到 500s 长睡，或 `time.sleep(nan)` 抛
        ValueError 直接打死本线程；这里统一钳制回 1.0。
        """
        now = time.monotonic()
        if now - self._speed_ts >= self._speed_refresh:
            new_speed = TaskController.read_speed(self._task_id, default=1.0)
            if not (isinstance(new_speed, (int, float))
                    and math.isfinite(new_speed) and new_speed > 0):
                print(f"[SituationSender] 倍速值非法({new_speed!r})，回退 1.0")
                new_speed = 1.0
            if new_speed != self._speed:
                print(f"[SituationSender] 倍速变更: {self._speed:.2f} -> {new_speed:.2f}")
            self._speed = new_speed
            self._speed_ts = now
        return self._speed

    def run(self):
        sent_count = 0
        last_empty_log = 0.0
        try:
            while not self._stop_event.is_set():
                try:
                    item = self._pool.get(timeout=5.0)
                except queue.Empty:
                    # 池空 5s：周期性提示（30s 一次）。长期池空 = 生产跟不上消费 = [2]，
                    # 此时 speed 无效（发送线程无数据可发）。
                    now = time.monotonic()
                    if now - last_empty_log >= 30.0:
                        print("[SituationSender] 池空：持续无局可发（长期如此=[2]生产跟不上，调 speed 无效）")
                        last_empty_log = now
                    continue
                if item is SituationPool.DRAIN:
                    break
                self._send_traj(item)
                sent_count += 1
                print(f"[SituationSender] 发完第 {sent_count} 局, 池中待发 {self._pool.qsize()} 局"
                      f"（待发>0=有积压，发送线程是瓶颈，speed 应生效）")
        except Exception:
            # daemon 线程：不在这里打印的话，异常会静默吞掉，表现为「推一轮后停止」且无从排查
            import traceback
            print(f"[SituationSender] 发送线程异常退出:\n{traceback.format_exc()}")

    def _send_traj(self, traj: SituationTrajectory):
        for payload in traj.frames:
            if self._stop_event.is_set():
                return

            self._publisher.send_payload(payload)
            speed = self._current_speed()
            time.sleep(self._base_interval / max(speed, 1e-3))

    def finish(self, timeout: float = 30.0):
        """正常结束：发完池中剩余所有局再退出（不丢轨迹）。

        超时仍未发完（前端消费过慢）则强制停止，避免训练进程退出挂起。
        """
        self._pool.put(SituationPool.DRAIN)
        self.join(timeout=timeout)
        if self.is_alive():
            self._stop_event.set()
            self.join(timeout=3)
            print(f"[SituationSender] 关闭超时（>{timeout}s），剩余帧已丢弃")

    def stop(self):
        """强制终止：立即丢弃池中剩余并退出。"""
        self._stop_event.set()
        self._pool.put(SituationPool.DRAIN)  # 唤醒阻塞在 get() 的线程
        self.join(timeout=3)
