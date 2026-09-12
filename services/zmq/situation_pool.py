"""态势轨迹池 —— 建池/存池：整局轨迹的数据结构 + 线程安全 FIFO。

SituationTrajectory 是入池的原子单位（一整局序列化帧字节），
SituationPool 是线程安全 FIFO，附高水位告警与容量上限（满则丢最旧防内存爆）。
"""

import queue
from dataclasses import dataclass, field
from typing import List


@dataclass
class SituationTrajectory:
    """一整局态势轨迹：episode_idx + 按 step 顺序的序列化帧字节。

    每帧字节已由 build_situation_frame 序列化好并烘焙了 EpisodeIdx/StepIdx，
    发送线程只负责按序 send，不再组装。
    """
    episode_idx: int
    frames: List[bytes] = field(default_factory=list)


class SituationPool:
    """线程安全 FIFO 池（queue.Queue 封装），附高水位告警与容量上限。

    max_episodes > 0 时启用容量上限：池满则丢最旧（丢弃队首整局轨迹），
    保证前端始终消费最新态势，避免消费落后时无界积压撑爆内存。
    """

    DRAIN = object()  # 哨兵：发送线程收到后发完现有数据并退出

    def __init__(self, warn_episodes: int = 200, max_episodes: int = 0):
        self._q = queue.Queue()
        self._warn_episodes = warn_episodes
        self._max_episodes = max_episodes
        self._warned = False
        self._dropped = 0
        self._drop_logged = False

    def put(self, traj: SituationTrajectory) -> None:
        # 容量上限：满则丢最旧，避免前端消费落后时无界积压撑爆内存
        if self._max_episodes > 0:
            while self._q.qsize() >= self._max_episodes:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
                self._dropped += 1
            if self._dropped and not self._drop_logged:
                print(f"[SituationPool] 态势轨迹池达容量上限 {self._max_episodes} 局，"
                      f"开始丢弃最旧轨迹（前端消费落后，已丢 {self._dropped} 局）")
                self._drop_logged = True

        # 将态势轨迹添加到队列中
        self._q.put(traj)
        n = self._q.qsize()
        if n >= self._warn_episodes and not self._warned:
            print(f"[SituationPool] 态势轨迹池水位 {n} 局已达阈值 {self._warn_episodes}，"
                  f"前端消费落后于采集，建议调大倍速或检查前端连接")
            self._warned = True
        elif n < self._warn_episodes // 2:
            self._warned = False

    def get(self, timeout=None):
        """阻塞取一局（或哨兵 DRAIN）；传 timeout（秒）时超时抛 queue.Empty。"""
        return self._q.get(timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()
