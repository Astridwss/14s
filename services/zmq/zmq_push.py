"""
ZMQ 态势推送 —— 训练过程中向消息总线推送态势帧数据。

提供的类:
    SituationPublisher          — ZMQ 通信层：封装 Proto 协议并推送态势帧
    SituationCollector          — 态势采集 hook：只采集不发送，攒成一整局轨迹（并行模式用）
    ZMQStepHook                 — step 回调适配器：串行模式下逐帧推送（按 min_interval 降频）
    TrainingSituationPushService — 训练态势推送服务：封装构建、调度、推送、资源释放

并行模式的推送路径已改为「采集 → 入池 → 独立发送线程」（见 services/zmq/trajectory_pool.py），
此处提供纯函数 build_situation_frame 供采集器与发送端共用，避免重复组装。
"""

import time
from typing import Optional

import zmq

from services.zmq.proto import ProtoStruct_pb2


def _safe_extract_id(id_str) -> int:
    """安全地从字符串中提取纯数字 ID，适应 '1' 或 'R01' 等格式。"""
    digits = ''.join(filter(str.isdigit, str(id_str)))
    return int(digits) if digits else 0


def build_situation_frame(
    current_time: float,
    data_type: int,
    episode_idx: int,
    step_idx: int,
    raw_obs,
    valid_cmds,
) -> bytes:
    """组装并序列化一帧态势 proto（纯函数，无网络 I/O）。

    供 SituationPublisher.push_frame（串行发送）、SituationCollector（并行采集入池）共用。
    """
    msg = ProtoStruct_pb2.PR_I_RL_TRAINING_SITUATION_TO_FRONT()
    msg.CurrentTime = float(current_time)
    msg.DataType = int(data_type)
    msg.EpisodeIdx = int(episode_idx)
    msg.StepIdx = int(step_idx)
    msg.PubCtrl.MsgHeader.MsgType = 9290

    # 1. 红方装备位置
    if hasattr(raw_obs, 'dict_equip_state') and raw_obs.dict_equip_state:
        for eid, equip_state in raw_obs.dict_equip_state.items():
            equip = msg.EquipPos.add()
            equip.EquipID = _safe_extract_id(equip_state.str_equip_id)
            equip.EquipType = equip_state.type
            equip.Time = msg.CurrentTime
            equip.GeoPos.X = float(equip_state.longitude)
            equip.GeoPos.Y = float(equip_state.latitude)
            equip.GeoPos.Z = float(equip_state.altitude)

    # 2. 蓝方目标位置
    if hasattr(raw_obs, 'dict_system_track') and raw_obs.dict_system_track:
        for tid, track_base in raw_obs.dict_system_track.items():
            t_pos = msg.TargetPos.add()
            t_pos.TargetID = _safe_extract_id(track_base.str_system_track_no)
            t_pos.Time = msg.CurrentTime
            t_pos.GeoPos.X = float(track_base.longitude)
            t_pos.GeoPos.Y = float(track_base.latitude)
            t_pos.GeoPos.Z = float(track_base.altitude)

    # 3. 探测/锁定关系
    if valid_cmds:
        for cmd in valid_cmds:
            if cmd.str_target_id and cmd.str_target_id not in ("", "0"):
                det = msg.Detection.add()
                det.EquipID = _safe_extract_id(cmd.str_equip_id)
                det.TargetID = _safe_extract_id(cmd.str_target_id)

    return msg.SerializeToString()


def _extract_frame_inputs(step_count, terminated, truncated, next_info):
    """从 next_info 提取一帧态势所需输入，raw_obs 缺失时返回 None。

    返回 (raw_obs, valid_cmds, action_time, data_type)。
    """
    raw_obs = next_info.get('raw_obs')
    if raw_obs is None:
        return None

    all_cmds = next_info.get('agent_actions_list', [])
    valid_cmds = [
        cmd for cmd in all_cmds
        if getattr(cmd, 'str_target_id', "") not in ("", "0")
    ]
    action_time = float(next_info.get('action_time', 0.0))

    if step_count == 0:
        data_type = 1   # 第一帧
    elif terminated or truncated:
        data_type = 2   # 最后一帧
    else:
        data_type = 0   # 中间过程帧

    return raw_obs, valid_cmds, action_time, data_type


# ============================================================
# SituationPublisher —— ZMQ 通信层
# ============================================================

class SituationPublisher:
    """ZMQ 态势数据推送器 —— 纯通信层，不关心训练逻辑。

    用法:
        pub = SituationPublisher(host="192.168.1.51", port=5558)
        pub.push_frame(task_id, raw_obs, valid_cmds, current_time)
        pub.close()
    """

    def __init__(self, host: str = "192.168.1.51", port: int = 5558):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)

        connect_addr = f"tcp://{self.host}:{self.port}"
        # LINGER=0：关闭时立即丢弃未发送消息，避免 context.term() 阻塞
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(connect_addr)
        self._drop_logged = False
        # 丢帧统计（③ 诊断）：累计丢帧 + 5s 窗口丢帧数，周期性汇总打印
        self._drop_count = 0
        self._drop_since_report = 0
        self._last_drop_report = 0.0
        print(f"[ZMQ Publisher] 态势推送服务已启动，已连接到消息总线: {connect_addr}")

    def push_frame(
        self,
        task_id: str,
        raw_obs,
        valid_cmds,
        current_time: float,
        data_type: int = 0,
        episode_idx: int = 0,
        step_idx: int = 0,
    ):
        """组装 Proto 并推送一帧态势数据（串行模式逐帧推送用）。

        :param task_id:    任务 ID（保留参数，实际 topic 固定为 9290）
        :param raw_obs:    AgentObservation 实例
        :param valid_cmds: List[AgentActionCommand]
        :param current_time: float
        :param data_type:  1-第一帧, 2-最后一帧, 0-中间过程帧
        :param episode_idx: 局索引
        :param step_idx:    局内步索引
        """
        payload = build_situation_frame(
            current_time=current_time, data_type=data_type,
            episode_idx=episode_idx, step_idx=step_idx,
            raw_obs=raw_obs, valid_cmds=valid_cmds,
        )
        self.send_payload(payload)

    def _record_drop(self):
        """丢帧计数 + 周期性告警（③ 诊断：持续丢帧 = 中间件消费跟不上）。"""
        self._drop_count += 1
        self._drop_since_report += 1
        now = time.monotonic()
        if not self._drop_logged:
            print("[ZMQ Publisher] 发送队列满/对端未就绪，开始丢帧（非阻塞，不影响训练）")
            self._drop_logged = True
            self._last_drop_report = now
        elif now - self._last_drop_report >= 5.0:
            print(f"[ZMQ Publisher] 持续丢帧：近 5 秒丢 {self._drop_since_report} 帧"
                  f"（累计 {self._drop_count}）→ 中间件消费跟不上=③")
            self._drop_since_report = 0
            self._last_drop_report = now

    def send_payload(self, payload: bytes):
        """发送已序列化好的态势帧字节（发送线程用，跳过重复组装）。"""
        # 整包可写才发：send_multipart(NOBLOCK) 在多帧消息中途 EAGAIN 会留下半包，
        # 污染 multipart 帧边界（前端 recv_multipart 错位 → 后续全乱帧）。先用 poll
        # 判可写，不可写则整包丢弃，绝不半包。
        if self.socket.poll(0, zmq.POLLOUT) == 0:
            self._record_drop()
            return
        try:
            msg_type_str = str(9290).encode('utf-8')
            self.socket.send_multipart([msg_type_str, payload], flags=zmq.NOBLOCK)

            if self._drop_logged:
                print(f"[ZMQ Publisher] 消息总线已恢复，恢复正常推送（累计已丢 {self._drop_count} 帧）")
                self._drop_logged = False
                self._drop_since_report = 0
        except zmq.Again:
            self._record_drop()
        except Exception:
            import traceback
            print(f"[ZMQ Publisher] 发送态势崩溃: {traceback.format_exc()}")

    def close(self):
        """安全关闭 Socket 与 Context。"""
        self.socket.close()
        self.context.term()


# ============================================================
# SituationCollector —— 并行模式：只采集不发送
# ============================================================

class SituationCollector:
    """态势采集 hook —— 每步把序列化帧攒成一整局轨迹，不直接发送。

    并行模式下替代 ZMQStepHook：worker 侧不再在 step 循环里做 ZMQ 网络 I/O，
    而是把整局轨迹回主进程入池，由 SituationSender 线程按前端倍速逐帧推送。
    """

    def __init__(self, episode_idx: int):
        self.episode_idx = episode_idx
        self.frames: list = []

    def __call__(self, step_count: int, terminated: bool, truncated: bool, next_info: dict):
        inputs = _extract_frame_inputs(step_count, terminated, truncated, next_info)
        if inputs is None:
            print(f"[ZMQ Debug] 第 {step_count} 步没有拿到 raw_obs，跳过采集")
            return
        raw_obs, valid_cmds, action_time, data_type = inputs

        payload = build_situation_frame(
            current_time=action_time, data_type=data_type,
            episode_idx=self.episode_idx, step_idx=step_count,
            raw_obs=raw_obs, valid_cmds=valid_cmds,
        )
        self.frames.append(payload)


# ============================================================
# ZMQStepHook —— 串行模式：逐帧推送
# ============================================================

class ZMQStepHook:
    """Rollout 步进事件的 ZMQ 回调钩子（串行模式）。

    将数据清洗与通信逻辑从核心算法层抽离，
    把 SituationPublisher 适配为 RolloutWorker 的 step_hook 接口。
    """

    def __init__(self, publisher: SituationPublisher, task_id: str,
                 episode_idx: int = 0, min_interval: float = 0.5):
        self.publisher = publisher
        self.task_id = task_id
        self.episode_idx = episode_idx
        self.min_interval = min_interval
        self._last_push = -float("inf") ## 初始 -inf，保证首帧必推

    def __call__(self, step_count: int, terminated: bool, truncated: bool, next_info: dict):
        """当环境执行完 step 后，触发此调用。"""
        inputs = _extract_frame_inputs(step_count, terminated, truncated, next_info)
        if inputs is None:
            print(f"[ZMQ Debug] 第 {step_count} 步没有拿到 raw_obs，跳过推送")
            return
        raw_obs, valid_cmds, action_time, data_type = inputs

        # 时间节流：首帧/末帧必推，中间帧按 min_interval 降频（跳过而非 sleep，不阻塞训练）
        if data_type == 0:
            now = time.monotonic()
            if now - self._last_push < self.min_interval:
                return
            self._last_push = now

        self.publisher.push_frame(
            task_id=self.task_id,
            raw_obs=raw_obs,
            valid_cmds=valid_cmds,
            current_time=action_time,
            data_type=data_type,
            episode_idx=self.episode_idx,
            step_idx=step_count,
        )


# ============================================================
# TrainingSituationPushService —— 训练态势推送服务
# ============================================================

class TrainingSituationPushService:
    """训练态势推送服务 —— 封装 ZMQ 推送的构建、调度、推送、资源释放。

    用法:
        zmq = TrainingSituationPushService.from_conf(conf)
        if zmq:
            hooks = zmq.get_step_hooks(episode_idx)
        # 或直接推送一帧：
            zmq.push_frame(task_id, raw_obs, valid_cmds, current_time)
    """

    def __init__(
        self,
        publisher: SituationPublisher,
        task_id: str,
        push_interval: int,
        max_episodes: int,
    ):
        self._publisher = publisher
        self._task_id = task_id
        self._push_interval = push_interval
        self._max_episodes = max_episodes

    @property
    def publisher(self) -> SituationPublisher:
        """底层 Publisher，供并行发送线程复用其 socket。"""
        return self._publisher

    # ---- 工厂方法 ----

    @classmethod
    def from_conf(cls, conf) -> Optional["TrainingSituationPushService"]:
        """从配置对象构建服务。push_interval <= 0 时返回 None（禁用）。"""
        push_interval = getattr(conf, 'push_interval', 0)
        if push_interval <= 0:
            return None

        return cls(
            publisher=SituationPublisher(
                host=getattr(conf, 'zmq_server_ip', '192.168.1.51'),
                port=getattr(conf, 'zmq_pub_port', 5558),
            ),
            task_id=getattr(conf, 'task_id', ''),
            push_interval=push_interval,
            max_episodes=conf.max_episodes,
        )

    # ---- 供 Runner 使用 ----

    def get_step_hooks(self, episode_idx: int) -> list:
        """返回当前 episode 应使用的 step hooks（含调度逻辑）。

        仅在 episode_idx 命中推送间隔或为最后一个 episode 时返回 hook。
        每个 episode 独立新建 ZMQStepHook，把 episode_idx 烘焙进去。
        """
        if self._push_interval <= 0:
            return []
        if episode_idx % self._push_interval == 0 or episode_idx == self._max_episodes:
            return [ZMQStepHook(self._publisher, self._task_id, episode_idx=episode_idx)]
        return []

    # ---- 直接推送（不依赖 step hook 机制） ----

    def push_frame(
        self,
        task_id: str,
        raw_obs,
        valid_cmds,
        current_time: float,
        data_type: int = 0,
        episode_idx: int = 0,
        step_idx: int = 0,
    ):
        """直接推送一帧态势数据，不依赖 RolloutWorker 的 step_hook 机制。"""
        self._publisher.push_frame(
            task_id=task_id,
            raw_obs=raw_obs,
            valid_cmds=valid_cmds,
            current_time=current_time,
            data_type=data_type,
            episode_idx=episode_idx,
            step_idx=step_idx,
        )

    # ---- 资源管理 ----

    def close(self):
        """释放 ZMQ 资源。"""
        self._publisher.close()
