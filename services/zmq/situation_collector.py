"""态势采集 —— 从 step 观测提取输入、组装成帧，攒成一整局轨迹。

采集阶段不碰网络 I/O：SituationCollector 每步把序列化帧 append 进 frames，
一局结束由 runner 打包成 SituationTrajectory 入池（见 situation_pool.py），
由发送线程统一推送。帧组装/输入提取是纯函数，串行(step_hook)/并行(collector)共用。
"""

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
    satellite_fov=None,
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

    # 3b. 卫星视场覆盖（仅绘制补充：视场内非锁定目标，复用 Detection 画覆盖连线，不参与训练）
    if satellite_fov:
        for sat_id, tgt_id in satellite_fov:
            if tgt_id and tgt_id not in ("", "0"):
                det = msg.Detection.add()
                det.EquipID = _safe_extract_id(sat_id)
                det.TargetID = _safe_extract_id(tgt_id)

    return msg.SerializeToString()


def _extract_frame_inputs(step_count, terminated, truncated, next_info):
    """从 next_info 提取一帧态势所需输入，raw_obs 缺失时返回 None。

    返回 (raw_obs, valid_cmds, action_time, data_type, satellite_fov)。
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
    satellite_fov = next_info.get('satellite_fov') or []

    if step_count == 0:
        data_type = 1   # 第一帧
    elif terminated or truncated:
        data_type = 2   # 最后一帧
    else:
        data_type = 0   # 中间过程帧

    return raw_obs, valid_cmds, action_time, data_type, satellite_fov


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
        raw_obs, valid_cmds, action_time, data_type, satellite_fov = inputs

        # 组装 proto 消息：将当前观测、动作、时间等信息打包成 Protobuf 消息
        payload = build_situation_frame(
            current_time=action_time, data_type=data_type,
            episode_idx=self.episode_idx, step_idx=step_count,
            raw_obs=raw_obs, valid_cmds=valid_cmds,
            satellite_fov=satellite_fov,
        )
        self.frames.append(payload)  # 将 proto 消息添加到 frames 列表中
