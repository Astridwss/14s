# 文件路径: communication/zmq_publisher.py
import zmq
import traceback
from communication import ProtoStruct_pb2

class SituationPublisher:
    """
    独立的 ZMQ 通信层：负责封装 Proto 协议并向前端推送态势数据
    """
    def __init__(self, port: int = 5556):
        self.zms_server_ip = "192.168.1.51"
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        
        connect_addr = f"tcp://{self.zms_server_ip}:{self.port}"
        self.socket.connect(connect_addr)
        print(f"[ZMQ Publisher] 态势推送服务已启动，已连接到消息总线: {connect_addr}")

    def _safe_extract_id(self, id_str: str) -> int:
        """安全地从字符串中提取纯数字ID，适应 '1' 或 'R01' 等格式"""
        digits = ''.join(filter(str.isdigit, str(id_str)))
        return int(digits) if digits else 0

    def push_frame(self, task_id: str, raw_obs, valid_cmds, current_time: float, data_type: int = 0):
        """
        组装 Proto 并推送一帧态势数据
        :param task_id: 任务ID (用于 ZMQ Topic 路由)
        :param raw_obs: AgentObservation 实例
        :param valid_cmds: List[AgentActionCommand]
        :param current_time: float
        :param data_type: 1-第一帧, 2-最后一帧, 0-中间过程帧
        """
        try:
            msg = ProtoStruct_pb2.PR_I_RL_TRAINING_SITUATION_TO_FRONT()
            msg.CurrentTime = float(current_time)
            msg.DataType = int(data_type)

            msg.PubCtrl.MsgHeader.MsgType = 12290
            
            # 1. 组装红方装备位置 (解析 dict_equip_state -> EquipmentState)
            if hasattr(raw_obs, 'dict_equip_state') and raw_obs.dict_equip_state:
                for eid, equip_state in raw_obs.dict_equip_state.items():
                    equip = msg.EquipPos.add()
                    equip.EquipID = self._safe_extract_id(equip_state.str_equip_id)
                    equip.EquipType = equip_state.type  # 1-雷达, 2-卫星
                    equip.Time = msg.CurrentTime
                    equip.GeoPos.X = float(equip_state.longitude)
                    equip.GeoPos.Y = float(equip_state.latitude)
                    equip.GeoPos.Z = float(equip_state.altitude)
                    
            # 2. 组装蓝方目标位置 (解析 dict_system_track -> SystemTrackBase)
            if hasattr(raw_obs, 'dict_system_track') and raw_obs.dict_system_track:
                for tid, track_base in raw_obs.dict_system_track.items():
                    t_pos = msg.TargetPos.add()
                    t_pos.TargetID = self._safe_extract_id(track_base.str_system_track_no)
                    t_pos.Time = msg.CurrentTime
                    t_pos.GeoPos.X = float(track_base.longitude)
                    t_pos.GeoPos.Y = float(track_base.latitude)
                    t_pos.GeoPos.Z = float(track_base.altitude)
                    
            # 3. 组装探测/锁定关系 (解析 valid_cmds -> AgentActionCommand)
            if valid_cmds:
                for cmd in valid_cmds:
                    if cmd.str_target_id and cmd.str_target_id != "0" and cmd.str_target_id != "":
                        det = msg.Detection.add()
                        det.EquipID = self._safe_extract_id(cmd.str_equip_id)
                        det.TargetID = self._safe_extract_id(cmd.str_target_id)

            # 4. 序列化
            payload = msg.SerializeToString()
            msg_type_str = str(12290).encode('utf-8')

            # 5. 发送消息
            self.socket.send_multipart([msg_type_str, payload])
            
        except Exception as e:
            print(f"[ZMQ Publisher] 发送态势崩溃: {traceback.format_exc()}")

    def close(self):
        """安全关闭 Socket"""
        self.socket.close()
        self.context.term()


class ZMQStepHook:
    """
    专门用于拦截 Rollout 步进事件的 ZMQ 回调钩子。
    彻底将数据清洗与通信逻辑从核心算法层抽离。
    """
    def __init__(self, publisher: SituationPublisher, task_id: str):
        self.publisher = publisher
        self.task_id = task_id

    def __call__(self, step_count: int, terminated: bool, truncated: bool, next_info: dict):
        """
        当环境执行完 step 后，触发此调用
        """
        raw_obs = next_info.get('raw_obs')
        if raw_obs is None:
            print(f"[ZMQ Debug] 第 {step_count} 步没有拿到 raw_obs，跳过推送")
            return
        print(f"[ZMQ Debug] 正在推送第 {step_count} 步的态势数据...")
        # 1. 业务数据清洗：提取并过滤有效动作
        all_cmds = next_info.get('agent_actions_list', [])
        valid_cmds = [cmd for cmd in all_cmds if getattr(cmd, 'str_target_id', "") not in ["", "0"]]
        action_time = float(next_info.get('action_time', 0.0))

        # 2. 帧类型逻辑判断
        if step_count == 0:
            data_type = 1  # 第一帧
        elif terminated or truncated:
            data_type = 2  # 最后一帧
        else:
            data_type = 0  # 中间过程帧

        # 3. 触发真实的网络推送
        self.publisher.push_frame(
            task_id=self.task_id,
            raw_obs=raw_obs,
            valid_cmds=valid_cmds,
            current_time=action_time,
            data_type=data_type
        )