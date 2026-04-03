# -*- coding: utf-8 -*-
# 新的20260320版本
from dataclasses import dataclass, field
from typing import List, Dict
from .two_dim_coordinate import TwoDimensionMinMax


# 目标轨迹点信息
@dataclass
class TargetTrajPtInfo:
    time: int = 0                   # 时间，相对于任务开始时间的相对时，单位：s
    longitude: float = 0.0          # 大地位置经度，单位：deg
    latitude: float = 0.0           # 大地位置纬度，单位：deg
    altitude: float = 0.0           # 大地位置高度，单位：m
    ecf_x: float = 0.0              # 地心地固坐标x，单位：km
    ecf_y: float = 0.0              # 地心地固坐标y，单位：km
    ecf_z: float = 0.0              # 地心地固坐标z，单位：km
    ecf_vx: float = 0.0             # 地心地固速度vx，单位：km/s
    ecf_vy: float = 0.0             # 地心地固速度vy，单位：km/s
    ecf_vz: float = 0.0             # 地心地固速度vz，单位：km/s
    rcs: float = 1.0                # RCS值，单位：m^2
    type: int = 0                   # 目标类型，1-导弹主目标，2-弹头，3-诱饵，4-碎片，5-空间目标，6-空中目标，0-其他目标


# 目标信息
@dataclass
class TargetInfo:
    str_target_id: str = ''                                                                 # 目标编号
    str_target_name: str = ''                                                               # 目标名称
    dict_target_traj_pt_info: Dict[int, TargetTrajPtInfo] = field(default_factory=dict)     # 目标轨迹点信息，[时间，轨迹点信息]


# 传感器信息
@dataclass
class SensorInfo:
    str_sensor_id: str = ''         # 传感器站号
    str_sensor_name: str = ''       # 传感器名称
    longitude: float = 0.0          # 大地位置经度，单位：deg
    latitude: float = 0.0           # 大地位置纬度，单位：deg
    altitude: float = 0.0           # 大地位置高度，单位：m
    range_min: float = 0.0          # 探测距离近界，单位：km
    range_max: float = 4000.0       # 探测距离远界，单位：km
    azi_min: float = 0.0            # 探测方位左界，单位：deg
    azi_max: float = 360.0          # 探测方位右界，单位：deg
    ele_min: float = 0.0            # 探测仰角下界，单位：deg
    ele_max: float = 90.0           # 探测仰角上界，单位：deg
    track_num_max: int = 1          # 最大跟踪容量


# 卫星信息
@dataclass
class SatelliteInfo:
    str_satellite_id: str = ''                                                                  # 卫星编号
    str_satellite_name: str = ''                                                                # 卫星名称
    dict_satellite_traj_pt_info: Dict[int, TargetTrajPtInfo] = field(default_factory=dict)      # 卫星轨迹点信息，[时间，轨迹点信息]
    azi_min: float = -10.0                                                                      # 探测方位左界，单位：deg
    azi_max: float = 10.0                                                                       # 探测方位右界，单位：deg
    ele_min: float = -10.0                                                                      # 探测仰角下界，单位：deg
    ele_max: float = 10.0                                                                       # 探测仰角上界，单位：deg
    track_num_max: int = 1                                                                      # 最大跟踪容量


# 作战场景
@dataclass
class BattleScene:
    start_time: int = 0                                                                 # 开始时间，相对于1970年1月1日的毫秒数
    end_time: int = 0                                                                   # 结束时间，相对于1970年1月1日的毫秒数
    dict_radar_id_info: Dict[str, SensorInfo] = field(default_factory=dict)             # 雷达信息，[雷达站号，雷达信息]
    dict_satellite_id_info: Dict[str, SatelliteInfo] = field(default_factory=dict)      # 卫星信息，[卫星站号，卫星信息]
    dict_target_id_info: Dict[str, TargetInfo] = field(default_factory=dict)            # 目标信息，[目标编号，目标名称，[时间，轨目标迹点信息]]


# 装备对目标的探测时间
@dataclass
class EquipmentDetectionTime:
    str_equip_id: str = ''                                                              # 装备站号
    str_target_id: str = ''                                                             # 目标编号
    time_range: TwoDimensionMinMax = field(default_factory=TwoDimensionMinMax)          # 探测起止时间，相对于任务开始时间的相对时，单位：s


# 规划结果
@dataclass
class PlanResult:
    dict_equip_id_target_id_detection_time: Dict[str, Dict[str, List[EquipmentDetectionTime]]] = field(default_factory=dict)    # 装备对目标的探测时序


# 预案文件信息
@dataclass
class PlanFileInfo:
    battle_scene: BattleScene = field(default_factory=BattleScene)          # 作战场景
    plan_result: PlanResult = field(default_factory=PlanResult)             # 规划结果


# 系统航迹
@dataclass
class SystemTrackBase:
    detect_time: int = 0                # 探测时间，单位：s
    str_system_track_no: str = ''       # 系统航迹批号
    longitude: float = 0.0              # 大地位置经度，单位：deg
    latitude: float = 0.0               # 大地位置纬度，单位：deg
    altitude: float = 0.0               # 大地位置高度，单位：m
    ecf_x: float = 0.0                  # 地心地固坐标x，单位：km
    ecf_y: float = 0.0                  # 地心地固坐标y，单位：km
    ecf_z: float = 0.0                  # 地心地固坐标z，单位：km
    ecf_vx: float = 0.0                 # 地心地固速度vx，单位：km/s
    ecf_vy: float = 0.0                 # 地心地固速度vy，单位：km/s
    ecf_vz: float = 0.0                 # 地心地固速度vz，单位：km/s
    rcs: float = 1.0                    # RCS值，单位：m^2
    type: int = 0                       # 目标类型，1-导弹主目标，2-弹头，3-诱饵，4-碎片，5-空间目标，6-空中目标，0-其他目标


# 装备状态
@dataclass
class EquipmentState:
    time: int = 0                                                       # 时间，单位：s
    str_equip_id: str = ''                                              # 装备站号
    longitude: float = 0.0                                              # 大地位置经度，单位：deg
    latitude: float = 0.0                                               # 大地位置纬度，单位：deg
    altitude: float = 0.0                                               # 大地位置高度，单位：m
    range_min: float = 0.0                                              # 探测距离近界，单位：km
    range_max: float = 4000.0                                           # 探测距离远界，单位：km
    azi_min: float = 0.0                                                # 探测方位左界，单位：deg
    azi_max: float = 360.0                                              # 探测方位右界，单位：deg
    ele_min: float = 0.0                                                # 探测仰角下界，单位：deg
    ele_max: float = 90.0                                               # 探测仰角上界，单位：deg
    azi_pointing: float = 0.0                                           # 方位指向，单位：deg
    ele_pointing: float = 0.0                                           # 俯仰指向，单位：deg
    type: int = 0                                                       # 装备类型，1-雷达，2-卫星，0-其他
    lst_track_no: List[str] = field(default_factory=list)               # 跟踪目标批号列表
    track_num_max: int = 1                                              # 最大跟踪容量
    residual_track_num: int = 1                                         # 剩余跟踪容量


# 装备对目标的可探测性结果
@dataclass
class EquipmentToTargetDetectionResult:
    str_equip_id: str = ''                      # 装备站号
    str_target_id: str = ''                     # 目标编号
    detectable_flag: bool = False               # 可探测性标识


# 智能体观测信息
@dataclass
class AgentObservation:
    dict_system_track: Dict[str, SystemTrackBase] = field(default_factory=dict)                                         # 系统航迹
    dict_equip_state: Dict[str, EquipmentState] = field(default_factory=dict)                                           # 装备状态
    dict_detection_result: Dict[str, Dict[str, EquipmentToTargetDetectionResult]] = field(default_factory=dict)         # 装备对目标的可探测性结果，[装备站号，[目标编号，可探测性结果]]
    current_time: int = 0                                                                                               # 场景当前时间，单位：s


# 智能体动作指令
@dataclass
class AgentActionCommand:
    time: int = 0                       # 时间，相对于任务开始时间的相对时，单位：s
    str_equip_id: str = ''              # 装备站号
    str_target_id: str = ''             # 探测目标编号
