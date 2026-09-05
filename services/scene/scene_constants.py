"""
场景归一化常量 —— 所有魔数的唯一来源。

改场景规模时只需要改这个文件。
"""

# ---- 观测维度 ----
RADAR_SELF_FEATURES = 9    # 雷达/卫星自身特征数 [通道, 锁定目标, range×4, azi×2, ele×2, type]
TARGET_FEATURES = 5        # 每个目标的特征数 [可探测, 距离, 方位, 俯仰, 锁定率]
SATELLITE_BROADCAST_FEATURES = 3  # 卫星广播特征/目标 [可见性, 边界距离, 置信度]

# ---- 全局状态维度 ----
RADAR_STATE_FEATURES = 5   # [ecf_x, ecf_y, ecf_z, 负载率, 分配目标]
TARGET_STATE_FEATURES = 8  # [ecf_x, ecf_y, ecf_z, ecf_vx, ecf_vy, ecf_vz, 锁定率, type]

# ---- 归一化基准 ----
MAX_RANGE_KM = 4000.0      # 雷达最大探测距离 (km)
AZI_RANGE = 360.0          # 方位角范围 (度)
ELE_RANGE = 90.0           # 俯仰角范围 (度)
ECF_POS_SCALE = 10000.0    # ECF 坐标归一化尺度
ECF_VEL_SCALE = 10.0       # ECF 速度归一化尺度
EPISODE_DURATION = 600.0   # 单局最大时长 (秒)

# ---- 智能体类型标签 ----
AGENT_TYPE_RADAR = 1.0
AGENT_TYPE_SATELLITE = -1.0

# ---- 容量上限（与 sim 层 track_num_max 对齐，供动作表征层引用） ----
LD_CAPACITY = 20   # 雷达每部最多锁定目标数（多选）
WX_CAPACITY = 1    # 卫星每部最多锁定目标数（单选）
