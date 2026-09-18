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

# ---- 可变实体数泛化：MAX 固定槽位（满载维度，向下兼容低实体数场景） ----
# 网络权重形状恒按这些 MAX 值构建，实体数不足时用 padding 占满槽位 + mask 屏蔽 dummy。
# 槽位布局（agent 维）：[0, MAX_RADARS) 雷达 LD ｜ [MAX_RADARS, MAX_AGENTS) 卫星 WX
MAX_RADARS = 200                          # 雷达固定槽位数
MAX_SATELLITES = 83                       # 卫星固定槽位数（由 50 扩充至 83）
MAX_TARGETS = 21                          # 目标固定槽位数
MAX_AGENTS = MAX_RADARS + MAX_SATELLITES  # 283 = 200 + 83
MAX_ACTIONS = MAX_TARGETS + 1             # 22（0=待机，1..21=锁定目标）


def build_agent_slot_map(agent_keys, satellite_keys=None):
    """按固定槽位布局构造「实体 ID → MAX agent 槽位」映射。

    可变实体数泛化的槽位真相源：雷达 → [0, MAX_RADARS)，卫星 → [MAX_RADARS, MAX_AGENTS)。
    ObservationBuilder / ActionMapper / RewardCalculator / expert_data 均由此函数生成映射，
    避免多处手写漂移。

    未传 satellite_keys 时全部按雷达槽位映射（此时 dense 顺序 == 槽位顺序，满载场景
    两种映射逐位一致，向下兼容未显式区分卫星的旧调用方）。
    """
    satellite_set = set(satellite_keys or [])
    slot = {}
    _radar_slot = 0
    _sat_slot = MAX_RADARS
    for sid in agent_keys:
        if sid in satellite_set:
            slot[sid] = _sat_slot
            _sat_slot += 1
        else:
            slot[sid] = _radar_slot
            _radar_slot += 1
    return slot


def build_agent_count_mask(agent_keys, satellite_keys=None):
    """构建 agent 数量掩码：(MAX_AGENTS,) 列表，真实实体槽位=1.0，dummy（padding）槽位=0.0。

    可变实体数泛化的「实体掩码」真相源：与 build_agent_slot_map 共享同一槽位布局，
    直接对槽位映射的值置 1，保证掩码与 obs / action / reward 的槽位逐位一致。

    用于 IL/RL 训练把 dummy agent 的损失贡献归零——dummy agent 的 obs=0、动作为待机，
    但 DRQN 有偏置与 one-hot 身份权重，仍会产出垃圾 logits / Q，混进损失会稀释真实梯度。
    """
    slot = build_agent_slot_map(agent_keys, satellite_keys)
    mask = [0.0] * MAX_AGENTS
    for s in slot.values():
        mask[s] = 1.0
    return mask
