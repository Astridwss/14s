"""
权重文件语义化命名 —— 实体数 + 分组数。

命名示例:
    LD100_WX25_TARGET21_groupsize10_drqn.pkl
    LD100_WX25_TARGET21_groupsize10_qmix.pkl
    LD100_WX25_TARGET21_groupsize10_lower_mixer.pkl
    LD100_WX25_TARGET21_groupsize10_upper_mixer.pkl

字段含义:
    LD       = 雷达 (radar) 数量
    WX       = 卫星 (satellite) 数量
    TARGET   = 目标 (target) 数量
    groupsize = H-QMIX 每组雷达数 (标准 QMIX 时为 0)

统一在此处生成文件名，保存与加载两侧共用同一规则，避免命名漂移。
"""


def build_weight_prefix(n_radars, n_satellites, n_targets, group_size):
    """生成权重文件名前缀，如 LD100_WX25_TARGET21_groupsize10。"""
    return (
        f"LD{int(n_radars)}_WX{int(n_satellites)}"
        f"_TARGET{int(n_targets)}_groupsize{int(group_size)}"
    )


def prefix_from_conf(conf):
    """从配置对象提取实体数 / 分组数，生成前缀。

    conf 为扁平 RuntimeConfig 或聚焦 AlgorithmConfig，字段缺失时回退 0。
    """
    return build_weight_prefix(
        getattr(conf, "n_radars", 0) or 0,
        getattr(conf, "n_satellites", 0) or 0,
        getattr(conf, "n_targets", 0) or 0,
        getattr(conf, "group_size", 0) or 0,
    )


def weight_file_name(prefix, role):
    """由前缀 + 角色生成完整文件名。

    role ∈ {'drqn', 'qmix', 'lower_mixer', 'upper_mixer'}
    """
    return f"{prefix}_{role}.pkl"
