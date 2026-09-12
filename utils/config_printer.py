"""
配置参数打印工具 —— 打印 RL/IL 训练、RL 推理实际使用的参数及值。

单一入口 print_config_params(conf)，按 conf.mode + conf.algorithm 聚焦打印
env / algo / train / infra 四类子配置，供 handler 装配完成后调用，便于核对
「接口传了什么、实际用了什么」。子配置是 frozen dataclass，直接反射字段打印，
新增/删除字段无需改动本模块。
"""
import dataclasses


def _dump(title: str, obj) -> None:
    """打印单个聚焦配置（dataclass 反射字段，或普通对象按 __dict__）。"""
    print(f"\n[{title}]")
    if obj is None:
        print("  (无)")
        return
    if dataclasses.is_dataclass(obj):
        items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    else:
        items = sorted(vars(obj).items())
    for k, v in items:
        print(f"  {k:<28} = {v!r}")


def print_config_params(conf) -> None:
    """打印 Runner 实际使用的参数与值（RL/IL 训练、RL 推理统一入口）。

    依据 mode（train/eval）与 algorithm（qmix=RL / drqn=IL）判断当前流程，
    并把 env / algo / train / infra 四类聚焦配置全部打印出来。
    """
    mode = str(getattr(conf, "mode", "?"))
    algorithm = str(getattr(conf, "algorithm", "?")).lower()
    task_id = getattr(conf, "task_id", "?")

    print("\n" + "=" * 62)
    print(f"  配置打印 | mode={mode} | algorithm={algorithm} | task_id={task_id}")
    print("=" * 62)

    _dump("EnvConfig 环境配置", getattr(conf, "env", None))
    _dump("AlgorithmConfig 算法配置", getattr(conf, "algo", None))
    _dump("TrainingConfig 训练配置", getattr(conf, "train", None))
    _dump("InfraConfig 基础设施配置", getattr(conf, "infra", None))
    print()
