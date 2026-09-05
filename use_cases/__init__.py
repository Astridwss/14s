"""use_cases — 用例层（配置装配、推送通道、Runner）。

顶层采用 PEP 562 惰性导入：__init__ 不再急加载 runners / config / pusher，
避免与 services.scene 形成循环 import（services.scene.env_wrapper 依赖
use_cases.config.config_types，而 use_cases.runners.rl_train_runner 又依赖
services.scene.env_wrapper）。对外 ``from use_cases import X`` 仍可用，首次
访问时才真正加载对应子模块。
"""

__all__ = [
    "ConfigAssembler", "RuntimeConfig",
    "Pusher",
    "BaseRunner", "RLTrainRunner", "ILTrainRunner", "ILInferRunner", "EvalRunner",
]

# 惰性导入映射：name -> (module, attr)
_LAZY_IMPORTS = {
    "ConfigAssembler": ("use_cases.config", "ConfigAssembler"),
    "RuntimeConfig": ("use_cases.config", "RuntimeConfig"),
    "Pusher": ("use_cases.pusher", "Pusher"),
    "BaseRunner": ("use_cases.runners", "BaseRunner"),
    "RLTrainRunner": ("use_cases.runners", "RLTrainRunner"),
    "ILTrainRunner": ("use_cases.runners", "ILTrainRunner"),
    "ILInferRunner": ("use_cases.runners", "ILInferRunner"),
    "EvalRunner": ("use_cases.runners", "EvalRunner"),
}


def __getattr__(name: str):
    """惰性导入：首次访问 __all__ 中的名字时才加载对应子模块。"""
    import importlib

    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value  # 缓存，避免重复 import
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
