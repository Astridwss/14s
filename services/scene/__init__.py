"""services.scene — 场景解析积木（环境包装、动作映射、观测构建、奖励）。

顶层采用 PEP 562 惰性导入：__init__ 不再急加载 env_wrapper / action / state，
避免与 use_cases 形成循环 import（env_wrapper 依赖 use_cases.config.config_types，
而 use_cases.runners 又依赖 env_wrapper）。对外 ``from services.scene import X``
仍可用，首次访问时才真正加载对应子模块。
"""

__all__ = ["GroupedEnvWrapper", "ActionMapper", "ObservationBuilder"]

# 惰性导入映射：name -> (module, attr)
_LAZY_IMPORTS = {
    "GroupedEnvWrapper": ("services.scene.env_wrapper", "GroupedEnvWrapper"),
    "ActionMapper": ("services.scene.action.mapper", "ActionMapper"),
    "ObservationBuilder": ("services.scene.state.builder", "ObservationBuilder"),
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
