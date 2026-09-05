"""torch 后端注册 —— 统一处理昇腾 NPU 与 CUDA/CPU 的 device 解析与显存释放。

背景:
    PyTorch 默认不识别 "npu" 设备。昇腾 910B 必须先 ``import torch_npu`` 注册后端，
    之后 ``torch.device("npu")`` 与 ``.to("npu")`` 才可用。主训练进程此前从未 import
    torch_npu，导致 device=npu 时 ``torch.device("npu")`` 抛异常被吞、回退成字符串，
    最终在 ``.to("npu")`` 处崩溃。本模块统一收口这一注册逻辑。
"""


def resolve_device(device_str):
    """把 device 字符串转成已注册后端的 torch.device（torch/torch_npu 不可用时回退原值）。

    - "npu" / "npu:0": 先 ``import torch_npu`` 注册 npu 后端，再 ``torch.device(...)``；
    - "cpu" / "cuda" / ...: 直接 ``torch.device(...)``；
    - torch 本身不可用：保持原字符串，交由调用方后续报出人话错误。

    Args:
        device_str: "cpu" | "cuda" | "npu" | "npu:3" | torch.device ...
    Returns:
        torch.device（后端已注册）；不可用时返回原字符串。
    """
    try:
        import torch
    except ImportError:
        return device_str

    if isinstance(device_str, str) and device_str.split(":")[0] == "npu":
        try:
            import torch_npu  # noqa: F401  注册 "npu" 后端 + torch.npu 命名空间
        except ImportError:
            # 本机无 torch_npu：保持字符串，让后续 .to("npu") 抛出具象错误
            return device_str

    try:
        return torch.device(device_str)
    except Exception:
        return device_str


def empty_device_cache() -> None:
    """释放 PyTorch 缓存的显存（CUDA 与 NPU 双后端），幂等、不抛异常。"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        import torch_npu
        torch_npu.npu.empty_cache()
    except Exception:
        pass
