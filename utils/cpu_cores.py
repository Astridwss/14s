"""
运行时 CPU 核数探测 —— 供并行数 clamp 使用。

并行 rollout 是纯 Python CPU-bound（sim 的 geodetic2aer 双循环），有效上限 = 部署机器
的物理核数（超线程只能再挤出 ~1.2×，且大小核异构机上 E 核会拖慢 pool.map 同步屏障的
整轮速度）。这里在启动时读一次机器核数，把请求的 n_workers 夹到 [1, 物理核数]，避免在
核数少的部署机上超订（超订会因上下文切换反而变慢）。

不依赖 psutil 时退化为 os.cpu_count()（逻辑核数）。
"""
import os


def logical_cpu_count() -> int:
    """逻辑核数（含超线程）。"""
    return int(os.cpu_count() or 1)


def physical_cpu_count() -> int:
    """物理核数。

    优先 psutil；没有则 Windows 下用 GetLogicalProcessorInformation 数
    RelationProcessorCore 条目（零额外依赖）；再不行退化为逻辑核数。
    """
    try:
        import psutil  # noqa: WPS433  可选依赖，缺失即跳过
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return _windows_physical_cores() or logical_cpu_count()


def _windows_physical_cores() -> int:
    """Windows 下读物理核数（无 psutil 依赖）。非 Windows 返回 0。"""
    import ctypes
    import sys

    if sys.platform != "win32":
        return 0

    class SLPI(ctypes.Structure):
        # SYSTEM_LOGICAL_PROCESSOR_INFORMATION：ProcessorMask + Relationship + 联合体占位。
        # 联合体含 ULONGLONG[2]，占 16 字节；只读前两个字段，联合体用 16 字节占位保证步长正确。
        _fields_ = [
            ("ProcessorMask", ctypes.c_size_t),
            ("Relationship", ctypes.c_uint),
            ("_reserved", ctypes.c_ubyte * 16),
        ]

    length = ctypes.c_ulong(0)
    kernel32 = ctypes.windll.kernel32
    # 第一次调用传 NULL buffer：返回 FALSE 且 GetLastError=ERROR_INSUFFICIENT_BUFFER，
    # 但会把所需长度写入 length —— 这是预期行为，不能据此 return。
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
    if length.value == 0:
        return 0
    n = length.value // ctypes.sizeof(SLPI)
    buf = (SLPI * n)()
    if not kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
        return 0
    return sum(1 for it in buf if it.Relationship == 0)  # 0 = RelationProcessorCore


def clamp_workers(requested) -> int:
    """把请求的并行数夹到 [1, 物理核数]。<=0 返回 0（表示不启用并行）。"""
    try:
        n = int(requested or 0)
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    cap = physical_cpu_count()
    if n > cap:
        print(f"[cpu_cores] 请求并行数 {n} 超过本机物理核数 {cap}，"
              f"已 clamp 到 {cap}（超订会因上下文切换变慢）")
        return cap
    return n
