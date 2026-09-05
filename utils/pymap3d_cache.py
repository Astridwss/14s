"""pymap3d 椭球常量缓存。

pymap3d 的 ``geodetic2aer`` / ``geodetic2ecef`` 在 ``ell`` 未显式传入时，每次都会
重建 WGS84 的 ``Ellipsoid`` 常量（``Ellipsoid.from_name`` 内含 ``sqrt`` 求偏心率）。
本平台雷达/卫星/目标全部使用 WGS84，且 ``Ellipsoid`` 是不可变纯数据，故这里把
``Ellipsoid.from_name`` 的结果 memoize，消除每次几何换算时的重复构造。

仅省去重复的纯计算，不改变任何数值结果；模块 import 即生效（monkey-patch pymap3d，
不动 sim/ 内核）。
"""
import pymap3d as pm

_ORIG_FROM_NAME = pm.Ellipsoid.from_name
_ELL_CACHE: dict = {}


def _cached_from_name(name: str):
    ell = _ELL_CACHE.get(name)
    if ell is None:
        ell = _ORIG_FROM_NAME(name)
        _ELL_CACHE[name] = ell
    return ell


pm.Ellipsoid.from_name = _cached_from_name
