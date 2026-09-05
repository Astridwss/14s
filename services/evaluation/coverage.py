"""覆盖率解析 —— 从已落盘的评估指标文件反解覆盖率。

推理侧与基线侧共用同一把尺：都读 ``PlanFileProcess.write_plan_metric_to_json``
产出的 ``*_metric.json``，而不是各自用 GanttChart 重算。这样口径与 sim 内核
100% 一致，杜绝两套实现随时间漂移。

指标文件结构（sim/plan_file_process.py:271）::

    {"schemeEvaluteResult": "<JSON 字符串>"}
      └─ {"schemeEvaResult": [
             {"vecTargetEvaResult": [
                 {"strTargetID": "7016",
                  "vecContinuityResult":  [{"dDetectCoverAge": 63.4, "uiInterruputNum": 2}],
                  "vecReliabilityResult": [{"vecCovernumCoverage": [...]}]}
             ]}
         ]}

⚠️ 分母陷阱：``write_plan_metric_to_json`` 只遍历规划结果里**出现过**的目标。
模型没调度到的目标根本不进指标文件，若按文件内目标数求平均会「美化」模型。
故本模块要求传入场景全量目标集合 ``all_target_ids``，缺席目标按 0% 计入。
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional


@dataclass
class CoverageSummary:
    """方案级覆盖率汇总。"""

    coverage: float = 0.0                                   # 全量目标算术平均覆盖率，%
    n_targets_total: int = 0                                # 分母：场景全量目标数
    n_targets_scored: int = 0                               # 指标文件中实际出现的目标数
    per_target: Dict[str, float] = field(default_factory=dict)      # 每目标覆盖率，%
    interrupts: Dict[str, int] = field(default_factory=dict)        # 每目标中断次数

    @property
    def n_targets_missing(self) -> int:
        """场景中存在但未被任何装备调度到的目标数（按 0% 计入平均）。"""
        return max(0, self.n_targets_total - self.n_targets_scored)

    def describe(self) -> str:
        """一行可读摘要，供日志打印。"""
        return (f"coverage={self.coverage:.2f}% "
                f"(全量目标 {self.n_targets_total}，已调度 {self.n_targets_scored}，"
                f"缺席按0%计 {self.n_targets_missing})")


def parse_metric_file(
    metric_path: str,
    all_target_ids: Optional[Iterable[str]] = None,
) -> CoverageSummary:
    """解析评估指标文件，返回方案级覆盖率汇总。

    Args:
        metric_path: ``*_metric.json`` 路径（推理侧或基线侧均可）。
        all_target_ids: 场景全量目标 ID，作为平均的分母。传 None 时退化为
            「按文件内目标求平均」，仅用于无场景上下文的调试，**不可用于
            推理/基线对比**。

    Returns:
        CoverageSummary；文件缺失或结构异常时返回全 0 的空汇总（不抛异常，
        由调用方按 ``n_targets_scored == 0`` 判断是否可用）。
    """
    per_target, interrupts = _read_per_target(metric_path)

    if all_target_ids is None:
        denominator = list(per_target.keys())
    else:
        denominator = [str(t) for t in all_target_ids]

    n_total = len(denominator)
    if n_total == 0:
        return CoverageSummary(
            per_target=per_target, interrupts=interrupts,
            n_targets_scored=len(per_target),
        )

    # 全量目标算术平均：分母固定为场景目标数，缺席目标按 0% 计入
    total = sum(per_target.get(tid, 0.0) for tid in denominator)

    return CoverageSummary(
        coverage=round(total / n_total, 4),
        n_targets_total=n_total,
        n_targets_scored=sum(1 for tid in denominator if tid in per_target),
        per_target=per_target,
        interrupts=interrupts,
    )


# ============================================================
# 内部
# ============================================================

def _read_per_target(metric_path: str):
    """从指标文件抽取 {目标ID: 覆盖率%} 与 {目标ID: 中断次数}。"""
    if not metric_path or not os.path.exists(metric_path):
        print(f"[Coverage] 指标文件不存在，覆盖率按空处理: {metric_path}")
        return {}, {}

    try:
        with open(metric_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"[Coverage] 指标文件读取失败: {metric_path} - {e}")
        return {}, {}

    scheme = data.get("schemeEvaluteResult") if isinstance(data, dict) else None
    # sim 落盘时把内层 dump 成字符串，这里兼容 str / dict 两种形态
    if isinstance(scheme, str):
        try:
            scheme = json.loads(scheme)
        except ValueError as e:
            print(f"[Coverage] schemeEvaluteResult 解析失败: {e}")
            return {}, {}
    if not isinstance(scheme, dict):
        print(f"[Coverage] 指标文件缺少 schemeEvaluteResult: {metric_path}")
        return {}, {}

    per_target: Dict[str, float] = {}
    interrupts: Dict[str, int] = {}

    for eva_result in scheme.get("schemeEvaResult") or []:
        for target_eva in (eva_result or {}).get("vecTargetEvaResult") or []:
            target_id = str((target_eva or {}).get("strTargetID", "")).strip()
            if not target_id:
                continue

            continuity = target_eva.get("vecContinuityResult") or []
            if not continuity:
                continue

            record = continuity[0] or {}
            try:
                per_target[target_id] = float(record.get("dDetectCoverAge", 0.0))
                interrupts[target_id] = int(record.get("uiInterruputNum", 0))
            except (TypeError, ValueError):
                print(f"[Coverage] 目标 {target_id} 覆盖率字段异常，按 0 处理")
                per_target[target_id] = 0.0
                interrupts[target_id] = 0

    return per_target, interrupts
