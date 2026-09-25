"""评价指标体系 —— 四个指标 × 权重 → 加权总分。

四个指标各封装成一个方法，均返回 0-100 分：
  1. 目标覆盖数   target_cover_count     权重 10%（已落地）
  2. 中断次数     interrupt_count        权重 10%（已落地）
  3. 覆盖重数     coverage_multiplicity  权重 60%（已落地）
  4. 跟踪覆盖率   tracking_coverage      权重 20%（已落地）

总分 = Σ(指标分 × 权重)。指标分全部从 sim 落盘的 ``*_metric.json``
（``services.evaluation.metric_writer`` 重写版）反解而来，推理侧与基线侧
共用同一把尺，与 ``services.evaluation.coverage`` 同源——不做第二次甘特图重算，
避免两套实现随时间漂移。

用法::

    from services.evaluation import SchemeMetrics, scene_target_ids
    metrics = SchemeMetrics(metric_path, all_target_ids=scene_target_ids(scene))
    print(metrics.target_cover_count())   # 目标覆盖数，0-100
"""

from typing import Dict, Iterable, Optional

from services.evaluation.coverage import parse_metric_file


class SchemeMetrics:
    """方案级评价指标体系。

    构造时解析一次 ``*_metric.json``（复用 coverage.parse_metric_file），
    四个指标方法各自从解析结果派生一个 0-100 分。
    """

    # 各指标权重（占比，四项之和 = 1.0）：目标覆盖数 10% / 中断次数 10% /
    # 覆盖重数 60% / 跟踪覆盖率 20%。可在实例/子类上覆盖用于标定。
    WEIGHTS: Dict[str, Optional[float]] = {
        "target_cover_count": 0.10,       # 目标覆盖数量
        "interrupt_count": 0.10,          # 中断次数
        "coverage_multiplicity": 0.60,    # 覆盖重数
        "tracking_coverage": 0.20,        # 跟踪覆盖率
    }

    def __init__(self, metric_path: str,
                 all_target_ids: Optional[Iterable[str]] = None):
        # 复用 coverage 的解析：per_target={目标ID:覆盖率%}，n_targets_total=场景全量目标数
        self._summary = parse_metric_file(metric_path, all_target_ids)
        # 分母目标列表（与 parse_metric_file 同口径）：all_target_ids 给定则用之，
        # 否则退化为指标文件内出现过的目标（仅调试，勿用于推理/基线对比）。
        if all_target_ids is None:
            self._all_target_ids = list(self._summary.per_target.keys())
        else:
            self._all_target_ids = [str(t) for t in all_target_ids]

    # ============================================================
    # 指标 1：目标覆盖数（权重 10%）
    # ============================================================

    def target_cover_count(self) -> float:
        """目标覆盖数（权重 10%，0-100）。

        定义：目标 t 的自身轨迹时长内，只要被动作选择过就得满分，否则 0 分。
        分子 = Σ_t [t 被选择 ? 100 : 0]，分母 = 场景目标数（标准满载 21），
        总分 = 100 × 被选择目标数 / 目标数。

        「被选择」判定 = 该目标在自身轨迹窗口内被 ≥1 部装备覆盖过任意正时长，
        即 metric 文件里 ``dDetectCoverAge > 0``（该值已由 metric_writer 按目标
        轨迹窗口裁剪过探测弧段，越界伪探测不计）。未出现在指标文件中的目标
        （从未被任何装备调度）按「未被选择」计 0，仍占分母——与
        coverage.parse_metric_file 的「缺席按 0% 计入」口径一致。
        """
        n = len(self._all_target_ids)   # 分母：场景全量目标数
        if n == 0:
            return 0.0
        # 只按分母目标判定：per_target.get(tid, 0.0) > 0 表示该目标被选择过。
        # 缺席目标（未出现在指标文件）按 0 处理；即使指标文件里混入了场景外的
        # 目标 ID（数据异常），也不会被算进分子。
        selected = sum(
            1 for tid in self._all_target_ids
            if self._summary.per_target.get(tid, 0.0) > 0
        )
        return round(selected * 100.0 / n, 4)

    # ============================================================
    # 指标 2：中断次数（权重 10%）
    # ============================================================

    def interrupt_count(self) -> float:
        """中断次数（权重 10%，0-100）。

        定义：逐装备统计——对每个目标，把「跟踪过它的每部装备」的跟踪段空隙数
        （段数 - 1，即该装备中途停-起的次数）求和，得到该目标的总中断次数。
        中断 1 次扣 10 分，上限 10 次（≥10 次得 0 分）：
            目标分 = max(0, 100 - 10 × 中断次数)
        方案分 = Σ 目标分 / 目标数（满载 21）。

        中断次数来自 metric 文件里每目标的 ``uiInterruputNum``（metric_writer
        已按逐装备口径算好，收进 ``CoverageSummary.interrupts``）。从未被任何
        装备跟踪的目标（不在指标文件中）按 0 分计，与其余三指标「缺席按 0」一致。
        """
        n = len(self._all_target_ids)
        if n == 0:
            return 0.0
        total = 0.0
        for tid in self._all_target_ids:
            if tid not in self._summary.interrupts:
                # 从未被任何装备跟踪 → 0 分（与目标覆盖数/覆盖重数/覆盖率口径一致）
                continue
            total += max(0.0, 100.0 - 10.0 * self._summary.interrupts[tid])
        return round(total / n, 4)

    # ============================================================
    # 指标 3：覆盖重数（权重 60%）
    # ============================================================

    def coverage_multiplicity(self) -> float:
        """覆盖重数（权重 60%，0-100）。

        定义：每个目标在其自身轨迹时长内的「平均覆盖重数」（时间加权，精确到
        每个时间片的覆盖装备数），10 重即满分，少一重扣十分：
            目标分 = min(平均覆盖重数, 10) × 10
        方案分 = Σ 目标分 / 目标数（满载 21）。

        平均覆盖重数来自 metric 文件里每目标的 ``dAvgCoverNum``（metric_writer
        已算好精确时间加权平均，非 0/1/2/≥3 桶）。缺席目标按 0 重计入。
        """
        n = len(self._all_target_ids)
        if n == 0:
            return 0.0
        total = sum(
            min(self._summary.avg_cover_num.get(tid, 0.0), 10.0) * 10.0
            for tid in self._all_target_ids
        )
        return round(total / n, 4)

    # ============================================================
    # 指标 4：跟踪覆盖率（权重 20%）
    # ============================================================

    def tracking_coverage(self) -> float:
        """跟踪覆盖率（权重 20%，0-100）。

        定义：即原「跟踪覆盖率」口径——每目标「覆盖秒数 ÷ 自身轨迹时长」的算术
        平均，分母为场景全量目标数，缺席目标按 0% 计入。等价于
        ``CoverageSummary.coverage``（parse_metric_file 已按修正后的分母口径算好，
        探测弧段也已裁剪到目标轨迹窗口，不会破百）。
        """
        return self._summary.coverage

    # ============================================================
    # 加权总分
    # ============================================================

    def weighted_score(self) -> float:
        """最后得分 = Σ(指标分 × 权重)。四个指标公式已全部落地。"""
        total = 0.0
        for name, method in (
            ("target_cover_count", self.target_cover_count),
            ("interrupt_count", self.interrupt_count),
            ("coverage_multiplicity", self.coverage_multiplicity),
            ("tracking_coverage", self.tracking_coverage),
        ):
            weight = self.WEIGHTS.get(name)
            if weight is None:
                raise ValueError(f"指标 {name} 权重未配置，无法计算加权总分")
            total += method() * weight
        return round(total, 4)

    def all_scores(self) -> Dict[str, float]:
        """一次算出四个指标分 + 最后得分，供基线/推理响应统一透出。

        键（snake_case，与指标方法同名）：
            target_cover_count / interrupt_count / coverage_multiplicity /
            tracking_coverage / weighted_score。
        """
        return {
            "target_cover_count": self.target_cover_count(),
            "interrupt_count": self.interrupt_count(),
            "coverage_multiplicity": self.coverage_multiplicity(),
            "tracking_coverage": self.tracking_coverage(),
            "weighted_score": self.weighted_score(),
        }
