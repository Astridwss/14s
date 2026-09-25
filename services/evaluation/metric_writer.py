"""评估指标文件写出 —— sim 外重写版（修正覆盖率分母）。

``sim/plan_file_process.py`` 的 ``write_plan_metric_to_json`` 把覆盖率分母写成
``len(dict_target_traj_pt_info)``（轨迹**点数**），而分子 ``over_one_cover_time``
是 ``Σ(value_max - value_min)``（覆盖**秒数**）。两者单位不一致：轨迹点采样间隔
不是 1 秒时（稀疏/密集），点数 ≠ 秒数，导致覆盖率破百或偏低，且对短航迹目标
不友好。

本模块在 sim 外重写同一个函数（sim 是冻结内核，只 import 复用其甘特图类，不改动）。
相对 sim 原实现有三处差异，均为评价口径修正，JSON 顶层结构与字段名保持不变：

1. 覆盖率分母换成**每个目标自身的轨迹时长（首末轨迹点时间差，秒）**，与分子
   同口径；0/1/2/≥3 重那几列 ``vecCovernumCoverage`` 用同一分母，一并修正。
2. ``vecContinuityResult`` 新增 ``dAvgCoverNum``（平均覆盖重数），供指标 3 评分。
3. ``uiInterruputNum``（中断次数）语义改版：sim 原是「目标级覆盖空隙数」（≥1重
   时间片之间的洞），本版按新评价口径改为「逐装备跟踪段空隙数求和」——单部装备
   对目标停-起一次记 1 次中断，再对装备求和（供指标 2 评分）。

其余逻辑（甘特图重叠分析、字段名）与 sim 原实现逐字一致，保证前端读到的
``evalFile`` 结构不变。
"""

import json
from typing import Dict, List, Tuple

from sim.datastruct import BattleScene, EquipmentDetectionTime, PlanResult
from sim.ganttchart import GanttChart, OverlappingNumberRequirement
from sim.two_dim_coordinate import TwoDimensionMinMax


def _target_trajectory_duration(target_id: str, battle_scene: BattleScene) -> int:
    """目标自身轨迹时长（秒）= 末轨迹点时间 - 首轨迹点时间。

    覆盖率分母必须与分子（覆盖秒数）同口径，都用「时间跨度（秒）」。
    空/单点轨迹回退 1s，避免除零；目标不在场景中（异常）回退 1800s 兜底。
    """
    target_info = battle_scene.dict_target_id_info.get(target_id)
    if target_info is None:
        return 1800  # 目标不在场景中（异常），沿用 sim 原兜底时长

    traj_times = sorted(target_info.dict_target_traj_pt_info.keys())
    if len(traj_times) >= 2:
        return max(traj_times[-1] - traj_times[0], 1)
    return 1  # 空/单点轨迹：按 1s 计，避免除零


def _target_trajectory_bounds(target_id: str, battle_scene: BattleScene):
    """目标自身轨迹的时间边界 (min_time, max_time)（秒）。

    探测弧段落在轨迹窗口之外时（卫星视场 index 错位误加、或预案把跟踪窗口
    排到目标飞行窗口之外），分子「覆盖秒数」会超过分母「轨迹时长」导致覆盖率
    破百。用此边界把探测弧段裁剪到目标真实飞行窗口内。
    目标缺失或轨迹点不足 2 个时返回 None（不裁剪，走原逻辑）。
    """
    target_info = battle_scene.dict_target_id_info.get(target_id)
    if target_info is None:
        return None
    traj_times = sorted(target_info.dict_target_traj_pt_info.keys())
    if len(traj_times) < 2:
        return None
    return traj_times[0], traj_times[-1]


def write_plan_metric_to_json(
    plan_result: PlanResult,
    battle_scene: BattleScene,
    dest_path: str,
) -> None:
    """将规划结果写成评估指标 JSON（覆盖率分母按目标自身轨迹时长）。

    与 ``sim/plan_file_process.py:411`` 的 ``write_plan_metric_to_json`` 结构一致，
    仅把分母从「轨迹点数」改为「轨迹时长（秒）」。
    """
    # equip→target→list 反转为 target→equip→list，按目标逐条评估
    dict_target_id_equip_id_detection_time: Dict[str, Dict[str, List[EquipmentDetectionTime]]] = {}

    for str_equip_id, dict_target_id_detection_time in plan_result.dict_equip_id_target_id_detection_time.items():
        for str_target_id, lst_detection_time in dict_target_id_detection_time.items():
            if str_target_id not in dict_target_id_equip_id_detection_time:
                dict_target_id_equip_id_detection_time[str_target_id] = {}

            dict_target_id_equip_id_detection_time[str_target_id][str_equip_id] = lst_detection_time

    data = {}
    data['schemeEvaluteResult'] = {}
    data['schemeEvaluteResult']['schemeEvaResult'] = []

    dict_eva_result = {}
    dict_eva_result['vecTargetEvaResult'] = []

    clipped_total = 0  # 诊断：落在目标轨迹窗口之外、被裁剪/剔除的探测弧段数

    for str_target_id, dict_equip_id_detection_time in dict_target_id_equip_id_detection_time.items():
        # for循环，每个目标一个
        dict_target_eva_result = {}
        dict_target_eva_result['strTargetID'] = str_target_id

        gantt_chart = GanttChart()
        gantt_chart.str_body_name = str_target_id

        # 轨迹时间边界：把探测弧段裁剪到目标真实飞行窗口内，剔除窗口之外的伪探测
        # （卫星视场 index 错位误加 / 预案跟踪窗口超界），否则分子会超过分母而破百。
        bounds = _target_trajectory_bounds(str_target_id, battle_scene)
        bounds_range = TwoDimensionMinMax(bounds[0], bounds[1]) if bounds is not None else None
        target_clipped = 0  # 本目标被裁剪/剔除的弧段数

        for str_equip_id, lst_detection_time in dict_equip_id_detection_time.items():
            for detection_time in lst_detection_time:
                if str_equip_id not in gantt_chart.dict_activity:
                    gantt_chart.dict_activity[str_equip_id] = []

                time_range = detection_time.time_range
                if bounds_range is not None:
                    if not time_range.whether_intersected(bounds_range):
                        clipped_total += 1
                        target_clipped += 1
                        continue  # 完全落在轨迹之外，不计入覆盖率
                    if time_range.value_min < bounds_range.value_min or time_range.value_max > bounds_range.value_max:
                        clipped_total += 1  # 部分越界，被裁剪
                        target_clipped += 1
                    time_range = time_range.intersected(bounds_range)

                gantt_chart.dict_activity[str_equip_id].append(time_range)

        lst_result_over_one_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(0, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_LARGER)
        lst_result_one_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(1, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_EQUAL)
        lst_result_two_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(2, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_EQUAL)
        lst_result_over_three_cover: List[Tuple[TwoDimensionMinMax, List[str]]] = gantt_chart.overlapping_number_analysis_in_total(2, OverlappingNumberRequirement.OVERLAPPING_NUMBER_REQUIREMENT_LARGER)

        # 分母：目标自身轨迹时长（秒），与分子 over_one_cover_time（覆盖秒数）
        target_traj_total_time = _target_trajectory_duration(str_target_id, battle_scene)

        if target_traj_total_time > 0:
            dict_target_eva_result['vecContinuityResult'] = []
            dict_target_eva_result['vecReliabilityResult'] = []

            over_one_cover_time = 0
            for time_range, lst_equip_id in lst_result_over_one_cover:
                over_one_cover_time = over_one_cover_time + time_range.range()

            # 中断次数（逐装备）：每部装备对目标的跟踪段数 - 1（段间空隙 = 中断），
            # 再对装备求和。单部装备中途「停→再起」算 1 次中断；两部装备干净交接
            # （各连续一段）不产生中断。统计口径与覆盖率一致：裁剪到目标轨迹窗口内。
            interrupt_num = 0
            for str_equip_id, lst_clipped in gantt_chart.dict_activity.items():
                intervals = sorted(lst_clipped, key=lambda r: r.value_min)
                for j in range(len(intervals) - 1):
                    if intervals[j + 1].value_min - intervals[j].value_max > 1e-6:
                        interrupt_num += 1

            # 兜底：裁剪后仍可能因浮点/边界略超，覆盖秒数绝不大于轨迹时长（绝不破百）
            over_one_cover_time = min(over_one_cover_time, target_traj_total_time)

            # 平均覆盖重数：目标轨迹时长内的时间加权平均覆盖重数（精确到每个时间片，
            # 非 0/1/2/≥3 桶）。覆盖重数评分用（10重满分、少一重扣十分），分母与
            # 覆盖率同口径（自身轨迹秒数）；未覆盖时段贡献 0 重，已天然计入。
            full_overlap = gantt_chart.overlapping_analysis_in_total()
            avg_cover_num = (sum(min(len(names), 10) * r.range() for r, names in full_overlap)
                             / target_traj_total_time)

            # 诊断：打印每个目标的分子/分母，便于在离线机核对口径（秒）与破百根因
            traj_desc = f"[{bounds[0]},{bounds[1]}]" if bounds is not None else "?"
            print(f"[metric_writer] target={str_target_id} traj={traj_desc} "
                  f"denom={target_traj_total_time}s num={over_one_cover_time:.1f}s "
                  f"rate={over_one_cover_time / target_traj_total_time * 100.0:.2f}% "
                  f"avgCoverNum={avg_cover_num:.2f} clipped={target_clipped}")

            dict_continuity_result = {'dDetectCoverAge': over_one_cover_time / target_traj_total_time * 100.0,
                                      'dAvgCoverNum': avg_cover_num,
                                      'regionType': 'ENUM_COMPREHENSIVE', 'uiInterruputNum': interrupt_num}
            dict_target_eva_result['vecContinuityResult'].append(dict_continuity_result)

            one_cover_time = 0
            for time_range, lst_equip_id in lst_result_one_cover:
                one_cover_time = one_cover_time + time_range.range()

            two_cover_time = 0
            for time_range, lst_equip_id in lst_result_two_cover:
                two_cover_time = two_cover_time + time_range.range()

            over_three_cover_time = 0
            for time_range, lst_equip_id in lst_result_over_three_cover:
                over_three_cover_time = over_three_cover_time + time_range.range()

            dict_reliability_result = {}
            dict_reliability_result['regionType'] = 'ENUM_COMPREHENSIVE'
            dict_reliability_result['vecCovernumCoverage'] = [
                {'dVal': (target_traj_total_time - over_one_cover_time) / target_traj_total_time * 100.0,
                 'uiCoverNum': 0},
                {'dVal': one_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 1},
                {'dVal': two_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 2},
                {'dVal': over_three_cover_time / target_traj_total_time * 100.0, 'uiCoverNum': 3}
            ]

            dict_target_eva_result['vecReliabilityResult'].append(dict_reliability_result)

            dict_eva_result['vecTargetEvaResult'].append(dict_target_eva_result)

    data['schemeEvaluteResult']['schemeEvaResult'].append(dict_eva_result)

    data['schemeEvaluteResult'] = json.dumps(data['schemeEvaluteResult'])
    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    if clipped_total:
        print(f"[metric_writer] 已裁剪 {clipped_total} 段落在目标轨迹窗口之外的探测弧段（覆盖率破百根因）")

    return
