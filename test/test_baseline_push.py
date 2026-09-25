"""基线推送冒烟测试 —— 无需模型权重即可验证基线链路与接口连通性。

背景：当前还没有可用于推理的权重，前台触发不到推演接口，因而无法通过真实
模型推理去联调基线推送。本用例用「真实预案 + 伪造推理动作流」把整条链路跑通：

    1. 真实基线      scene.json splitQuduanResult → BaselineEvaluator → 产物 + 覆盖率
    2. 伪造推理      构造 AgentActionCommand 流 → 走真实推理写函数 → 产物 + 覆盖率
    3. 结构一致性    断言基线产物与推理产物的 JSON 结构逐键同构
    4. 接口连通性    同步探针拿状态码 + 真实 Pusher 异步推送
    5. 降级路径      mock 场景（无 splitQuduanResult）应跳过推送而非推 0%

运行前先起伪装平台::

    python test/mock_server.py          # 监听 127.0.0.1:8080

然后::

    python test/test_baseline_push.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import requests

from sim import PlanFileProcess
from sim.datastruct import AgentActionCommand
from services.evaluation import (
    BaselineEvaluator, parse_metric_file, scene_target_ids, write_plan_metric_to_json,
)
from use_cases.pusher import Pusher

# ---- 测试参数 ----
SCENE_PATH = str(_PROJ_ROOT / "scene.json")                     # 含 splitQuduanResult
MOCK_SCENE_PATH = str(_PROJ_ROOT / "mock_scene_100r_25s.json")  # 不含，用于降级路径
PLAN_ID = 867
TASK_ID = "BASELINE_SMOKE_TEST"
OUT_DIR = str(_PROJ_ROOT / "eval_records")
BASE_URL = "http://127.0.0.1:8080"


# ============================================================
# 工具：JSON 结构指纹
# ============================================================

def _expand(obj):
    """递归展开嵌套的 JSON 字符串（sim 把内层 dump 成了字符串）。"""
    if isinstance(obj, str):
        stripped = obj.strip()
        if stripped[:1] in ("{", "["):
            try:
                return _expand(json.loads(stripped))
            except ValueError:
                return obj
        return obj
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(i) for i in obj]
    return obj


def _shape(obj):
    """提取结构指纹：只保留键名与叶子类型，忽略具体取值。"""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        # 列表元素同构，取第一个作代表；空列表记为 []
        return [_shape(obj[0])] if obj else []
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, (int, float)):
        return "number"
    return type(obj).__name__


def _file_shape(path):
    with open(path, "r", encoding="utf-8") as f:
        return _shape(_expand(json.load(f)))


# ============================================================
# 步骤 1：真实基线
# ============================================================

def step1_baseline():
    print("\n" + "=" * 64)
    print("[步骤1] 从真实预案生成基线产物")
    print("=" * 64)

    artifacts = BaselineEvaluator().generate(
        scene_path=SCENE_PATH, plan_id=PLAN_ID,
        out_dir=OUT_DIR, task_id=TASK_ID,
    )

    assert artifacts.available, f"基线不可用: {artifacts.reason}"
    assert os.path.exists(artifacts.records_path), "基线时序文件未落盘"
    assert os.path.exists(artifacts.metric_path), "基线指标文件未落盘"

    cov = artifacts.coverage
    print(f"  时序文件: {artifacts.records_path}")
    print(f"  指标文件: {artifacts.metric_path}")
    print(f"  覆盖率  : {cov.describe()}")
    print(f"  逐目标  : {cov.per_target}")
    return artifacts


# ============================================================
# 步骤 2：伪造推理产物（走真实推理写函数）
# ============================================================

def _fake_inference_stream(battle_scene, cover_ratio=0.5):
    """构造伪推理动作流：每个目标由一部雷达跟踪其前半段航迹。

    走的是真实的 ``Dict[int, List[AgentActionCommand]]`` 格式，因此后续能直接
    喂给 ``write_model_inference_result_to_json``，与线上推理路径完全一致。
    """
    radar_ids = sorted(battle_scene.dict_radar_id_info)
    target_ids = sorted(battle_scene.dict_target_id_info)
    assert radar_ids and target_ids, "场景缺少雷达或目标，无法伪造推理数据"

    stream = {}
    for idx, target_id in enumerate(target_ids):
        equip_id = radar_ids[idx % len(radar_ids)]
        times = sorted(
            battle_scene.dict_target_id_info[target_id].dict_target_traj_pt_info
        )
        for t in times[: int(len(times) * cover_ratio)]:
            stream.setdefault(t, []).append(
                AgentActionCommand(
                    time=t, str_equip_id=equip_id, str_target_id=target_id,
                )
            )
    return stream


def step2_fake_inference(battle_scene):
    print("\n" + "=" * 64)
    print("[步骤2] 伪造推理动作流并走真实推理写函数")
    print("=" * 64)

    stream = _fake_inference_stream(battle_scene)
    processor = PlanFileProcess()

    records_path = os.path.abspath(
        os.path.join(OUT_DIR, f"{TASK_ID}_eval_records.json")
    )
    metric_path = os.path.abspath(
        os.path.join(OUT_DIR, f"{TASK_ID}_eval_metric.json")
    )

    processor.write_model_inference_result_to_json(
        dict_model_inference_result=stream, battle_scene=battle_scene,
        dest_path=records_path,
    )
    # 指标文件改用 sim 外重写版（分母=轨迹时长 + 平均覆盖重数 dAvgCoverNum），
    # 与线上推理 eval_runner 同路径，且与基线(metric_writer)在 JSON 结构上同构。
    plan_result = processor.write_model_inference_result_to_plan_result(
        dict_model_inference_result=stream, battle_scene=battle_scene,
    )
    write_plan_metric_to_json(
        plan_result=plan_result, battle_scene=battle_scene, dest_path=metric_path,
    )

    summary = parse_metric_file(metric_path, scene_target_ids(battle_scene))
    print(f"  动作流时刻数: {len(stream)}")
    print(f"  时序文件: {records_path}")
    print(f"  指标文件: {metric_path}")
    print(f"  覆盖率  : {summary.describe()}")
    return records_path, metric_path, summary


# ============================================================
# 步骤 3：结构一致性
# ============================================================

def step3_structure(baseline, infer_records, infer_metric):
    print("\n" + "=" * 64)
    print("[步骤3] 基线产物 vs 推理产物 结构一致性")
    print("=" * 64)

    pairs = [
        ("时序文件", baseline.records_path, infer_records),
        ("指标文件", baseline.metric_path, infer_metric),
    ]
    for label, base_path, infer_path in pairs:
        base_shape = _file_shape(base_path)
        infer_shape = _file_shape(infer_path)
        assert base_shape == infer_shape, (
            f"{label}结构不一致\n  基线: {json.dumps(base_shape, ensure_ascii=False)}\n"
            f"  推理: {json.dumps(infer_shape, ensure_ascii=False)}"
        )
        print(f"  [OK] {label}结构同构")
        print(f"       {json.dumps(base_shape, ensure_ascii=False)}")


# ============================================================
# 步骤 4：接口连通性
# ============================================================

def step4_push(baseline, infer_records, infer_metric, infer_summary):
    print("\n" + "=" * 64)
    print("[步骤4] 接口连通性")
    print("=" * 64)

    infer_payload = {
        "taskId": TASK_ID,
        "timeSeriesFile": infer_records,
        "evalFile": infer_metric,
        "coverage": infer_summary.coverage,
    }
    baseline_payload = {
        "taskId": TASK_ID,
        "taskName": "baseline",
        "timeSeriesFile": baseline.records_path,
        "evalFile": baseline.metric_path,
        "coverage": baseline.coverage.coverage,
    }

    # 4a. 同步探针：拿到真实状态码，作为连通性的硬证据
    probes = [
        ("推理接口 /api/receive/eval_result",
         f"{BASE_URL}/api/receive/eval_result", infer_payload),
        ("基线接口 /api/receive/compare_eval_result",
         f"{BASE_URL}/api/receive/compare_eval_result", baseline_payload),
    ]
    for label, url, payload in probes:
        try:
            resp = requests.post(url, json=payload, timeout=5)
            print(f"  [同步探针] {label} -> HTTP {resp.status_code} {resp.text}")
            assert resp.status_code == 200, f"{label} 返回非 200"
        except requests.RequestException as e:
            raise AssertionError(
                f"{label} 连接失败，请先启动 python test/mock_server.py : {e}"
            )

    # 4b. 走真实 Pusher（异步 daemon 线程，taskId/taskName 由 Pusher 补）
    push = Pusher(task_id=TASK_ID, base_url=BASE_URL)
    push.push_eval_result({
        "timeSeriesFile": infer_records,
        "evalFile": infer_metric,
        "coverage": infer_summary.coverage,
    })
    push.push_baseline_eval_result({
        "timeSeriesFile": baseline.records_path,
        "evalFile": baseline.metric_path,
        "coverage": baseline.coverage.coverage,
    })
    time.sleep(2)  # daemon 线程非阻塞，等它把包发完再退出进程
    print("  [Pusher] 推理 + 基线 已异步推送（详见伪装平台控制台）")

    # 4c. 提升率（后端自行计算，这里只做展示）
    base_cov = baseline.coverage.coverage
    if base_cov > 0:
        improve = (infer_summary.coverage - base_cov) / base_cov * 100.0
        print(f"  [参考] 推理 {infer_summary.coverage:.2f}% vs "
              f"基线 {base_cov:.2f}% -> 提升 {improve:+.2f}%")


# ============================================================
# 步骤 5：降级路径
# ============================================================

def step5_degrade():
    print("\n" + "=" * 64)
    print("[步骤5] 降级路径：预案无 splitQuduanResult 时应跳过推送")
    print("=" * 64)

    if not os.path.exists(MOCK_SCENE_PATH):
        print(f"  [跳过] mock 场景不存在: {MOCK_SCENE_PATH}")
        return

    artifacts = BaselineEvaluator().generate(
        scene_path=MOCK_SCENE_PATH, plan_id=PLAN_ID,
        out_dir=OUT_DIR, task_id=f"{TASK_ID}_MOCK",
    )
    assert not artifacts.available, "mock 场景无专家规划，却报告基线可用"
    assert not artifacts.records_path, "降级路径不应产出文件"
    print(f"  [OK] 已正确跳过: {artifacts.reason}")

    missing = BaselineEvaluator().generate(
        scene_path=str(_PROJ_ROOT / "__not_exist__.json"), plan_id=PLAN_ID,
        out_dir=OUT_DIR, task_id=f"{TASK_ID}_MISSING",
    )
    assert not missing.available, "预案文件不存在，却报告基线可用"
    print(f"  [OK] 已正确跳过: {missing.reason}")


def main():
    global BASE_URL

    parser = argparse.ArgumentParser(description="基线推送冒烟测试")
    parser.add_argument("--base-url", default=BASE_URL,
                        help="平台地址（伪装平台或真实平台）")
    parser.add_argument("--skip-push", action="store_true",
                        help="跳过步骤4，只验证基线生成/结构/降级（无需任何服务）")
    args = parser.parse_args()
    BASE_URL = args.base_url

    baseline = step1_baseline()
    battle_scene = PlanFileProcess().read_battle_scene_from_json(
        plan_id=PLAN_ID, file_path=SCENE_PATH,
    )
    infer_records, infer_metric, infer_summary = step2_fake_inference(battle_scene)
    step3_structure(baseline, infer_records, infer_metric)

    if args.skip_push:
        print("\n[步骤4] 已跳过（--skip-push）。接口连通性请用 "
              "python test/probe_platform.py 单独探测")
    else:
        step4_push(baseline, infer_records, infer_metric, infer_summary)

    step5_degrade()

    print("\n" + "=" * 64)
    print("[结论] 基线链路全部通过")
    print("=" * 64)


if __name__ == "__main__":
    main()
