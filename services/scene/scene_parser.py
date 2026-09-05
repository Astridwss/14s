"""
场景解析 —— 读取场景 JSON，返回实体信息和网络维度。

纯函数，不依赖 Config，不产生副作用。
"""
import os
import json
from typing import Tuple, List
from services.scene.scene_constants import (
    RADAR_SELF_FEATURES, TARGET_FEATURES, SATELLITE_BROADCAST_FEATURES,
    RADAR_STATE_FEATURES, TARGET_STATE_FEATURES,
)


def _resolve_plan_id(scene_path: str, plan_id: int) -> int:
    """预校验 plan_id 与场景文件是否匹配，不匹配时给出明确报错。

    ``sim/plan_file_process.py`` 是固定内核，内部用 ``current_id == plan_id`` 严格
    ``==`` 比较。这里提前读 JSON，找出 planInfoList 里每条预案的真实 id
    （planId / associatedTaskId / id）；多预案且传入 plan_id 一条都匹配不上时，
    抛明确的「planid 不匹配」错误，避免内核静默返回空 battle_scene 后再伪装成
    「实体数为 0」。plan_id 类型已由 schema 层（plan_id: int）保证为 int。
    """
    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return plan_id

    plan_info_list = data.get("planInfoList", [])
    if not isinstance(plan_info_list, list) or not plan_info_list:
        return plan_id

    available_ids = []
    for entry in plan_info_list:
        if not isinstance(entry, dict):
            continue
        current_id = entry.get("planId") or entry.get("associatedTaskId") or entry.get("id")
        available_ids.append(current_id)
        if str(current_id) == str(plan_id):
            return plan_id

    if len(plan_info_list) > 1:
        raise ValueError(
            f"[SceneParser] planid 不匹配！传入 plan_id={plan_id!r}，"
            f"但场景文件 {os.path.basename(scene_path)} 的 planInfoList 中不存在该 id。"
            f"文件内可用预案 id: {available_ids}"
        )
    return plan_id


def _normalize_scene_id_types(scene_path: str) -> str:
    """把 analyseParam 白名单 id 的类型对齐到外层实体列表 id 的类型。

    平台下发的场景里，外层列表（radarList/satellifeList/missileTargetList）的 id 是
    int，而 analyseParam 里的白名单（radarInfos/sateIds/airTargetIds）有时是字符串，
    导致 sim 内核 ``dict_xx.get('id') not in 白名单`` 恒为 True，实体被全部跳过
    （表现为「雷达有、卫星0、目标0」）。这里把白名单 id 强制成与外层 id 相同的
    类型，写一份临时副本交给 sim 解析；不动原文件、不改 sim 内核。

    返回归一化后的临时文件路径；若无需修改则原样返回 scene_path。
    """
    import tempfile

    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return scene_path

    plan_info_list = data.get("planInfoList", [])
    if not isinstance(plan_info_list, list):
        return scene_path

    # 白名单字段 -> 对应外层实体列表字段（用于对齐 id 类型）
    whitelist_to_outer = {
        "radarInfos": ("radarList",),
        "sateIds": ("satellifeList", "satelliteList"),
        "airTargetIds": ("missileTargetList", "targetList"),
    }

    changed = False
    for entry in plan_info_list:
        if not isinstance(entry, dict):
            continue
        plan_scene = entry.get("planScene")
        if not isinstance(plan_scene, str):
            continue
        try:
            plan_scene_data = json.loads(plan_scene)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(plan_scene_data, dict):
            continue
        ap = plan_scene_data.get("analyseParam")
        if not isinstance(ap, dict):
            continue

        entry_changed = False
        for whitelist_key, outer_keys in whitelist_to_outer.items():
            whitelist = ap.get(whitelist_key)
            if not isinstance(whitelist, list) or not whitelist:
                continue

            outer_list = None
            for ok in outer_keys:
                candidate = entry.get(ok)
                if isinstance(candidate, list):
                    outer_list = candidate
                    break
            if not outer_list:
                continue

            outer_ids = [
                x.get("id") for x in outer_list
                if isinstance(x, dict) and x.get("id") is not None
            ]
            if not outer_ids:
                continue
            outer_type = type(outer_ids[0])
            if outer_type not in (int, str):
                continue

            coerced = []
            for wid in whitelist:
                if isinstance(wid, outer_type):
                    coerced.append(wid)
                else:
                    try:
                        coerced.append(outer_type(wid))
                    except (TypeError, ValueError):
                        coerced.append(wid)

            if coerced != whitelist:
                ap[whitelist_key] = coerced
                entry_changed = True

        if entry_changed:
            entry["planScene"] = json.dumps(plan_scene_data, ensure_ascii=False)
            changed = True

    if not changed:
        return scene_path

    fd, tmp_path = tempfile.mkstemp(
        suffix=".json", prefix=".scene_normalized_",
        dir=os.path.dirname(scene_path) or ".",
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return tmp_path


def extract_entities(scene_path: str, plan_id: int = 867) -> Tuple[int, int, int, List[str], List[str], List[str]]:
    """解析场景 JSON，返回实体数量与 ID 列表。

    Returns:
        (n_radars, n_satellites, n_targets, radar_keys, target_keys, satellite_keys)
    """
    if not scene_path or not os.path.exists(scene_path):
        raise FileNotFoundError(f"[SceneParser] 场景文件不存在: {scene_path}")

    normalized_path = scene_path
    try:
        plan_id = _resolve_plan_id(scene_path, plan_id)
        normalized_path = _normalize_scene_id_types(scene_path)

        from sim.plan_file_process import PlanFileProcess
        processor = PlanFileProcess()
        battle_scene = processor.read_battle_scene_from_json(plan_id=plan_id, file_path=normalized_path)

        radar_keys = list(battle_scene.dict_radar_id_info.keys())
        satellites_keys = list(battle_scene.dict_satellite_id_info.keys())
        target_keys = list(battle_scene.dict_target_id_info.keys())

        n_radars = len(radar_keys)
        n_satellites = len(satellites_keys)
        n_targets = len(target_keys)

        if n_radars == 0 or n_targets == 0:
            raise ValueError(
                f"[SceneParser] 场景数据异常！"
                f"雷达={n_radars}, 卫星={n_satellites}, 目标={n_targets}。"
                f"核心实体数量绝不能为 0，请检查场景预案 (PlanID: {plan_id})"
            )

        print(f"[SceneParser] 提取成功 (PlanID:{plan_id}): "
              f"雷达={n_radars}, 卫星={n_satellites}, 目标={n_targets}")
        return n_radars, n_satellites, n_targets, radar_keys, target_keys, satellites_keys

    except (json.JSONDecodeError, KeyError, AttributeError, ImportError, OSError) as e:
        print(f"[SceneParser] 场景文件读取或解析失败: {e}")
        raise
    except ValueError:
        raise
    finally:
        if normalized_path and normalized_path != scene_path:
            try:
                os.remove(normalized_path)
            except OSError:
                pass


def compute_dimensions(n_radars: int, n_satellites: int, n_targets: int) -> dict:
    """根据实体数量推导网络张量维度。

    统一骨架：智能体 = 雷达(LD) + 卫星(WX)，从第一天固定 n_agents，
    避免 phase1(LD) → phase2(LD+WX) 之间任何权重维度变化。

    动作空间：
      - WX 单选：n_actions = n_targets + 1（0=待机，1..N=锁定目标）
      - LD 多标签：ld_n_actions = n_targets（无待机维，逐目标独立 Q，top-20 多选）

    Returns:
        {
            "n_agents": int, "n_ld": int, "n_wx": int,
            "n_actions": int, "ld_n_actions": int,
            "radar_obs_dim": int, "obs_shape": int, "state_shape": int,
        }
    """
    n_ld = n_radars
    n_wx = n_satellites
    n_agents = n_radars + n_satellites          # 统一骨架：200 LD + 25 WX = 225
    n_actions = n_targets + 1                   # WX 单选空间（0=待机，1..N=目标）
    ld_n_actions = n_targets                    # LD 多标签空间（仅目标维）
    radar_obs_dim = (
        RADAR_SELF_FEATURES
        + n_targets * (TARGET_FEATURES + SATELLITE_BROADCAST_FEATURES)
    )
    state_dim = (n_targets * TARGET_STATE_FEATURES) + (n_agents * RADAR_STATE_FEATURES) + 1

    dims = {
        "n_agents": n_agents,
        "n_ld": n_ld,
        "n_wx": n_wx,
        "n_actions": n_actions,
        "ld_n_actions": ld_n_actions,
        "radar_obs_dim": radar_obs_dim,
        "obs_shape": radar_obs_dim,
        "state_shape": state_dim,
    }

    print(f"[SceneParser] 维度推导: n_agents={n_agents}(LD={n_ld}, WX={n_wx}), "
          f"n_actions={n_actions}(WX单选), ld_n_actions={ld_n_actions}(LD多标签), "
          f"obs={radar_obs_dim}, state={state_dim}")
    return dims
