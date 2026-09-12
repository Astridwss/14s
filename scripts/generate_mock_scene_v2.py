# -*- coding: utf-8 -*-
"""Mock 场景生成器 v2 —— 按 new_scene.json 的真实结构生成 (默认 200雷达 + 50卫星 + 21目标)。

用法（在仓库根目录执行）:
    uv run python scripts/generate_mock_scene_v2.py \
        --n_radars 200 --n_satellites 50 --n_targets 21 --duration 2600

产物: mock_scene_200r_50s_21t.json (仓库根目录，与 mock_server 下发路径对齐)

与 v1 (scripts/generate_mock_scene.py) 的关键差异（对齐 sim 内核真实读取字段）:
  1. 卫星补上 ``sensorInfo`` —— v1 缺失此字段，sim 内核
     ``json.loads(dict_satellite.get('sensorInfo'))`` 对 None 抛 TypeError，
     导致整段卫星 + 目标解析中断，最终表现为「200 雷达 / 0 卫星 / 0 目标」。
  2. 实体 ``id`` 全部用 int（对齐 new_scene.json 与 sim 内核 ``==`` 严格比较），
     白名单 ``analyseParam.radarInfos/sateIds/airTargetIds`` 同样用 int。
  3. ``targetCalc`` / ``missileCalc`` 用 new_scene.json 的完整嵌套层级
     （strDDName / strDDID / t0 / vBfList / tarVel）。
  4. ``planScene`` 补全 ``sceneList``/``sceneChecked``，``analyseParam`` 补全
     regionIds/weaponIds/... 全字段；顶层补 ``taskInfo`` 等平台字段。
  5. 单位陷阱：目标轨迹 z 单位 = km（sim 内核 ×1000 转米）；卫星轨迹 z 单位 = m（不转）。

sim 内核 (sim/plan_file_process.py::read_battle_scene_from_json) 实际读取字段:
  雷达  : id / radarMC / radarSZWZJD(经) / radarSZWZWD(纬) / radarSZWZGD(高)
          / maxDetectionRCS / maxDetectionRange / radarZMCX / radarDSFW / min/maxElePower
  卫星  : id / satellifeName / sensorInfo(workMode.workModeParameters.azimuth/elevation/pointing)
          / targetCalc(vPtList[].dTime + geoPos.x/y/z)
  目标  : id / targetName / missileCalc(curTarInfo.vBfList[].vPtList[].dTime + geoPos.x/y/z)
"""
import argparse
import json
import math
import os
import random

# 轨迹采样间隔（秒）—— 与仿真引擎 time_step 对齐取 1，保证任何 time_step 都能命中轨迹点。
SAMPLING_INTERVAL_S = 1

# 卫星红外传感器视场参数（与 new_scene.json 保持一致）：
#   azimuth=2 / elevation=2 → azi_min=-1° / ele_min=-1°（±1° 窄视场），pointing=5° 最大指向角。
SENSOR_AZIMUTH = 2.0
SENSOR_ELEVATION = 2.0
SENSOR_POINTING = 5.0


def _linspace_points(start, end, n):
    """n 个点从 start 线性插值到 end（含两端）。"""
    if n <= 1:
        return [start]
    return [start + (end - start) * (i / (n - 1)) for i in range(n)]


def _make_radar(rid: int, i: int) -> dict:
    """一部雷达：中国区域内随机位置 + 全向扫描 + 全俯仰。

    radarZMCX=180 / radarDSFW=360 → azi_min=0 / azi_max=360（全向）。
    """
    lat = 18.0 + random.random() * 36.0        # 18 ~ 54
    lon = 73.0 + random.random() * 62.0        # 73 ~ 135
    alt = random.uniform(50, 3000)             # 50 ~ 3000 m
    return {
        "id": rid,
        "radarMC": f"雷达_{i:04d}",
        "radarSZWZJD": round(lon, 6),
        "radarSZWZWD": round(lat, 6),
        "radarSZWZGD": round(alt, 2),
        "maxDetectionRCS": round(random.uniform(0.5, 1.5), 4),
        "maxDetectionRange": round(random.uniform(2000, 5000), 2),
        "radarZMCX": 180.0,                    # 方位中心
        "radarDSFW": 360.0,                    # 方位扫描宽度 → 全向
        "minElePower": 0.0,
        "maxElePower": 90.0,
    }


def _make_satellite(sid: int, i: int, duration_s: int, n_static_sats: int) -> dict:
    """一颗卫星：GEO(静止) 或 LEO(圆形轨道)，含 sensorInfo + targetCalc 完整结构。"""
    vpt_list = []
    if i < n_static_sats:
        # GEO: 赤道上方 ~35786 km，固定经度（所有时间点同位置）
        geo_lon = 50.0 + (i / max(n_static_sats, 1)) * 200.0
        geo_lat = 0.0
        geo_alt = 35786000.0
        for t in range(0, duration_s + 1, SAMPLING_INTERVAL_S):
            vpt_list.append({
                "dTime": float(t),
                "geoPos": {"x": round(geo_lon, 6), "y": geo_lat, "z": geo_alt},
                "tarVel": {"x": 0.0, "y": 0.0, "z": 0.0},
            })
    else:
        # LEO: 倾斜圆形轨道，~500 km 高度（简化二体近似，不追求物理精确）
        orbit_period = random.uniform(300, 600)
        inclination = random.uniform(30, 98)
        raan = random.uniform(0, 360)
        start_anomaly = random.uniform(0, 360)
        alt_base = 500000.0 + random.uniform(-10000, 10000)  # 单位 m
        for t in range(0, duration_s + 1, SAMPLING_INTERVAL_S):
            anomaly = start_anomaly + (t / orbit_period) * 360.0
            lat = inclination * math.sin(math.radians(anomaly))
            lon = raan + (anomaly * math.cos(math.radians(inclination)))
            lon = lon % 360.0
            if lon > 180:
                lon -= 360
            vpt_list.append({
                "dTime": float(t),
                "geoPos": {"x": round(lon, 6), "y": round(lat, 6), "z": round(alt_base, 2)},
                "tarVel": {"x": 0.0, "y": 0.0, "z": 0.0},
            })

    sensor_info = {
        "sensorInfo": [{
            "sensor": "infrared",
            "id": "1",
            "workMode": [{
                "code": 1,
                "workModeParameters": {
                    "capabilityIndicator": [],
                    "pointing": SENSOR_POINTING,
                    "azimuth": SENSOR_AZIMUTH,
                    "elevation": SENSOR_ELEVATION,
                },
                "name": "搜索/跟踪模式",
            }],
        }],
    }
    target_calc = {
        "vPtList": vpt_list,
        "strDDName": f"卫星_{i:04d}",
        "strDDID": str(sid),
        "t0": 0.0,
    }
    return {
        "id": sid,
        "satellifeName": f"卫星_{i:04d}",
        "sensorInfo": json.dumps(sensor_info, ensure_ascii=False),
        "targetCalc": json.dumps(target_calc, ensure_ascii=False),
    }


def _make_target(tid: int, i: int, duration_s: int) -> dict:
    """一个目标：直线弹道，z 单位 km（sim 内核 ×1000 转米）。"""
    start_lat = random.uniform(20, 50)
    start_lon = random.uniform(75, 130)
    end_lat = random.uniform(20, 50)
    end_lon = random.uniform(75, 130)
    start_alt = random.uniform(100, 300)   # km
    end_alt = random.uniform(10, 50)       # km

    vpt_list = []
    for t in range(0, duration_s + 1, SAMPLING_INTERVAL_S):
        frac = t / duration_s
        vpt_list.append({
            "dTime": float(t),
            "geoPos": {
                "x": round(start_lon + (end_lon - start_lon) * frac, 6),
                "y": round(start_lat + (end_lat - start_lat) * frac, 6),
                "z": round(start_alt + (end_alt - start_alt) * frac, 6),
            },
            "tarVel": {"x": 0.0, "y": 0.0, "z": 0.0},
        })

    missile_calc = {
        "curTarInfo": {
            "strDDID": str(tid),
            "strDDName": f"目标_{i:04d}",
            "t0": 0.0,
            "vBfList": [{
                "strBfName": "主动段",
                "strBfType": "1",
                "vPtList": vpt_list,
            }],
        },
        "strCalcResult": "",
        "vecRegionlist": [],
    }
    return {
        "id": tid,
        "targetName": f"目标_{i:04d}",
        "missileCalc": json.dumps(missile_calc, ensure_ascii=False),
    }


def generate_mock_scene(
    n_radars: int = 200,
    n_satellites: int = 50,
    n_static_sats: int = 10,
    n_targets: int = 21,
    duration_s: int = 2600,
    plan_id: int = 9999,
    output_path: str = "mock_scene_200r_50s_21t.json",
    seed: int = 42,
):
    random.seed(seed)

    # ---- 1. 雷达 ----
    radar_ids = [100000 + i for i in range(n_radars)]
    radar_list = [_make_radar(rid, i) for i, rid in enumerate(radar_ids)]

    # ---- 2. 卫星 ----
    satellite_ids = [200000 + i for i in range(n_satellites)]
    satellite_list = [
        _make_satellite(sid, i, duration_s, n_static_sats)
        for i, sid in enumerate(satellite_ids)
    ]

    # ---- 3. 目标 ----
    target_ids = [300000 + i for i in range(n_targets)]
    missile_list = [_make_target(tid, i, duration_s) for i, tid in enumerate(target_ids)]

    # ---- 4. planScene（analyseParam 白名单 = 全部实体，int id）----
    analyse_param = {
        "regionIds": [],
        "radarInfos": radar_ids,
        "weaponIds": [],
        "positionIds": [],
        "airTargetIds": target_ids,
        "cruiseTargetIds": [],
        "volleyTargetIds": [],
        "areaFDIds": [],
        "areaLDIds": [],
        "sateIds": satellite_ids,
        "platformEquipmentIds": [],
    }
    plan_scene = {
        "sceneList": [],           # 前端场景树，sim 内核不读，本地训练用空即可
        "analyseParam": analyse_param,
        "sceneChecked": True,
    }

    # ---- 5. 单预案 ----
    plan_info = {
        "planId": plan_id,
        "associatedTaskId": plan_id,
        "associatedTaskName": f"mock场景-{n_radars}r{n_satellites}s{n_targets}t",
        "planName": f"mock场景-{n_radars}r{n_satellites}s{n_targets}t",
        "startTime": 0,
        "endTime": duration_s * 1000,          # 毫秒；sim 内核取差 /1000 = duration_s 秒
        "relativeStartTime": 0,
        "k0Time": 0,
        "schemeMakeParam": "11111111111111",
        "enabledFlag": True,
        "planScene": json.dumps(plan_scene, ensure_ascii=False),
        "radarList": radar_list,
        "satellifeList": satellite_list,
        "missileTargetList": missile_list,
        "cruiseTargetList": [],
        "volleyTargetList": [],
        "weaponList": [],
    }

    # ---- 6. 顶层（对齐 new_scene.json 的平台字段）----
    data = {
        "planInfoList": [plan_info],
        "taskDeployRegionList": [],
        "taskInfo": {
            "id": plan_id,
            "name": f"mock场景-{n_radars}r{n_satellites}s{n_targets}t",
            "startTime": 0,
            "endTime": duration_s * 1000,
            "planStartTime": 0,
            "planEndTime": duration_s * 1000,
            "type": 1,
            "source": 0,
            "enabledFlag": True,
        },
        "taskRegionList": [],
        "taskTargetList": [],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Mock v2] 生成完成: {output_path}")
    print(f"          雷达={n_radars}, 卫星={n_satellites}(GEO={n_static_sats}), "
          f"目标={n_targets}, 时长={duration_s}s, plan_id={plan_id}")
    print(f"          文件大小: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_radars", type=int, default=200)
    ap.add_argument("--n_satellites", type=int, default=50)
    ap.add_argument("--n_static_sats", type=int, default=10)
    ap.add_argument("--n_targets", type=int, default=21)
    ap.add_argument("--duration", type=int, default=2600)
    ap.add_argument("--plan_id", type=int, default=9999)
    ap.add_argument("--output", type=str,
                    default="mock_scene_200r_50s_21t.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate_mock_scene(
        n_radars=args.n_radars,
        n_satellites=args.n_satellites,
        n_static_sats=args.n_static_sats,
        n_targets=args.n_targets,
        duration_s=args.duration,
        plan_id=args.plan_id,
        output_path=args.output,
        seed=args.seed,
    )
