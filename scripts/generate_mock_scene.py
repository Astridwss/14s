"""Mock 场景 JSON 生成器 —— 25雷达 + 25卫星 + 21目标（不饱和分组协调场景）。

用法:
    # 在仓库根目录执行，产物落在根目录，与 mock_server 的下发路径对齐
    uv run python test/generate_mock_scene.py

产出:
    mock_scene_25r_25s.json (仓库根目录)
"""
import json
import math
import os
import random
import sys

# 确保能 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 轨迹采样间隔（秒）——必须与仿真引擎的 time_step 一致。
# sim/client.py 的 load_battle_scene(time_step=1) 固定 1 秒步进，
# 采样间隔取 1 才能保证每个仿真步 current_time 都命中轨迹点。
SAMPLING_INTERVAL_S = 1


def generate_mock_scene(
    n_radars: int = 25,
    n_satellites: int = 25,
    n_static_sats: int = 10,   # GEO 静止卫星数量
    n_targets: int = 21,
    duration_s: int = 600,
    plan_id: int = 9999,
    output_path: str = "mock_scene_25r_25s.json",
    seed: int = 42,
):
    random.seed(seed)

    # ============================================================
    # 1. 生成雷达 (25 个，均匀分布在中国区域)
    # ============================================================
    # 中国区域大致范围: lat 18-54, lon 73-135
    radar_list = []
    radar_ids = []
    for i in range(n_radars):
        rid = f"R{i:04d}"
        radar_ids.append(rid)
        lat = 18.0 + random.random() * 36.0       # 18 ~ 54
        lon = 73.0 + random.random() * 62.0        # 73 ~ 135
        alt = random.uniform(50, 3000)              # 50m ~ 3000m
        radar_list.append({
            "id": rid,
            "radarMC": f"雷达_{i:04d}",
            "radarSZWZJD": lon,
            "radarSZWZWD": lat,
            "radarSZWZGD": alt,
            "maxDetectionRange": random.uniform(2000, 4000),
            "maxDetectionRCS": random.uniform(0.5, 1.5),
            "minElePower": 0.0,
            "maxElePower": 90.0,
        })

    # ============================================================
    # 2. 生成卫星 (25 个)
    #    - 前 n_static_sats 个: GEO 静止 (所有时间点相同位置)
    #    - 剩余: LEO 移动 (沿简单轨道运动)
    # ============================================================
    satellite_list = []
    satellite_ids = []
    for i in range(n_satellites):
        sid = f"S{i:04d}"
        satellite_ids.append(sid)

        if i < n_static_sats:
            # GEO: 赤道上方 ~35786km，固定经度
            geo_lon = 50.0 + (i / n_static_sats) * 200.0  # 50°E ~ 250°E
            vpt_list = []
            for t in range(0, duration_s + 1, SAMPLING_INTERVAL_S):
                vpt_list.append({
                    "dTime": t,
                    "geoPos": {"x": geo_lon, "y": 0.0, "z": 35786000.0},
                })
        else:
            # LEO: 倾斜轨道，~500km 高度
            orbit_period = random.uniform(300, 600)  # 轨道周期 (秒)
            inclination = random.uniform(30, 98)      # 倾角 (度)
            raan = random.uniform(0, 360)              # 升交点赤经
            start_anomaly = random.uniform(0, 360)     # 初始近点角
            vpt_list = []
            for t in range(0, duration_s + 1, SAMPLING_INTERVAL_S):
                # 简化二体运动: 圆形轨道
                anomaly = start_anomaly + (t / orbit_period) * 360.0
                # 近似计算 sub-satellite point
                lat = inclination * math.sin(math.radians(anomaly))
                lon = raan + (anomaly * math.cos(math.radians(inclination)))
                # 范围修正
                lon = lon % 360.0
                if lon > 180:
                    lon -= 360
                vpt_list.append({
                    "dTime": t,
                    "geoPos": {
                        "x": lon,
                        "y": lat,
                        "z": 500000.0 + random.uniform(-10000, 10000),
                    },
                })

        satellite_list.append({
            "id": sid,
            "satellifeName": f"卫星_{i:04d}",
            "targetCalc": json.dumps({"vPtList": vpt_list}),
        })

    # ============================================================
    # 3. 生成目标 (21 个导弹目标，直线轨迹)
    # ============================================================
    missile_list = []
    target_ids = []
    for i in range(n_targets):
        tid = f"T{i:04d}"
        target_ids.append(tid)

        # 起点和终点 (在中国区域内随机)
        start_lat = random.uniform(20, 50)
        start_lon = random.uniform(75, 130)
        end_lat = random.uniform(20, 50)
        end_lon = random.uniform(75, 130)

        # 高度 (km): plan_file_process 会对目标 z × 1000 转为米
        start_alt = random.uniform(100, 300)     # 100-300 km
        end_alt = random.uniform(10, 50)          # 10-50 km

        vbf_list = []
        vpt_list = []
        for t in range(0, duration_s + 1, SAMPLING_INTERVAL_S):
            frac = t / duration_s
            lat = start_lat + (end_lat - start_lat) * frac
            lon = start_lon + (end_lon - start_lon) * frac
            alt = start_alt + (end_alt - start_alt) * frac
            vpt_list.append({
                "dTime": t,
                "geoPos": {"x": lon, "y": lat, "z": alt},
            })

        vbf_list.append({"vPtList": vpt_list})
        missile_list.append({
            "id": tid,
            "targetName": f"目标_{i:04d}",
            "missileCalc": json.dumps({
                "curTarInfo": {"vBfList": vbf_list},
            }),
        })

    # ============================================================
    # 4. 组装 JSON
    # ============================================================
    radar_info_ids = [str(rid) for rid in radar_ids]
    sate_info_ids = [str(sid) for sid in satellite_ids]
    air_target_ids = [str(tid) for tid in target_ids]

    plan_scene_data = {
        "analyseParam": {
            "radarInfos": radar_info_ids,
            "sateIds": sate_info_ids,
            "airTargetIds": air_target_ids,
        },
    }

    plan_info = {
        "planId": plan_id,
        "associatedTaskId": plan_id,
        "startTime": 0,
        "endTime": duration_s * 1000,  # 毫秒
        "planScene": json.dumps(plan_scene_data),
        "radarList": radar_list,
        "satellifeList": satellite_list,
        "missileTargetList": missile_list,
    }

    data = {"planInfoList": [plan_info]}

    # ============================================================
    # 5. 写文件
    # ============================================================
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Mock] 生成完成: {output_path}")
    print(f"        雷达={n_radars}, 卫星={n_satellites}(GEO={n_static_sats}), "
          f"目标={n_targets}, 时长={duration_s}s")
    print(f"        文件大小: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    generate_mock_scene()
