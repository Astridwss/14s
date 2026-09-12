"""
专家数据生成 —— 接收外部构建好的积木，按时间步生成 RL 张量并保存为 CSV。

不调用同层 services/scene/ —— 依赖由调用方显式注入。

性能：时间步彼此独立（sweep 事件已预计算成 equip_events[eid][t]，无跨步状态），
是天然的 embarrassingly parallel。默认用 fork 进程池按「连续时间步块」切分并行：
  - 大对象（battle_scene 等）放模块级 _G，fork 后经 COW 共享，零序列化开销；
  - 每 worker 只读 battle_scene（不写），COW 页保持共享，内存不随 worker 数膨胀；
  - 每 worker 写独立块文件，父进程按序拼接 → 输出与串行版逐字节一致；
  - N=1 或无 fork（macOS/Windows）自动回退串行路径。

同时保留 numpy 向量化（逐字节等价）：
  - B 段一次广播 geodetic2aer 算全雷达×目标 az/el/斜距；
  - C 段目标 ECEF 每步只算一次（原在每颗卫星循环内重复）；
  - E 段 obs/avail 整矩阵 astype(str) 替代逐行 Python 级 str()。
"""
import os
import io
import csv
import math
import time
import multiprocessing
from collections import defaultdict

import numpy as np
import pymap3d as pm

from sim.datastruct import (
    AgentObservation, SystemTrackBase, EquipmentState, EquipmentToTargetDetectionResult,
)

# 父进程 fork 前填充的共享只读数据；子进程经 COW 继承，避免经 Pool 参数 pickle 大对象。
_G = {}


def _fork_available() -> bool:
    try:
        return multiprocessing.get_context('fork') is not None
    except ValueError:
        return False


def _split_range(start: int, end: int, n: int):
    """把 [start, end] 连续整数区间切成 n 段（尽量均衡），返回 [(cs, ce), ...]。

    若区间长度 < n，则多余段 size=0 被跳过（实际 worker 数自动减少）。
    """
    total = end - start + 1
    base, extra = divmod(total, n)
    chunks = []
    cur = start
    for i in range(n):
        size = base + (1 if i < extra else 0)
        if size <= 0:
            continue
        chunks.append((cur, cur + size - 1))
        cur += size
    return chunks


def _process_chunk(args):
    """worker：处理 [t_start, t_end] 连续时间步，写独立块文件（仅数据行，无表头），
    返回各阶段耗时 dict。

    大对象从模块级 _G 读取（fork COW 共享，零序列化）。sweep 计数是 t 的纯函数：
    先结算 t_start 之前的事件，再逐步应用 t 处事件，与串行版逐字节一致。
    """
    chunk_idx, t_start, t_end, chunk_csv_path = args
    battle_scene = _G['battle_scene']
    radar_keys = _G['radar_keys']
    sat_keys = _G['sat_keys']
    all_equips = _G['all_equips']
    equip_ordered = _G['equip_ordered']
    equip_events = _G['equip_events']
    target_key_to_idx = _G['target_key_to_idx']
    obs_builder = _G['obs_builder']
    action_mapper = _G['action_mapper']

    # 初始化 counts：结算 t_start 之前的事件（保证本块首步激活集合正确）
    counts = {eid: np.zeros(len(equip_ordered[eid]), dtype=np.int32) for eid in all_equips}
    for eid in all_equips:
        for tm, lst in equip_events[eid].items():
            if tm < t_start:
                for (i, d) in lst:
                    counts[eid][i] += d

    # ── 雷达静态参数预索引（B 段向量化：一次广播 geodetic2aer 替代逐雷达标量调用） ──
    _radar_infos = [battle_scene.dict_radar_id_info[rid] for rid in radar_keys]
    radar_lats = np.array([r.latitude for r in _radar_infos], dtype=np.float64)
    radar_lons = np.array([r.longitude for r in _radar_infos], dtype=np.float64)
    radar_alts = np.array([r.altitude for r in _radar_infos], dtype=np.float64)
    radar_azi_min = np.array([r.azi_min for r in _radar_infos], dtype=np.float64)
    radar_azi_max = np.array([r.azi_max for r in _radar_infos], dtype=np.float64)
    radar_ele_min = np.array([r.ele_min for r in _radar_infos], dtype=np.float64)
    radar_ele_max = np.array([r.ele_max for r in _radar_infos], dtype=np.float64)
    radar_range_min = np.array([r.range_min for r in _radar_infos], dtype=np.float64)
    radar_range_max = np.array([r.range_max for r in _radar_infos], dtype=np.float64)

    phase = defaultdict(float)
    last_heartbeat = time.time()
    with open(chunk_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for t in range(t_start, t_end + 1):
            # 心跳：每 30s 打一次本块进度（worker 编号 + 局部百分比），用于判活/判卡
            now = time.time()
            if now - last_heartbeat >= 30.0:
                last_heartbeat = now
                done = t - t_start
                total = t_end - t_start + 1
                pct = done / max(1, total) * 100.0
                print(f"[ExpertData] worker{chunk_idx} 进度 t={t} ({pct:.1f}%)", flush=True)

            raw_obs = AgentObservation()
            raw_obs.current_time = float(t)
            alive_targets = []

            # 推进覆盖区间 sweep：应用时刻 t 的启停事件，更新各装备的激活目标集合
            for eid in all_equips:
                for (i, d) in equip_events[eid].get(t, []):
                    counts[eid][i] += d

            _tA = time.time()
            # A. 填充目标航迹（每目标新建 trk，避免共享引用污染）
            for tgt_id, target_info in battle_scene.dict_target_id_info.items():
                if t in target_info.dict_target_traj_pt_info:
                    pt = target_info.dict_target_traj_pt_info[t]
                    trk = SystemTrackBase(
                        detect_time=int(t),
                        str_system_track_no=tgt_id,
                        longitude=pt.longitude,
                        latitude=pt.latitude,
                        altitude=pt.altitude,
                        ecf_x=pt.ecf_x,
                        ecf_y=pt.ecf_y,
                        ecf_z=pt.ecf_z,
                        ecf_vx=pt.ecf_vx,
                        ecf_vy=pt.ecf_vy,
                        ecf_vz=pt.ecf_vz,
                        rcs=pt.rcs,
                        type=pt.type,
                    )
                    raw_obs.dict_system_track[tgt_id] = trk
                    alive_targets.append(tgt_id)

            # 向量化预备：一次性取出本步所有存活目标的经纬高数组，
            # 供 B/C 段的 geodetic2aer / geodetic2ecef 批量计算复用（替代逐目标反复取数）
            if alive_targets:
                _tgt_pts = [battle_scene.dict_target_id_info[tid].dict_target_traj_pt_info[t]
                            for tid in alive_targets]
                alive_lats = np.array([pt.latitude for pt in _tgt_pts])
                alive_lons = np.array([pt.longitude for pt in _tgt_pts])
                alive_alts = np.array([pt.altitude for pt in _tgt_pts])
            else:
                alive_lats = alive_lons = alive_alts = np.array([])

            phase['A_prep'] += time.time() - _tA

            _tB = time.time()
            # B. 填充雷达状态与可视性（每雷达新建 s，避免共享引用污染）
            # 向量化：一次广播 geodetic2aer 算出全部雷达 × 存活目标的 az/el/斜距，再统一判可视
            if alive_targets:
                az_mat, el_mat, sr_mat = pm.geodetic2aer(
                    alive_lats[None, :], alive_lons[None, :], alive_alts[None, :],
                    radar_lats[:, None], radar_lons[:, None], radar_alts[:, None],
                )
                az_mat = np.asarray(az_mat); el_mat = np.asarray(el_mat); sr_mat = np.asarray(sr_mat)
                rng_km = sr_mat / 1000.0
                detect_mat = (
                    (radar_azi_min[:, None] <= az_mat) & (az_mat <= radar_azi_max[:, None]) &
                    (radar_ele_min[:, None] <= el_mat) & (el_mat <= radar_ele_max[:, None]) &
                    (radar_range_min[:, None] <= rng_km) & (rng_km <= radar_range_max[:, None])
                )
            else:
                detect_mat = None

            for r_idx, radar_id in enumerate(radar_keys):
                radar_info = battle_scene.dict_radar_id_info[radar_id]
                s = EquipmentState()
                s.time = int(t)
                s.str_equip_id = radar_id
                s.latitude = radar_info.latitude
                s.longitude = radar_info.longitude
                s.altitude = radar_info.altitude
                s.range_min, s.range_max = radar_info.range_min, radar_info.range_max
                s.azi_min, s.azi_max = radar_info.azi_min, radar_info.azi_max
                s.ele_min, s.ele_max = radar_info.ele_min, radar_info.ele_max
                s.azi_pointing = (radar_info.azi_min + radar_info.azi_max) / 2
                s.ele_pointing = (radar_info.ele_min + radar_info.ele_max) / 2
                s.type = 1
                s.track_num_max = radar_info.track_num_max
                s.lst_track_no = [equip_ordered[radar_id][i]
                                  for i, c in enumerate(counts[radar_id]) if c > 0]

                s.residual_track_num = max(0, s.track_num_max - len(s.lst_track_no))
                raw_obs.dict_equip_state[radar_id] = s

                raw_obs.dict_detection_result[radar_id] = {}
                if detect_mat is not None:
                    for tgt_id, flag in zip(alive_targets, detect_mat[r_idx]):
                        dr = EquipmentToTargetDetectionResult()
                        dr.str_equip_id = radar_id
                        dr.str_target_id = tgt_id
                        dr.detectable_flag = bool(flag)
                        raw_obs.dict_detection_result[radar_id][tgt_id] = dr

            phase['B_radar'] += time.time() - _tB

            _tC = time.time()
            # C. 填充卫星状态与可视性（复刻 sim/client.py step_forward 的卫星逻辑）
            # 目标 ECEF 只算一次（原来在每颗卫星循环内重复计算 len(sat_keys) 次）
            if alive_targets:
                _tx, _ty, _tz = pm.geodetic2ecef(alive_lats, alive_lons, alive_alts)
                targets_xyz = np.column_stack((np.asarray(_tx), np.asarray(_ty), np.asarray(_tz)))  # (N, 3)
                earth_center = np.array([0.0, 0.0, 0.0])
            else:
                targets_xyz = None

            for sat_id in sat_keys:
                sat_info = battle_scene.dict_satellite_id_info.get(sat_id)
                if sat_info is None:
                    continue

                s = EquipmentState()
                s.time = int(t)
                s.str_equip_id = sat_id

                if t in sat_info.dict_satellite_traj_pt_info:
                    sat_pt = sat_info.dict_satellite_traj_pt_info[t]
                    s.longitude = sat_pt.longitude
                    s.latitude = sat_pt.latitude
                    s.altitude = sat_pt.altitude

                s.range_min = 0.0
                s.range_max = 36000.0
                s.azi_min = sat_info.azi_min
                s.azi_max = sat_info.azi_max
                s.ele_min = sat_info.ele_min
                s.ele_max = sat_info.ele_max
                s.azi_pointing = (sat_info.azi_min + sat_info.azi_max) / 2
                s.ele_pointing = (sat_info.ele_min + sat_info.ele_max) / 2
                s.type = 2
                s.track_num_max = sat_info.track_num_max
                s.lst_track_no = [equip_ordered[sat_id][i]
                                  for i, c in enumerate(counts[sat_id]) if c > 0]

                s.residual_track_num = max(0, s.track_num_max - len(s.lst_track_no))
                raw_obs.dict_equip_state[sat_id] = s

                raw_obs.dict_detection_result[sat_id] = {}
                if alive_targets:
                    # 向量化：卫星位置只算一次，复用已算好的目标 ECEF 做视场夹角判定
                    sat_xyz = np.asarray(pm.geodetic2ecef(s.latitude, s.longitude, s.altitude))
                    vec_cam = earth_center - sat_xyz                       # (3,)
                    vec_target = targets_xyz - sat_xyz                     # (N, 3)
                    norm_cam = np.linalg.norm(vec_cam)
                    norm_target = np.linalg.norm(vec_target, axis=1)       # (N,)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cos_theta = (vec_target @ vec_cam) / (norm_cam * norm_target)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta_degrees = np.degrees(np.arccos(cos_theta))
                    detect = (theta_degrees < s.azi_max) & (norm_cam != 0) & (norm_target != 0)
                    for tgt_id, flag in zip(alive_targets, detect):
                        dr = EquipmentToTargetDetectionResult()
                        dr.str_equip_id = sat_id
                        dr.str_target_id = tgt_id
                        dr.detectable_flag = bool(flag)
                        raw_obs.dict_detection_result[sat_id][tgt_id] = dr

            phase['C_sat'] += time.time() - _tC

            _tD = time.time()
            # D. 产出张量
            obs_tensors = obs_builder.build_observations(raw_obs)
            avail_actions = action_mapper.build_action_mask(raw_obs)
            state_tensor = obs_builder.build_global_state(raw_obs)
            state_str = "|".join(state_tensor.astype(str).tolist())

            phase['D_tensor'] += time.time() - _tD

            _tE = time.time()
            # 向量化字符串化：obs/avail 整矩阵一次性 astype(str)，替代每行 177+22 次 Python 级
            # str() 调用。astype(str) 走 numpy C 层，且 float32 位级往返一致（已验证含 nan/inf）。
            obs_str_rows = obs_tensors.astype(str).tolist()       # (n_agents, obs_dim) → list[list[str]]
            avail_str_rows = avail_actions.astype(str).tolist()   # (n_agents, n_actions)

            # E. 提取专家动作 & 掩码校正（雷达 LD + 卫星 WX，统一按 agent_keys 顺序）
            for agent_idx, equip_id in enumerate(all_equips):
                expert_action = 0
                for i, c in enumerate(counts[equip_id]):
                    if c > 0:
                        expert_action = target_key_to_idx[equip_ordered[equip_id][i]]
                        break

                if expert_action > 0 and avail_actions[agent_idx, expert_action] == 0.0:
                    expert_action = 0  # 目标在盲区，纠正为待机

                obs_arr_str = "|".join(obs_str_rows[agent_idx])
                avail_arr_str = "|".join(avail_str_rows[agent_idx])
                # state 是每步共享的全局状态：ExpertDataset 只读每步首行，其余行写空即可，
                # 避免 225× 冗余膨胀 CSV（原来每步把同一条 state 重复写 225 遍）。
                row_state = state_str if agent_idx == 0 else ""
                writer.writerow([t, agent_idx, expert_action, obs_arr_str, row_state, avail_arr_str])
            phase['E_write'] += time.time() - _tE

    return dict(phase)


def _assemble_final(tmp_csv_path: str, chunk_paths) -> None:
    """写表头 + 按序拼接各块数据行（二进制拷贝，保留 csv 的 \\r\\n 行尾，与串行版逐字节一致）。"""
    with open(tmp_csv_path, 'wb') as out:
        buf = io.StringIO()
        csv.writer(buf).writerow(
            ['time_step', 'agent_id', 'expert_action', 'obs', 'state', 'avail_actions'])
        out.write(buf.getvalue().encode('utf-8'))
        for cp in chunk_paths:
            with open(cp, 'rb') as cf:
                out.write(cf.read())
            os.remove(cp)


def _print_phase(phase, t0, dest_csv_path, n_workers=1) -> None:
    _tot = sum(phase.values())
    if n_workers > 1:
        # 并行：phase 是各 worker 的 CPU 时间之和（≈ n_workers × 单块墙钟），非墙钟
        print(f"[ExpertData] 各worker阶段CPU耗时之和(总 {_tot:.0f}s, {n_workers} worker): " +
              " | ".join(f"{k}={v:.1f}s" for k, v in sorted(phase.items(), key=lambda x: -x[1])))
    else:
        print(f"[ExpertData] 阶段耗时(总 {_tot:.0f}s): " +
              " | ".join(f"{k}={v:.1f}s" for k, v in sorted(phase.items(), key=lambda x: -x[1])))
    print(f"[ExpertData] CSV 数据集生成完毕，已保存至 {dest_csv_path}（墙钟耗时 {time.time() - t0:.0f}s）")


def generate_expert_csv(conf, dest_csv_path: str,
                        obs_builder, action_mapper,
                        plan_id: int, scene_file_path: str,
                        radar_keys, target_keys, sat_keys):
    """数据管道：读取预案 → 抽取 RL 张量 → 保存 CSV。

    所有场景相关的积木（obs_builder, action_mapper, keys）由调用方提供，
    本函数只负责时间步遍历和 CSV 写入。

    覆盖雷达(LD)与卫星(WX)两类智能体：
      - 雷达探测用 range/azi/ele 判定（与 sim 内核一致）；
      - 卫星探测用 ECEF 视场夹角判定（复刻 sim/client.py step_forward）；
      - 专家动作从 plan_result 按装备 ID 提取，单目标索引（LD 多标签简化成单目标，
        WX 单选正好吻合）；有调度的一方正常生成，无调度的一方退化为 0（待机）。

    并行：默认 fork 进程池按连续时间步块切分（worker 数 = min(cpu, 8)，
    可用 conf.expert_csv_workers 覆盖；N=1 或无 fork 时回退串行）。
    """
    if not scene_file_path or not os.path.exists(scene_file_path):
        raise FileNotFoundError(f"[ExpertData] 致命错误：找不到场景文件 {scene_file_path}")

    print(f"[ExpertData] 开始处理场景预案: {scene_file_path}")

    # ── 读取预案 ──
    from sim.plan_file_process import PlanFileProcess
    processor = PlanFileProcess()
    plan_file_info = processor.read_plan_file_info_from_json(plan_id=plan_id, file_path=scene_file_path)
    battle_scene = plan_file_info.battle_scene
    plan_result = plan_file_info.plan_result

    # ── 专家规划结果非空校验：LD 与 WX 均无调度时拒绝，避免训练出「永远待机」 ──
    radar_set = set(radar_keys)
    sat_set = set(sat_keys or [])
    ld_has = any(eid in radar_set for eid in plan_result.dict_equip_id_target_id_detection_time)
    wx_has = any(eid in sat_set for eid in plan_result.dict_equip_id_target_id_detection_time)
    if not ld_has and not wx_has:
        raise ValueError(
            "[ExpertData] 预案无专家规划结果(splitQuduanResult)：LD 与 WX 均无调度，"
            "IL 数据将全为待机，拒绝训练。"
        )
    print(f"[ExpertData] 专家规划结果覆盖: LD={'有' if ld_has else '无'} / WX={'有' if wx_has else '无'}")

    all_times = []
    for target in battle_scene.dict_target_id_info.values():
        all_times.extend(target.dict_target_traj_pt_info.keys())
    if not all_times:
        print("[ExpertData]: 未能从 JSON 预案中提取出目标的轨迹时间")
        return

    start_time, end_time = int(min(all_times)), int(max(all_times))

    # 目标 ID → 动作索引（+1），替代 section E 里 O(N) 的 target_keys.index() 查找
    target_key_to_idx = {tid: i + 1 for i, tid in enumerate(target_keys)}

    # ── 覆盖区间预索引（sweep line）──
    # 把 plan_result 的「装备→目标→时间段列表」转成「装备→时刻→启停事件」，替代 B/C/E 段
    # 每步对 (装备, 目标, 时间段) 的三重线性扫描。区间 [value_min, value_max] 覆盖整数时刻
    # t ⟺ t ∈ [ceil(value_min), floor(value_max)]，据此在 ceil/floor 处登记 +1/-1 事件。
    target_keys_set = set(target_keys)
    all_equips = list(radar_keys) + list(sat_keys or [])
    equip_ordered = {}   # eid -> 目标 ID 列表（plan_result .items() 顺序，已过滤到 target_keys）
    equip_events = {}    # eid -> {时刻(int): [(目标下标, 增量)]}
    for eid in all_equips:
        tdict = plan_result.dict_equip_id_target_id_detection_time.get(eid, {})
        ordered = [tgt for tgt in tdict.keys() if tgt in target_keys_set]
        equip_ordered[eid] = ordered
        events = defaultdict(list)
        for i, tgt in enumerate(ordered):
            for tr in tdict[tgt]:
                st = math.ceil(tr.time_range.value_min)
                en = math.floor(tr.time_range.value_max)
                if st <= en:
                    events[st].append((i, +1))
                    events[en + 1].append((i, -1))
        equip_events[eid] = dict(events)

    # ── 填充共享只读全局（fork 前），供 worker 经 COW 继承 ──
    _G.update({
        'battle_scene': battle_scene,
        'radar_keys': list(radar_keys),
        'sat_keys': list(sat_keys or []),
        'all_equips': all_equips,
        'equip_ordered': equip_ordered,
        'equip_events': equip_events,
        'target_key_to_idx': target_key_to_idx,
        'obs_builder': obs_builder,
        'action_mapper': action_mapper,
    })

    # ── 决定 worker 数 ──
    n_workers = int(getattr(conf, 'expert_csv_workers', 0) or 0)
    if n_workers <= 0:
        n_workers = min(os.cpu_count() or 1, 8)
    n_workers = max(1, n_workers)
    use_parallel = n_workers > 1 and _fork_available()

    # ── 分块（连续时间步）──
    chunks = _split_range(start_time, end_time, n_workers if use_parallel else 1)
    n_chunks = len(chunks)

    _dir = os.path.dirname(dest_csv_path)
    if _dir:
        os.makedirs(_dir, exist_ok=True)
    tmp_csv_path = dest_csv_path + '.tmp'
    chunk_paths = [f"{tmp_csv_path}.{i}" for i in range(n_chunks)]
    t0 = time.time()

    if use_parallel:
        print(f"[ExpertData] 并行生成: {n_chunks} 个时间步块 × "
              f"[{start_time},{end_time}]（fork 进程池）")
        tasks = [(i, cs, ce, chunk_paths[i]) for i, (cs, ce) in enumerate(chunks)]
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(processes=n_chunks) as pool:
            phase_list = pool.map(_process_chunk, tasks)
    else:
        # 串行回退：单块在本进程处理（与并行版共用同一 worker 函数，输出逐字节一致）
        print(f"[ExpertData] 串行生成: [{start_time},{end_time}]")
        phase_list = [_process_chunk((0, start_time, end_time, chunk_paths[0]))]

    # ── 按序拼接：表头 + 各块数据行 → 原子替换 ──
    _assemble_final(tmp_csv_path, chunk_paths)
    os.replace(tmp_csv_path, dest_csv_path)  # 原子替换：dest 始终是完整文件，中途中断只留 .tmp

    phase = defaultdict(float)
    for p in phase_list:
        for k, v in p.items():
            phase[k] += v
    _print_phase(phase, t0, dest_csv_path, n_workers=n_chunks if use_parallel else 1)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    print("================  启动专家数据生成流水线 ================")

    test_task_id = "ILtest867"
    local_scene_path = os.path.join(str(_project_root), "scenarios", test_task_id, "scene.json")
    dest_csv_path = os.path.join(str(_project_root), "data", test_task_id, "expert_data.csv")

    if not os.path.exists(local_scene_path):
        print(f"找不到场景文件 {local_scene_path}")
        sys.exit(1)

    class DummyConf:
        pass

    mock_conf = DummyConf()
    mock_conf.plan_id = 867
    mock_conf.local_scene_path = local_scene_path

    try:
        generate_expert_csv(mock_conf, dest_csv_path)
    except Exception as e:
        print(f"生成 CSV 失败: {e}")
