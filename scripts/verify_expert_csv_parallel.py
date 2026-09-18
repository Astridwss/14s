"""验证 generate_expert_csv 并行版：
  1) 串行(workers=1) 与 并行(workers=8) 输出 CSV 逐字节一致（md5 相同）
  2) 打印两者耗时，确认并行加速比

用法:
    python scripts/verify_expert_csv_parallel.py <scene.json> <plan_id>
"""
import os
import sys
import time
import hashlib

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from services.scene.scene_parser import extract_entities, compute_dimensions
from services.scene.state.builder import ObservationBuilder
from services.scene.action.mapper import ActionMapper
from services.sample.expert_data import generate_expert_csv


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def run_once(scene_path, plan_id, out_csv, workers):
    n_radars, n_sat, n_targets, radar_keys, target_keys, sat_keys = extract_entities(scene_path, plan_id)
    dims = compute_dimensions(n_radars, n_sat, n_targets)
    obs_builder = ObservationBuilder(
        agent_keys=radar_keys + sat_keys, target_keys=target_keys,
        n_agents=dims["n_agents"], n_targets=n_targets,
        n_actions=dims["n_actions"], radar_obs_dim=dims["radar_obs_dim"],
        satellite_ids=sat_keys,
    )
    action_mapper = ActionMapper(
        agent_keys=radar_keys + sat_keys, target_keys=target_keys,
        n_agents=dims["n_agents"], n_actions=dims["n_actions"],
        satellite_keys=sat_keys,
    )

    class Conf:
        pass

    conf = Conf()
    conf.plan_id = plan_id
    conf.local_scene_path = scene_path
    conf.expert_csv_workers = workers

    t0 = time.time()
    generate_expert_csv(
        conf=conf, dest_csv_path=out_csv,
        obs_builder=obs_builder, action_mapper=action_mapper,
        plan_id=plan_id, scene_file_path=scene_path,
        radar_keys=radar_keys, target_keys=target_keys, sat_keys=sat_keys,
    )
    return time.time() - t0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    scene_path = sys.argv[1]
    plan_id = int(sys.argv[2])

    base = os.path.splitext(scene_path)[0]
    serial_csv = base + '.verify_serial.csv'
    parallel_csv = base + '.verify_parallel.csv'

    print("=" * 60)
    print("串行 (workers=1)")
    t_serial = run_once(scene_path, plan_id, serial_csv, workers=1)
    print(f"[verify] 串行耗时 {t_serial:.1f}s  md5={md5(serial_csv)}")

    print("=" * 60)
    print("并行 (workers=8)")
    t_par = run_once(scene_path, plan_id, parallel_csv, workers=8)
    print(f"[verify] 并行耗时 {t_par:.1f}s  md5={md5(parallel_csv)}")

    print("=" * 60)
    same = md5(serial_csv) == md5(parallel_csv)
    print(f"加速比: {t_serial / max(t_par, 1e-6):.2f}x")
    print(f"输出逐字节一致: {'✅ 是' if same else '❌ 否'}")
    if not same:
        sys.exit(2)


if __name__ == "__main__":
    main()