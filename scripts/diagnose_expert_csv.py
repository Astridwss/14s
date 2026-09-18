"""诊断脚本：单独跑 generate_expert_csv（含阶段计时），不启动训练。

用法:
    python scripts/diagnose_expert_csv.py <scene.json> <plan_id> [输出csv路径]

输出各阶段（A_prep/B_radar/C_sat/D_tensor/E_write）耗时，定位瓶颈。
"""
import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from services.scene.scene_parser import extract_entities, compute_dimensions
from services.scene.state.builder import ObservationBuilder
from services.scene.action.mapper import ActionMapper
from services.sample.expert_data import generate_expert_csv


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    scene_path = sys.argv[1]
    plan_id = 1801
    out_csv = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(scene_path)[0] + '.diag.csv'

    t0 = time.time()
    n_radars, n_sat, n_targets, radar_keys, target_keys, sat_keys = extract_entities(scene_path, plan_id)
    dims = compute_dimensions(n_radars, n_sat, n_targets)
    print(f"[Diag] 实体: 雷达={n_radars} 卫星={n_sat} 目标={n_targets} | 解析耗时 {time.time()-t0:.1f}s")

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

    class DummyConf:
        pass

    conf = DummyConf()
    conf.plan_id = plan_id
    conf.local_scene_path = scene_path

    t1 = time.time()
    generate_expert_csv(
        conf=conf, dest_csv_path=out_csv,
        obs_builder=obs_builder, action_mapper=action_mapper,
        plan_id=plan_id, scene_file_path=scene_path,
        radar_keys=radar_keys, target_keys=target_keys, sat_keys=sat_keys,
    )
    print(f"[Diag] 全流程耗时 {time.time()-t1:.0f}s（含读JSON）")


if __name__ == "__main__":
    main()