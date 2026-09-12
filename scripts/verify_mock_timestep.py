# -*- coding: utf-8 -*-
"""验证 mock 场景在 time_step=5 下的运行时行为：目标/卫星是否命中轨迹点 + 终止步数。"""
import itertools
import sys

from sim import TrainingEnv


def main(scene_path: str = "mock_scene_200r_50s_21t.json", time_step: int = 5):
    env = TrainingEnv()
    env.load_battle_scene(9999, scene_path, time_step=time_step)
    print(f"end_time={env._end_time}s  time_step={env._time_step}s  "
          f"radars={len(env._dict_radar_info)} satellites={len(env._dict_satellite_info)} "
          f"targets={len(env._dict_target_info)}")

    obs = env.reset()
    print(f"reset: current_time={obs.current_time} "
          f"tracks={len(obs.dict_system_track)} equips={len(obs.dict_equip_state)}")

    for i in itertools.count(1):
        obs, done = env.step_forward([])
        if i in (1, 2, 3, 520, 521, 522):
            print(f"step {i}: current_time={obs.current_time} "
                  f"tracks={len(obs.dict_system_track)} equips={len(obs.dict_equip_state)} "
                  f"done={done}")
        if done:
            print(f"-> 自然终止于 step {i} (current_time={obs.current_time})")
            break
        if i > 600:
            print("-> 600 步内未自然终止")
            break


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mock_scene_200r_50s_21t.json",
         int(sys.argv[2]) if len(sys.argv) > 2 else 5)
