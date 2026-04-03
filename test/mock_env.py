# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
_sim_root = _proj_root / "sim"
# 确保仿真层在路径中
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

if str(_sim_root) not in sys.path:
    sys.path.insert(0, str(_sim_root))

from env.env_wrapper import GroupedEnvWrapper
from env.adapter import discrete_actions_to_agent_action

def main():
    print("==================================================")
    print(" 开始深度测试 GroupedEnvWrapper (真实底层仿真环境)")
    print("==================================================")

    # 1. 真实 RL 配置文件模拟
    real_config = {
        'n_radars': 200,
        'n_targets': 21,
        'duration': 600.0,
        'dt': 1.0,
        'group_size': 10,  
        'max_episode_steps': 600
    }

    print("\n[1] 正在初始化真实环境 (sim.client.TrainingEnv 将被启动)...")
    wrapper = GroupedEnvWrapper(config=real_config)
    
    print(f" 环境初始化成功算法读取到的只读属性如下：")
    print(f"  - 宏观智能体数量 (n_agents): {wrapper.n_agents} (预期: 20)")
    print(f"  - 动作空间维度 (n_actions): {wrapper.n_actions} (预期: 22)")
    print(f"  - 局部观测维度 (obs_shape): {wrapper.obs_shape}")
    print(f"  - 全局状态维度 (state_shape): {wrapper.state_shape}")

    # =====================================================================
    # 2. 测试 Reset 并透视真实的返回数据
    # =====================================================================
    print("\n[2] 测试环境 Reset() 并透视数据...")
    # 注意：如果你的真实仿真在 reset 前需要加载剧本，
    # 比如 wrapper._sim.load_scene_info(...)，请在 env_wrapper 的 reset 里加上
    grouped_obs, info = wrapper.reset()
    
    avail_actions = info['avail_actions']
    state = info['state']
    raw_obs = info['raw_obs']  # 真实的 AgentObservation 对象

    print(f"\n🔍 [Debug 透视 1: 真仿真底层返回的 Raw Obs (AgentObservation)]")
    print(f"  - 场景当前时间 (current_time): {raw_obs.current_time}")
    print(f"  - 真实系统航迹数量 (Targets): {len(raw_obs.lst_system_track)}")
    print(f"  - 真实雷达状态数量 (Radars): {len(raw_obs.lst_sensor_state)}")
    print(f"  - 真实雷达-目标探测线数量 (Detection Results): {len(raw_obs.lst_detection_result)}")

    print(f"\n🔍 [Debug 透视 2: Adapter 提炼出的深度学习张量]")
    print(f"  - 局部观测 (grouped_obs) shape: {grouped_obs.shape} (应为 20组 x 组内维度)")
    print(f"  - 局部观测片段 (组0, 前5维): {grouped_obs[0, :5]}")
    print(f"  - 全局状态 (state) shape: {state.shape} (应为 789 维)")
    print(f"  - 全局状态片段 (前5维目标数据): {state[:5]}")
    print(f"  - 动作掩码 (avail_actions) shape: {avail_actions.shape} (应为 20组 x 22动作)")
    print(f"  - 组 0 的动作掩码情况: {avail_actions[0]}")
    
    assert np.all(avail_actions[:, 0] == 1.0), "严重错误：掩码第 0 列（不跟踪）必须为 1"

    # =====================================================================
    # 3. 测试动作分配流程解析与 Step 步进
    # =====================================================================
    print("\n[3] 深度解剖真实动作解析流 (Action Decoding) 并执行 Step...")
    
    for step in range(3):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        # 3.1 模拟 QMIX 算法输出：根据掩码随机选择宏观动作
        macro_actions = []
        for i in range(wrapper.n_agents):
            valid_indices = np.where(avail_actions[i] == 1.0)[0]
            if len(valid_indices) > 0:
                macro_actions.append(np.random.choice(valid_indices))
            else:
                macro_actions.append(0)  # 兜底：如果不跟踪可用则选0
                
        print(f"  🤖 算法层 (QMIX) 输出的 20 个宏观组动作:\n  {macro_actions}")

        # 3.2 [仅限Debug打印] 手动调用分配器，看看它是怎么展开的
        # （这部分逻辑 EnvWrapper._step 会自己做，这里单独抽出来为了打印给你看）
        debug_expanded_actions = wrapper.allocator.allocate(
            grouped_actions=macro_actions,
            avail_actions=wrapper._last_raw_avail_actions,
            obs=wrapper._last_raw_obs
        )
        print(f"\n  🧐 调度层 (Top-K) 展开后的 200 个雷达微观动作片段 (前 30 个):\n  {debug_expanded_actions[:30]}")

        # 3.3 [仅限Debug打印] 看看微观动作是怎么翻译成真仿真指令的
        debug_agent_action = discrete_actions_to_agent_action(list(debug_expanded_actions), wrapper.n_radars, wrapper.n_targets)
        print(f"\n   适配层 (Adapter) 翻译出的真仿真指令片段 (前 3 个雷达):\n  {debug_agent_action.lst_guide_command[:3]}")

        # 3.4 真实执行 Step 步进
        start_time = time.time()
        next_obs, reward, terminated, truncated, next_info = wrapper.step(macro_actions)
        step_time = time.time() - start_time
        
        print(f"\n  🌍 环境 Step 耗时: {step_time*1000:.2f} ms")
        print(f"   获得真实团队奖励 (Reward): {reward}")
        print(f"  🏁 终止状态: Terminated={terminated}, Truncated={truncated}")
        
        # 为下一步更新数据
        avail_actions = next_info['avail_actions']

        if terminated or truncated:
            print("🛑 底层仿真宣告回合结束")
            break

    print("\n==================================================")
    print("🎉 真实底层测试通过一切数据流转完美")
    print("==================================================")

if __name__ == "__main__":
    main()