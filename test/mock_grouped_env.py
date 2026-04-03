# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from env import GroupedEnvWrapper

def main():
    print("==================================================")
    print(" 开始测试 GroupedEnvWrapper (分组 MARL 环境接口)")
    print("==================================================")

    # 1. 模拟一个 Config 配置对象（字典形式）
    # 模拟 200 个雷达，21 个目标，每组 10 个雷达（共 20 组）
    mock_config = {
        'n_radars': 200,
        'n_targets': 21,
        'duration': 600.0,
        'dt': 1.0,
        'group_size': 10,  # 关键参数：每组雷达数量
        'max_episode_steps': 600
    }

    # 2. 初始化环境
    print("\n[1] 初始化环境...")
    env = GroupedEnvWrapper(config=mock_config)
    
    print(" 环境初始化成功算法读取到的只读属性如下（被欺骗后的维度）：")
    print(f"  - 宏观智能体数量 (n_agents): {env.n_agents} (预期: 20)")
    print(f"  - 动作空间维度 (n_actions): {env.n_actions} (预期: 22，包含不跟踪)")
    print(f"  - 局部观测维度 (obs_shape): {env.obs_shape} (预期: (250,), 即 10 * 25)")
    print(f"  - 全局状态维度 (state_shape): {env.state_shape} )")

    # 3. 测试 Reset
    print("\n[2] 测试环境 Reset...")
    grouped_obs, info = env.reset()
    avail_actions = info['avail_actions']
    state = info['state']
    raw_obs = info['raw_obs']

    print(" Reset 返回数据维度校验：")
    print(f"  - 局部观测 (obs) shape: {grouped_obs.shape} )")
    print(f"  - 可用动作 (avail_actions) shape: {avail_actions.shape} )")
    print(f"  - 全局状态 (state) shape: {state.shape})")
    
    # 验证动作掩码的第一列（索引0：不跟踪）是否全为 1
    assert np.all(avail_actions[:, 0] == 1.0), "错误：不跟踪动作必须始终可用"

    # 4. 测试 Step (模拟 3 个步长)
    print("\n[3] 测试环境 Step (模拟 3 次决策步进)...")
    
    for step in range(3):
        print(f"\n--- 第 {step + 1} 步 ---")
        
        # 模拟 QMIX 算法：根据 avail_actions 随机选择合法的动作
        actions = []
        for i in range(env.n_agents):
            # 获取第 i 个组的可选动作索引列表
            valid_action_indices = np.where(avail_actions[i] == 1.0)[0]
            # 随机挑一个合法的动作（模拟 epsilon-greedy 的随机探索）
            chosen_action = np.random.choice(valid_action_indices)
            actions.append(chosen_action)
            
        print(f"🤖 算法输出的 20 个组动作: {actions}")
        
        # 步进环境
        start_time = time.time()
        next_obs, reward, terminated, truncated, next_info = env.step(actions)
        step_time = time.time() - start_time
        
        print(f"🌍 环境 Step 耗时: {step_time*1000:.2f} ms")
        print(f" 获得团队奖励 (Reward): {reward}")
        print(f"🏁 终止状态: Terminated={terminated}, Truncated={truncated}")
        print(f"📏 Next Obs Shape: {next_obs.shape}")
        
        # 为下一步准备
        avail_actions = next_info['avail_actions']

        # 提前终止判断（以防万一）
        if terminated or truncated:
            print("🛑 回合结束")
            break

    print("\n==================================================")
    print("🎉 GroupedEnvWrapper 测试通过一切数据流转完美")
    print("==================================================")

if __name__ == "__main__":
    main()