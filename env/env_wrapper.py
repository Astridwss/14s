from sim import TrainingEnv
from .adapter import ScenarioAdapter

class GroupedEnvWrapper:
    def __init__(self, conf):
        self.conf = conf
        
        # 1. 引擎与配置加载
        self._sim = TrainingEnv()
        plan_id = getattr(self.conf, 'plan_id', 867) 
        local_scene_path = getattr(self.conf, 'local_scene_path', 'E:\webace_2026\14s\code\wangss-dev\scenarios\RLtest1\scene.json')

        self._sim.load_battle_scene(plan_id, local_scene_path)

        self._max_episode_steps = self.conf.max_episode_steps
        self._step_count = 0
        
        # 2. 挂载无状态特征转换枢纽
        self.adapter = ScenarioAdapter(self.conf)
        
        self._current_time = 0 

    def reset(self):
        self._step_count = 0
        
        # 1. 引擎重置，吐出第 0 帧画面
        raw_obs = self._sim.reset()
        
        # 获取真实
        self._current_time = raw_obs.current_time 
        
        avail_actions = self.adapter.extract_action_masks(raw_obs)
        obs = self.adapter.extract_observations(raw_obs)
        state = self.adapter.extract_global_state(raw_obs)

        return obs, {"avail_actions": avail_actions, "state": state, "raw_obs": raw_obs} 

    def step(self, actions):
        self._step_count += 1
        
        # 当前记录的时间，为即将下达的指令打上时间戳
        action_time = self._current_time
        
        agent_actions_list = self.adapter.discrete_actions_to_agent_action(list(actions), action_time)

        # 2. 把打好时间戳的指令发给仿真层推演，拿回下一帧数据
        raw_obs, sim_terminated = self._sim.step_forward(agent_actions_list)
        
        self._current_time = raw_obs.current_time
        
        reward = self._sim.generate_reward()
        
        terminated = bool(sim_terminated)
        truncated = bool(self._step_count >= self._max_episode_steps)

        avail_actions = self.adapter.extract_action_masks(raw_obs)
        obs = self.adapter.extract_observations(raw_obs)
        state = self.adapter.extract_global_state(raw_obs)

        info = {
            "avail_actions": avail_actions,
            "state": state,
            "raw_obs": raw_obs,
            "agent_actions_list": agent_actions_list,
            "action_time": action_time 
        }
        
        return obs, reward, terminated, truncated, info