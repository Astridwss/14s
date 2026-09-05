import os
import time
import threading
import http.server
import socketserver
import numpy as np

# from API.main import prepare_task_context
# from API.notifier import RLTrainRequest, RLHyperparameters
# from env.env_wrapper import GroupedEnvWrapper

MOCK_DIR = "./mock_server_data"

def generate_mock_xml():
    """生成包含 200雷达 + 21目标轨迹 的大一统 XML 剧本"""
    os.makedirs(MOCK_DIR, exist_ok=True)
    xml_path = os.path.join(MOCK_DIR, "scene.json")
    
    print("1. 正在生成 [雷达+目标轨迹] 大一统假场景 XML...")
    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<Scenario>\n'
    xml_content += '  <Time start="0" end="600" step="1"/>\n'
    
    # 写入 200 个雷达节点
    for i in range(200):
        xml_content += f'  <Radar id="RADAR_{i:03d}" name="R{i}" lon="110" lat="30" alt="100" r_min="0" r_max="4000" a_min="0" a_max="360" e_min="0" e_max="90" track_num_max="10"/>\n'
        
    # 写入 21 个目标节点及轨迹
    for i in range(21):
        xml_content += f'  <Target id="TARGET_{i:03d}">\n'
        xml_content += '    <Trajectory>\n'
        # 每个目标假装有 10 个轨迹点
        for t in range(500):
            xml_content += f'      <Point time="{t}" lon="110.1" lat="30.1" alt="10000" x="100" y="200" z="300" vx="1" vy="2" vz="3" rcs="1.0" type="2"/>\n'
        xml_content += '    </Trajectory>\n'
        xml_content += '  </Target>\n'
        
    xml_content += '</Scenario>'
    
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

def start_mock_server():
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args): pass 
            
    os.chdir(MOCK_DIR)
    with socketserver.TCPServer(("", 8080), QuietHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    print("==========  启动全链路集成测试 ==========")
    
    generate_mock_xml()
    # server_thread = threading.Thread(target=start_mock_server, daemon=True)
    # server_thread.start()
    # time.sleep(1) 
    
    # test_url = "http://127.0.0.1:8080/scene.json"
    # print(f"\n2. 模拟前端下发请求，URL: {test_url}")
    
    # hypers = RLHyperparameters(show_log=True, seed=42)
    # fake_request = RLTrainRequest(
    #     task_id="TEST_TASK_999",
    #     scene_url=test_url,
    #     max_episodes=5,
    #     algorithm="qmix",
    #     hyperparameters=hypers
    # )
    
    # print("\n3. 触发 API 的 prepare_task_context 流水线 (开始下载并解析)...")
    # ready_conf = prepare_task_context(
    #     task_id="TEST_TASK_999", 
    #     request_data=fake_request, 
    #     mode="train"
    # )
    
    # print("\n管家处理完毕，检查 Config 注入结果：")
    # print(f"   - 本地剧本路径: {ready_conf.local_scene_path}")
    # print(f"   - 网络动作空间: {ready_conf.n_actions}")
    # print(f"   - 单兵观测维度: {ready_conf.obs_shape}")
    # print(f"   - 全局状态维度: {ready_conf.state_shape}")
    
    # print("\n4. 触发 GroupedEnvWrapper，底层 C++ 引擎加载物理世界...")
    # env = GroupedEnvWrapper(conf=ready_conf)
    
    # print("\n5. 尝试调用环境 reset() 获取第一帧数据...")
    # obs, info = env.reset()
    
    # print("\n🎉 大功告成全链路数据流转 100% 成功")
    # print(f"最终输出的分组观测张量形状 (n_agents, group_obs): {obs.shape}")