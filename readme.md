# 本地测试启动方式，测试数据可能含有实装，没有上传到仓库，还没拿到陶博士测试数据
cd ./test python /mock_server.py
cd ./API/ && python -m uvicorn main:app --reload

# 暂时取消了，后面优化补充
# 查看曲线  
tensorboard --logdir ./results
tensorboard --logdir=results --bind_all


# 暂时取消了，后面优化补充
# 态势球启动：浏览器调用/map接口，网页执行index.html-->加载离线Cesium.js--->fetch触发获取/api/v1/scene/init_data?task_id=${taskId}路由-->拿数据并渲染
http://127.0.0.1:8000/map?task_id=任务ID


# 测试强化学习训练启动

curl -X POST http://127.0.0.1:8000/api/v1/train/rl \
-H "Content-Type: application/json" \
-d '{
    "max_episodes": 3000,
    "max_episode_steps": 520,
    "group_size": 10,
    "algorithm": "qmix",
    "task_id": "rl_test_828_2",
    "plan_id": "9999",
    "load_dir": "",
    "scene_url": "C:/webace_2026/14s/code/webace-3/mock_scene_200r_50s_21t.json",
    "push_interval": 1,
    "hyperparameters": {
      "show_log": true,
      "seed": 42,
      "device": "cuda",
      "learning_rate": 0.0005,
      "gamma": 0.995,
      "batch_size": 32,
      "buffer_size": 256,
      "train_seq_len": 60,
      "update_target_params": 200,
      "grad_norm_clip": 10.0,
      "epsilon_start": 1.0,
      "epsilon_finish": 0.1,
      "epsilon_anneal_time": 400000,
      "drqn_hidden_dim": 128,
      "qmix_hidden_dim": 32,
      "hyper_hidden_dim": 128
    }
}'


# 模仿学习训练启动
curl -X POST http://7.81.21.52:8066/api/v1/train/il \
-H "Content-Type: application/json" \
-d '{
  "task_id": "ILtest867", 
  "plan_id": "1081",
  "load_dir": "",
  "scene_url": "/home/kylin/14229/back/20260906/code-0906/f18d2315-7d7d-4ad6-b00f-85aaaa68074f.json",         
  "algorithm": "drqn",     
  "epochs": 200,                                         
  "hyperparameters": {
    "seq_len": 50,
    "device": "npu",                      
    "learning_rate": 0.001,
    "batch_size": 64            
  }
}'



curl -X POST http://0.0.0.0:8000/api/v1/train/il \
-H "Content-Type: application/json" \
-d '{
  "task_id": "ILtest867", 
  "plan_id": "1081",
  "load_dir": "",
  "scene_url": "/home/kylin/14229/back/20260906/code-0906/f18d2315-7d7d-4ad6-b00f-85aaaa68074f.json",         
  "algorithm": "drqn",     
  "epochs": 200,                                         
  "hyperparameters": {
    "seq_len": 50,
    "device": "npu",                      
    "learning_rate": 0.001,
    "batch_size": 32            
  }
}'

# 启动推演评估
curl -X POST http://127.0.0.1:8000/api/v1/eval/start \
-H "Content-Type: application/json" \
-d '{
  "task_id": "RLinfer",
  "plan_id": "1081",
  "scene_url": "/home/kylin/14229/back/20260906/code-0906/f18d2315-7d7d-4ad6-b00f-85aaaa68074f.json",
  "algorithm": "qmix",
  "load_dir": "E:/webace_2026/14s/code/wangss-dev/models/ILtest867", 
  "max_episodes": 1
}'

# Baseline启动推演评估
curl -X POST http://127.0.0.1:8000/api/v1/baseline_eval/start \
-H "Content-Type: application/json" \
-d '{
  "plan_id": "1081",
  "scene_url": "/home/kylin/14229/back/20260906/code-0906/f18d2315-7d7d-4ad6-b00f-85aaaa68074f.json",
  "algorithm": "qmix",
  "load_dir": "E:/webace_2026/14s/code/wangss-dev/models/ILtest867", 
  "max_episodes": 1
}'

# 暂停训练
curl -X POST http://127.0.0.1:8000/api/v1/train/pause \
-H "Content-Type: application/json" \
-d '{
  "task_id": "RLtest1"
}'

# 恢复训练
curl -X POST http://127.0.0.1:8000/api/v1/train/resume \
-H "Content-Type: application/json" \
-d '{
  "task_id": "RLtest1"
}'

# 终止训练
curl -X POST http://127.0.0.1:8000/api/v1/train/terminate \
-H "Content-Type: application/json" \
-d '{
  "task_id": "RLtest1"
}'


┌───────────────────┬─────────────────────────────────────────────────┐
│       角色        │                     文件名                      │
├───────────────────┼─────────────────────────────────────────────────┤
│ DRQN 智能体网络   │ LD100_WX25_TARGET21_groupsize10_drqn.pkl        │
├───────────────────┼─────────────────────────────────────────────────┤
│ 标准 QMIX 混频器  │ LD100_WX25_TARGET21_groupsize10_qmix.pkl        │
├───────────────────┼─────────────────────────────────────────────────┤
│ H-QMIX 组内混频器 │ LD100_WX25_TARGET21_groupsize10_lower_mixer.pkl │
├───────────────────┼─────────────────────────────────────────────────┤
│ H-QMIX 组间混频器 │ LD100_WX25_TARGET21_groupsize10_upper_mixer.pkl │



┌──────────────┬───────────────┬─────────────────┐
│   下发类型   │ epsilon_start │ phase1_episodes │
├──────────────┼───────────────┼─────────────────┤
│ 同场景微调   │ 0.2           │ 0               │
├──────────────┼───────────────┼─────────────────┤
│ 换场景重适应 │ 0.5           │ 300             │
├──────────────┼───────────────┼─────────────────┤
│ IL→RL 热启动 │ 0.5~1.0       │ 500             │
└──────────────┴───────────────┴─────────────────┘