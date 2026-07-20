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
  "max_episodes": 30,
  "max_episode_steps": 1800,
  "algorithm": "qmix",
  "task_id": "RLtest1",
  "plan_id": "867",
  "load_dir": "",
  "scene_url": "http://127.0.0.1:8080/scene.json",
  "push_interval": 1,
  "hyperparameters": {
    "show_log": true,
    "seed": 42,
    "device": "npu",
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 2000
  }
}'

# 测试模仿学习训练启动
curl -X POST http://127.0.0.1:8000/api/v1/train/il \
-H "Content-Type: application/json" \
-d '{
  "task_id": "ILtest867", 
  "plan_id": "867",
  "load_dir": "",
  "scene_url": "http://127.0.0.1:8080/scene.json",         
  "algorithm": "drqn_il",     
  "epochs": 2,                                         
  "hyperparameters": {             
    "seq_len": 50,                   
    "learning_rate": 0.001,
    "batch_size": 32            
  }
}'

# 启动推演评估
curl -X POST http://127.0.0.1:8000/api/v1/eval/start \
-H "Content-Type: application/json" \
-d '{
  "task_id": "RLinfer",
  "plan_id": "867",
  "scene_url": "http://127.0.0.1:8080/scene.json",
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


