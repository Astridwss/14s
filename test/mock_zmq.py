import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# 以 `python test/mock_zmq.py` 独立运行时，将仓库根加入 sys.path，使 services 包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zmq
from services.zmq.proto import ProtoStruct_pb2

def start_mock_frontend():
    # 🌟 修改点 1：IP 和端口必须指向 ZmsServer 的出口端 5559
    zms_server_ip = "192.168.1.51"
    zms_out_port = 5559
    connect_addr = f"tcp://{zms_server_ip}:{zms_out_port}"
    
    print(f"🚀 [伪装前端] 启动 ZMQ 监听，准备连接总线出口...")
    
    # 1. 初始化 ZMQ 上下文和 SUB 套接字
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # 2. 连接到 ZmsServer 的 5559 端口
    socket.connect(connect_addr)
    
    # 3. 设置订阅主题 (空字符串表示接收总线上的所有数据)
    # 如果你只想接收强化学习的数据，可以写成 socket.setsockopt_string(zmq.SUBSCRIBE, "12290")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f"✅ [伪装前端] 已成功连接到总线 {connect_addr}，等待接收态势数据...\n" + "="*50)

    try:
        while True:
            # 4. 接收多帧消息 (Topic 和 Payload)
            multipart_msg = socket.recv_multipart()
            
            # 通常我们发送了两帧：[topic, payload]
            if len(multipart_msg) >= 2:
                # 🌟 这里的 topic 现在应该是纯字符串 "12290"
                topic = multipart_msg[0].decode('utf-8')
                payload = multipart_msg[1]
                
                print(f"\n📥 收到主题: {topic} (数据大小: {len(payload)} bytes)")
                
                # 只有当主题是 12290 时才按我们的结构体解析
                if topic == "12290":
                    # 5. Protobuf 反序列化
                    msg = ProtoStruct_pb2.PR_I_RL_TRAINING_SITUATION_TO_FRONT()
                    try:
                        msg.ParseFromString(payload)
                        
                        # 6. 优雅地打印解析后的数据
                        print(f"️  仿真时间: {msg.CurrentTime}s | 帧类型 DataType: {msg.DataType}")
                        print(f"📡 装备数量: {len(msg.EquipPos)} | 🎯 目标数量: {len(msg.TargetPos)}")
                        
                        # 验证内部 MsgType 是否正确
                        if msg.HasField('PubCtrl') and msg.PubCtrl.HasField('MsgHeader'):
                            print(f"✅ 内部协议码 (MsgType): {msg.PubCtrl.MsgHeader.MsgType}")
                        
                        print("-" * 40)
                        
                    except Exception as e:
                        print(f"❌ 解析 Protobuf 失败: {e}")
                else:
                    print(f"⚠️ 收到其他主题的数据包，跳过解析。")
            else:
                print(f"⚠️ 收到格式不符的消息: {multipart_msg}")
                
    except KeyboardInterrupt:
        print("\n🛑 [伪装前端] 停止监听")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    start_mock_frontend()