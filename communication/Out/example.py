import configparser
import socket
import struct
import zmq
import threading
import time
from datetime import datetime
from typing import Dict
import copy
import xml.etree.ElementTree as et
import numpy as np
import heapq
import pymap3d as pm
from datastruct import SystemTrackBase, SensorTrackBase, TargetInfo, TargetTrajPtInfo, SensorInfo, SensorState
import torch
from model import DQN
from guideagent import GuideAgent
import ProtoStruct_V3_0_0_Pub_pb2
import ProtoStruct_V3_0_0_1_pb2
import ProtoStruct_V3_0_0_2_pb2

# 智能引导处理
def dqn_guide_process():
    global g_dict_system_track, g_lock_system_track, g_dict_predict_trajectory, g_lock_predict_trajectory,\
        g_dict_sensor_info, g_lock_sensor_info, g_current_date, g_current_time, g_lock_current_date_time,\
        g_network_message_config, g_middleware_type

    # Step 1：初始化DQN模型和ZeroMq
    action_num = 2
    observation_num = 13
    policy_net = DQN(observation_num, action_num)
    policy_net.load_state_dict(torch.load('./ModelWeights'))
    policy_net.eval()

    print('DQN模型参数加载成功，智能引导开始')

    msg_type = 12290
    if g_middleware_type == 2:
        pub_server = g_network_message_config.get('ZeroMQ', 'PubServer')
        zmq_context = zmq.Context()
        zmq_publisher = zmq_context.socket(zmq.PUSH)
        zmq_publisher.connect(pub_server)

    # Step 2：根据预测弹道、系统航迹信息，进行模型推理，生成引导指令
    while True:
        # Step 2-1：获取引导指令生成所需的全局变量
        with g_lock_system_track:
            dict_system_track = copy.deepcopy(g_dict_system_track)

        with g_lock_predict_trajectory:
            dict_predict_trajectory = copy.deepcopy(g_dict_predict_trajectory)

        with g_lock_sensor_info:
            dict_sensor_info = copy.deepcopy(g_dict_sensor_info)

        # Step 2-2：遍历传感器信息、预测弹道信息
        for sensor_info in dict_sensor_info.values():
            dict_target_id_q_value: Dict[str, float] = {}

            for predict_trajectory in dict_predict_trajectory.values():
                # Step 2-3：根据预测弹道中的系统航迹批号，获取目标系统航迹信息
                if predict_trajectory.str_target_id not in dict_system_track:
                    continue

                system_track = dict_system_track[predict_trajectory.str_target_id]

                # Step 2-4：计算传感器对目标的可探测性
                a, e, r = pm.geodetic2aer(system_track.latitude, system_track.longitude, system_track.altitude,
                                          sensor_info.latitude, sensor_info.longitude, sensor_info.altitude)

                if sensor_info.azi_min <= a <= sensor_info.azi_max and\
                        sensor_info.ele_min <= e <= sensor_info.ele_max and\
                        sensor_info.range_min <= r / 1000.0 <= sensor_info.range_max:
                    detectable_flag = 1

                    # with open('./Results/DetectableArcs.txt', 'a', encoding='utf-8') as file:
                    #     file.write(
                    #         f"{system_track.detect_time}\t{system_track.str_system_track_no}\t{sensor_info.str_sensor_id}\t{r / 1000.0}\t{a}\t{e}\n")
                else:
                    detectable_flag = 0

                # Step 2-5：将DQN推理需要的信息转换成张量
                sensor_lon = sensor_info.longitude
                sensor_lat = sensor_info.latitude
                sensor_alt = sensor_info.altitude
                sensor_range_min = sensor_info.range_min
                sensor_range_max = sensor_info.range_max
                sensor_azi_min = sensor_info.azi_min
                sensor_azi_max = sensor_info.azi_max
                sensor_ele_min = sensor_info.ele_min
                sensor_ele_max = sensor_info.ele_max
                target_lon = system_track.longitude
                target_lat = system_track.latitude
                target_alt = system_track.altitude

                state = np.array(
                    [sensor_lon, sensor_lat, sensor_alt, sensor_range_min, sensor_range_max, sensor_azi_min,
                     sensor_azi_max, sensor_ele_min, sensor_ele_max, target_lon, target_lat, target_alt,
                     detectable_flag], dtype=np.float32)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                # Step 2-6：模型推理
                action = policy_net(state)
                action_max = action.max(1)
                # action_max = policy_net(state).max(1)
                if action_max.indices.item() == 1:
                    # 可以引导，记录网络输出的Q值
                    q_value = action_max.values.item()
                    dict_target_id_q_value[system_track.str_system_track_no] = q_value

                # with open('./Results/QValue.txt', 'a', encoding='utf-8') as file:
                #     file.write(
                #         f"{system_track.detect_time}\t{system_track.str_system_track_no}\t{sensor_info.str_sensor_id}\t{action[0, 0].item()}\t{action[0, 1].item()}\n")
                #
                # print(f"{system_track.detect_time}\t{system_track.str_system_track_no}\t{sensor_info.str_sensor_id}\t{action[0, 0].item()}\t{action[0, 1].item()}\n")

            # Step 2-7：根据传感器的最大跟踪容量，选择Q值最高的相应数量的目标，进行目标引导
            guide_num_max = sensor_info.track_num_max
            if guide_num_max <= 0:
                guide_num_max = 10

            if len(dict_target_id_q_value) <= guide_num_max:
                lst_guide_target_id = list(dict_target_id_q_value.keys())
            else:
                top_n = heapq.nlargest(guide_num_max, dict_target_id_q_value.items(), key=lambda x: x[1])
                lst_guide_target_id = [item[0] for item in top_n]

            # Step 2-8：处理结果转换为proto
            with g_lock_current_date_time:
                guide_date = g_current_date
                # guide_time = g_current_time

            for str_target_id in lst_guide_target_id:
                pro = ProtoStruct_V3_0_0_1_pb2.PR_I_TRACK_GUIDE_COMMAND()
                pro.PubCtrl.MsgHeader.MsgType = msg_type
                pro.PubCtrl.DestID.append(int(sensor_info.str_sensor_id))
                pro.GuideDate = guide_date
                pro.GuideTime = int(dict_system_track[str_target_id].detect_time)
                pro.TrackNo = int(str_target_id)
                pro.SrcStationNo = 1
                pro.DataSrcType = 1

                target_class = dict_system_track[str_target_id].type
                if target_class == 1 or target_class == 2 or target_class == 3 or target_class == 4:
                    pro.TargetClass = 2
                elif target_class == 5:
                    pro.TargetClass = 3
                elif target_class == 6:
                    pro.TargetClass = 5
                elif target_class == 7:
                    pro.TargetClass = 4
                else:
                    pro.TargetClass = 8

                # Step 2-9：结果发送至前台
                if g_middleware_type == 2:
                    str_pro = pro.SerializeToString()
                    zmq_publisher.send_multipart([str(msg_type).encode('utf-8'), str_pro])

                # Step 2-10：结果保存文件
                # with open('./Results/DQNGuideCommand.txt', 'a', encoding='utf-8') as file:
                #     file.write(
                #         f"{pro.GuideTime}\t{pro.SrcStationNo}\t{pro.TrackNo}\t{sensor_info.str_sensor_id}\n")

                print(f"{pro.GuideTime}\t{pro.SrcStationNo}\t{pro.TrackNo}\t{sensor_info.str_sensor_id}\n")

        # Step 3：休眠，1秒执行1次
        time.sleep(1)

