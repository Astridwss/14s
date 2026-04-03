# -*- coding: utf-8 -*-
"""
算法层 - 网络结构、动作选择、学习逻辑。
仅依赖：config；不依赖 env、sim_client。
向训练管理器提供：get_agent(conf)、get_actions(obs, info)、update(trajectory) 等接口。
"""
from .qmix import Agents, DRQN, QMIXNET, QMIX



# def get_agent(conf):
#     algo = getattr(conf, 'algo', 'ppo')
#     if algo == 'ppo' and GroupPPOAgents is not None:
#         return GroupPPOAgents(conf)
#     return RandomAgent(
#         n_agents=getattr(conf, 'n_agents', 1),
#         n_actions=getattr(conf, 'n_actions', 1),
#     )


__all__ = ['Agents', 'DRQN', 'QMIXNET', 'QMIX']
