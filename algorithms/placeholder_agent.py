# -*- coding: utf-8 -*-
"""
算法层占位：随机动作智能体
"""
import numpy as np


class RandomAgent:
    """占位：随机动作，供训练管理器跑通 episode/step 循环；"""
    def __init__(self, n_agents: int, n_actions: int):
        self.n_agents = n_agents
        self.n_actions = n_actions

    def get_actions(self, obs, info=None, evaluate=False):
        actions = np.random.randint(0, self.n_actions, size=self.n_agents).tolist()
        return actions, None, None


def get_agent(conf):
    """根据配置构造当前算法。"""
    n_agents = getattr(conf, 'n_agents', 1)
    n_actions = getattr(conf, 'n_actions', 1)
    return RandomAgent(n_agents=n_agents, n_actions=n_actions)
