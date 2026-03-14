__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
'''
Specify parameters of the env
'''
from typing import Union
import numpy as np


class Args:
    # 环境相关配置
    env_size: Union[list, tuple, np.ndarray] = (10, 10)
    start_state: Union[list, tuple, np.ndarray] = (2, 2)
    target_state: Union[list, tuple, np.ndarray] = (7, 7)
    forbidden_states: list = [
        (2, 1), (3, 3), (1, 3), (3, 1),
        (4, 6), (5, 7), (7, 6),
        (4, 4), (5, 4), (6, 4),
        (6, 6),
        (9, 3), (8, 3), (9, 4), (7, 3),
    ]
    reward_target: float = 10
    reward_forbidden: float = -5
    reward_step: float = -1
    action_space: list = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
    debug: bool = False
    animation_interval: float = 0.2

    # 训练/算法相关通用配置
    alpha: float = 0.1          # 学习率
    gamma: float = 0.9          # 折扣因子
    epsilon: float = 0.1        # ε-greedy 探索率
    episodes: int = 500         # 训练轮数


args = Args()

