import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict

from .environment import GridWorld
from .arguments import args


@dataclass
class TrainConfig:
    alpha: float = args.alpha
    gamma: float = args.gamma
    epsilon: float = args.epsilon
    episodes: int = args.episodes


@dataclass
class TrainResult:
    name: str
    Q: np.ndarray
    episode_returns: List[float]
    episode_lengths: List[int]


def _state_to_index(state, env: GridWorld) -> int:
    x, y = state
    return y * env.env_size[0] + x


def train_q_learning(env: GridWorld, config: TrainConfig) -> TrainResult:
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    for _ in range(config.episodes):
        state, _ = env.reset()
        done = False

        G = 0.0
        steps = 0

        while not done:
            s = _state_to_index(state, env)

            if np.random.rand() < config.epsilon:
                action_id = np.random.randint(num_actions)
            else:
                action_id = np.argmax(Q[s])

            action = env.action_space[action_id]
            next_state, reward, done, _ = env.step(action)

            s_next = _state_to_index(next_state, env)

            Q[s, action_id] += config.alpha * (
                reward + config.gamma * np.max(Q[s_next]) - Q[s, action_id]
            )

            state = next_state
            G += reward
            steps += 1

        episode_returns.append(G)
        episode_lengths.append(steps)

    return TrainResult(
        name="Q-Learning",
        Q=Q,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )


def train_sarsa(env: GridWorld, config: TrainConfig) -> TrainResult:
    num_states = env.num_states
    num_actions = len(env.action_space)

    Q = np.zeros((num_states, num_actions))
    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    for _ in range(config.episodes):
        state, _ = env.reset()
        done = False

        s = _state_to_index(state, env)
        if np.random.rand() < config.epsilon:
            action_id = np.random.randint(num_actions)
        else:
            action_id = np.argmax(Q[s])

        G = 0.0
        steps = 0

        while not done:
            action = env.action_space[action_id]
            next_state, reward, done, _ = env.step(action)

            s_next = _state_to_index(next_state, env)

            if np.random.rand() < config.epsilon:
                next_action_id = np.random.randint(num_actions)
            else:
                next_action_id = np.argmax(Q[s_next])

            Q[s, action_id] += config.alpha * (
                reward + config.gamma * Q[s_next, next_action_id] - Q[s, action_id]
            )

            state = next_state
            s = s_next
            action_id = next_action_id

            G += reward
            steps += 1

        episode_returns.append(G)
        episode_lengths.append(steps)

    return TrainResult(
        name="SARSA",
        Q=Q,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )


ALGORITHMS: Dict[str, Callable[[GridWorld, TrainConfig], TrainResult]] = {
    "q_learning": train_q_learning,
    "sarsa": train_sarsa,
}

