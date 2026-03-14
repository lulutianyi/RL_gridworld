import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld, TrainConfig, train_q_learning, args


def state_to_index(state, env: GridWorld) -> int:
    x, y = state
    return y * env.env_size[0] + x


def main():
    env = GridWorld()
    config = TrainConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        episodes=args.episodes,
    )

    print("开始使用 Q-Learning 训练智能体...")
    result = train_q_learning(env, config)
    print("训练完成")

    print("开始测试最优策略并绘制轨迹")
    Q = result.Q

    state, _ = env.reset()
    done = False

    while not done:
        s = state_to_index(state, env)
        action_id = int(np.argmax(Q[s]))
        action = env.action_space[action_id]
        state, reward, done, _ = env.step(action)
        env.render()

    plt.savefig("qlearning_trajectory.png", dpi=300, bbox_inches="tight")
    print("轨迹图片已保存为 qlearning_trajectory.png")
    plt.show()


if __name__ == "__main__":
    main()

