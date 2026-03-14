import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld


def main():
    env = GridWorld()

    state, _ = env.reset()

    max_steps = 100
    episode_done = False
    step = 0

    while not episode_done and step < max_steps:
        action = env.action_space[np.random.randint(len(env.action_space))]
        state, reward, done, _ = env.step(action)
        env.render()
        episode_done = done
        step += 1

    plt.savefig("grid_world_trajectory.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)

    print("图片已保存为 grid_world_trajectory.png")
    input("按 Enter 键关闭窗口...")
    plt.close("all")


if __name__ == "__main__":
    main()

