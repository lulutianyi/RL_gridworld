import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld, TrainConfig, ALGORITHMS, args


def smooth_curve(values, window_size: int = 10):
    if window_size <= 1 or len(values) < window_size:
        return np.arange(len(values)), np.array(values)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    x = np.arange(window_size - 1, len(values))
    return x, smoothed


def main():
    env = GridWorld()
    config = TrainConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        episodes=args.episodes,
    )

    results = []
    for name, trainer in ALGORITHMS.items():
        print(f"训练算法: {name}")
        result = trainer(env, config)
        results.append(result)

    plt.figure(figsize=(8, 5))
    for result in results:
        x, y = smooth_curve(result.episode_returns, window_size=10)
        plt.plot(x, y, label=result.name)

    plt.xlabel("Episode")
    plt.ylabel("Return (sum of rewards)")
    plt.title("Convergence of Different Training Methods")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("convergence_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()

