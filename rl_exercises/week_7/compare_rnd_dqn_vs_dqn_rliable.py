import glob
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_7.rnd_dqn import RNDDQNAgent
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_sample_efficiency_curve


def run_rnd_dqn(env_name, num_episodes, seed=42, num_frames=10000):
    env = gym.make(env_name)
    set_seed(env, seed)
    agent = RNDDQNAgent(env, seed=seed)
    agent.train(num_frames)
    df = pd.read_csv(f"training_data_seed_{seed}.csv")
    rewards = df["rewards"].values[-num_episodes:]
    return rewards


def run_dqn(env_name, num_episodes, seed=42, num_frames=10000):
    env = gym.make(env_name)
    set_seed(env, seed)
    agent = DQNAgent(env, seed=seed)
    agent.train(num_frames)
    if hasattr(agent, "recent_rewards"):
        rewards = agent.recent_rewards[-num_episodes:]
    else:
        rewards = np.zeros(num_episodes)
    return rewards


def ensure_training_data(env_name, seeds, num_frames=20000):
    for seed in seeds:
        rnd_file = f"training_data_seed_{seed}.csv"
        dqn_file = f"dqn_training_data_seed_{seed}.csv"
        if not os.path.exists(rnd_file):
            print(f"Running RND DQN for seed {seed}...")
            run_rnd_dqn(env_name, num_episodes=0, seed=seed, num_frames=num_frames)
        if not os.path.exists(dqn_file):
            print(f"Running DQN for seed {seed}...")
            run_dqn(env_name, num_episodes=0, seed=seed, num_frames=num_frames)


def main():
    env_name = "LunarLander-v3"
    seeds = [0, 1, 42]  # Add or change seeds as needed
    num_frames = 10000
    ensure_training_data(env_name, seeds, num_frames)
    # Collect all available training data for RND DQN and DQN
    rnd_files = sorted(glob.glob("training_data_seed_*.csv"))
    dqn_files = sorted(glob.glob("dqn_training_data_seed_*.csv"))
    n_seeds = len(rnd_files)
    assert n_seeds > 0, "No RND DQN training data found!"
    # Read and stack rewards for RND DQN
    rnd_rewards_list, rnd_steps_list = [], []
    for f in rnd_files:
        df = pd.read_csv(f)
        rnd_rewards_list.append(df["rewards"].values)
        rnd_steps_list.append(df["steps"].values)
    # Use steps from the first seed as reference
    steps = rnd_steps_list[0] if rnd_steps_list else np.array([])
    dqn_rewards_list = []
    if dqn_files:
        dqn_rewards_list = [pd.read_csv(f)["rewards"].values for f in dqn_files]
    # Find minimum length across all reward arrays and steps
    all_lengths = [len(r) for r in rnd_rewards_list]
    if dqn_rewards_list:
        all_lengths += [len(r) for r in dqn_rewards_list]
    all_lengths.append(len(steps))
    min_len = min(all_lengths)
    # Truncate all arrays to min_len
    rnd_rewards = np.stack([r[:min_len] for r in rnd_rewards_list])
    steps = steps[:min_len]
    if dqn_rewards_list:
        dqn_rewards = np.stack([r[:min_len] for r in dqn_rewards_list])
    else:
        dqn_rewards = np.zeros_like(rnd_rewards)
    # Prepare dict for rliable
    train_scores = {"RND DQN": rnd_rewards}
    if dqn_files:
        train_scores["DQN"] = dqn_rewards

    # IQM metric
    def iqm(scores):
        return np.array(
            [
                metrics.aggregate_iqm(scores[:, eval_idx])
                for eval_idx in range(scores.shape[-1])
            ]
        )

    iqm_scores, iqm_cis = get_interval_estimates(
        train_scores,
        iqm,
        reps=2000,
    )
    # Plot sample efficiency curve
    plot_sample_efficiency_curve(
        steps,
        iqm_scores,
        iqm_cis,
        algorithms=list(train_scores.keys()),
        xlabel="Environment Steps",
        ylabel="IQM Reward",
    )
    plt.title("Sample Efficiency Curve (IQM)")
    handles, labels = plt.gca().get_legend_handles_labels()
    # Only keep unique labels (algorithms)
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()
    plt.savefig("sample_efficiency_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
