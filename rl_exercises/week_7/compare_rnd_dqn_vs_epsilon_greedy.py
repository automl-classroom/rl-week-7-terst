from collections import defaultdict

import gymnasium as gym
import numpy as np
import pandas as pd
from rl_exercises.week_3.epsilon_greedy_policy import EpsilonGreedyPolicy
from rl_exercises.week_4.dqn import set_seed
from rl_exercises.week_7.rnd_dqn import RNDDQNAgent


def run_rnd_dqn(env_name, num_episodes, seed=42, num_frames=10000):
    env = gym.make(env_name)
    set_seed(env, seed)
    agent = RNDDQNAgent(env, seed=seed)
    agent.train(num_frames)
    # Read rewards from CSV
    df = pd.read_csv(f"training_data_seed_{seed}.csv")
    rewards = df["rewards"].values[-num_episodes:]
    return rewards


def run_epsilon_greedy_qlearning(
    env_name, num_episodes, seed=42, alpha=0.1, gamma=0.99, epsilon=0.1
):
    env = gym.make(env_name)
    set_seed(env, seed)
    policy = EpsilonGreedyPolicy(env, epsilon, seed)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episode_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = policy(Q, tuple(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            best_next = np.max(Q[tuple(next_state)])
            Q[tuple(state)][action] += alpha * (
                reward + gamma * best_next - Q[tuple(state)][action]
            )
            state = next_state
            total_reward += reward
        episode_rewards.append(total_reward)
    return episode_rewards


def main():
    env_name = "LunarLander-v3"
    num_episodes = 15
    seed = 1
    print("Training RND DQN agent...")
    rnd_rewards = run_rnd_dqn(env_name, num_episodes, seed)
    print("Training epsilon-greedy Q-learning agent...")
    eps_rewards = run_epsilon_greedy_qlearning(env_name, num_episodes, seed)
    avg_rnd = np.mean(rnd_rewards)
    avg_eps = np.mean(eps_rewards)
    print(f"RND DQN avg reward: {avg_rnd}")
    print(f"Epsilon Greedy avg reward: {avg_eps}")
    if avg_rnd >= avg_eps - 5:
        print("RND DQN performs at least as well as epsilon greedy.")
    else:
        print("Epsilon greedy outperformed RND DQN.")


if __name__ == "__main__":
    main()
