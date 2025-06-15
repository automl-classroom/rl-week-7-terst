"""
Deep Q-Learning with RND implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed


class RNDNetwork(nn.Module):
    """Simple MLP for RND embedding."""

    def __init__(self, obs_dim: int, hidden_size: int, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        obs_dim = env.observation_space.shape[0]
        self.rnd_target = RNDNetwork(obs_dim, rnd_hidden_size, rnd_n_layers)
        self.rnd_predictor = RNDNetwork(obs_dim, rnd_hidden_size, rnd_n_layers)
        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=rnd_lr)
        self.rnd_update_freq = rnd_update_freq
        self.rnd_reward_weight = rnd_reward_weight
        # Freeze target
        for p in self.rnd_target.parameters():
            p.requires_grad = False
        self.rnd_criterion = nn.MSELoss(reduction="none")

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # Get next_states from batch
        next_states = np.array([t[3] for t in training_batch], dtype=np.float32)
        next_states = torch.tensor(next_states)
        # Forward pass
        with torch.no_grad():
            target_emb = self.rnd_target(next_states)
        pred_emb = self.rnd_predictor(next_states)
        loss = self.rnd_criterion(pred_emb, target_emb).mean(dim=1).mean()
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        return loss.item()

    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            target_emb = self.rnd_target(state)
            pred_emb = self.rnd_predictor(state)
            bonus = self.rnd_criterion(pred_emb, target_emb).mean().item()
        return bonus

    def save_trajectory_snapshot(
        self, step_label: str, num_episodes: int = 5, out_dir: str = "outputs/"
    ):
        """Collect and save agent trajectories for visualization."""
        import os

        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        all_trajs = []
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            traj = [state]
            done = False
            while not done:
                action = self.predict_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                traj.append(next_state)
                state = next_state
                if done or truncated:
                    break
            all_trajs.append(np.array(traj))
        # Plot
        plt.figure(figsize=(6, 4))
        for traj in all_trajs:
            if traj.shape[1] >= 2:
                plt.plot(traj[:, 0], traj[:, 1], alpha=0.7)
            else:
                plt.plot(traj[:, 0], np.zeros_like(traj[:, 0]), alpha=0.7)
        plt.title(f"Agent Trajectories at {step_label}")
        plt.xlabel("State dim 0")
        plt.ylabel("State dim 1")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"trajectory_{step_label}.png"))
        plt.close()

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        snapshot_points = [
            int(num_frames * 0.1),
            int(num_frames * 0.5),
            int(num_frames * 1.0),
        ]
        snapshot_labels = ["early", "mid", "late"]
        snapshot_idx = 0

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # Apply RND bonus
            rnd_bonus = self.get_rnd_bonus(next_state)
            reward += self.rnd_reward_weight * rnd_bonus

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)
                if self.total_steps % self.rnd_update_freq == 0:
                    self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards)
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

            if (
                snapshot_idx < len(snapshot_points)
                and frame == snapshot_points[snapshot_idx]
            ):
                self.save_trajectory_snapshot(snapshot_labels[snapshot_idx])
                snapshot_idx += 1

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("Training complete.")
        training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards})
        training_data.to_csv(f"training_data_seed_{self.seed}.csv", index=False)


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = RNDDQNAgent(env, seed=cfg.seed)
    agent.train(cfg.train.num_frames)


if __name__ == "__main__":
    main()
