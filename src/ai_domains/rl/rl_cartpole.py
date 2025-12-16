from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def make_env(render_mode: str | None = None):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    return env


def train_cartpole(total_timesteps: int = 20_000, seed: int = 42, save_path: str | None = None):
    env = make_env()
    env.reset(seed=seed)

    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"Model saved to {save_path}")

    env.close()
    return model


def evaluate_cartpole(model: PPO, n_eval_episodes: int = 10):
    env = make_env()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    env.close()
    print(f"\n=== Evaluation over {n_eval_episodes} episodes ===")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/rl/ppo_cartpole",
        help="Eğitilen modelin kaydedileceği yol",
    )
    args = parser.parse_args()

    print(f"Training CartPole-v1 for {args.timesteps} timesteps...")
    model = train_cartpole(
        total_timesteps=args.timesteps,
        seed=args.seed,
        save_path=args.save_path,
    )

    evaluate_cartpole(model, n_eval_episodes=args.episodes)


if __name__ == "__main__":
    main()
