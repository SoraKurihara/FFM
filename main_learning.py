import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from model.ffm_learning_core import FloorFieldModel

# -----------------------------
# Utilities
# -----------------------------

def get_learning_dir(learning_id: str = "Qlearning12", base_dir: str = "output/logs") -> str:
    """Create and return a directory for this learning run."""
    learning_dir = os.path.join(base_dir, learning_id)
    os.makedirs(learning_dir, exist_ok=True)
    return learning_dir


def seed_everything(seed: int | None) -> None:
    """Fix random seeds (numpy & random)."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def compute_beta(episode_step: int) -> float:
    """
    External beta schedule (offset from warm-up start):
      - episode_step <= 50        -> 1.0  (保持)
      - 50 < episode_step <= 650 -> 線形に 1.0 -> 0.0 へ 600ステップで減衰
      - episode_step > 650       -> 0.0
    Note: caller passes (episode - 500), because the first 500 episodes are warm-up at β=1.0.
    """
    if episode_step <= 50:
        return 1.0
    elif episode_step <= 650:
        return 1.0 - (episode_step - 50) / 600.0
    else:
        return 0.0


def compute_agent_count(episode: int, full_N: int) -> int:
    """
    First 500 episodes: ramp N by 10% every 50 episodes.
    After 500 episodes: use full_N.
    """
    if episode < 500:
        ratio = (episode // 50 + 1)  # 1,2,...,10
        return max(1, full_N * ratio // 10)
    return full_N


# -----------------------------
# One episode runner
# -----------------------------

def run_episode(
    map_array: np.ndarray,
    sff_path: str,
    params: Dict[str, Any],
    N: int,
    beta: float,
    shared_Q: Dict[Any, np.ndarray],
    *,
    alpha: float = 0.1,
    gamma: float = 0.99,
    log_interval: int = 100,
    max_steps: int = 0,
) -> Tuple[np.ndarray, int, Dict[Any, np.ndarray]]:
    """
    Run a single episode and return (episode_log, steps, shared_Q).
    - Uses shared_Q across episodes to accumulate learning.
    - Respects beta schedule passed from the caller.
    - If max_steps > 0 and agents remain, finalize_timeouts() is called.
    """
    model = FloorFieldModel(map_array, sff_path, N, params)
    model.alpha = alpha
    model.gamma = gamma
    model.Q = shared_Q  # share Q across episodes

    model.reset()
    step = 0
    episode_log: list[np.ndarray] = []

    # main loop
    if max_steps and max_steps > 0:
        while model.positions.shape[0] > 0 and step < max_steps:
            model.step(beta)
            episode_log.append(np.copy(model.positions))
            step += 1
            if log_interval and (step % log_interval == 0):
                print(f"[beta={beta:.3f}] step={step}, remaining={model.positions.shape[0]}")
        # timeouts handling
        if model.positions.shape[0] > 0:
            model.finalize_timeouts()
    else:
        while model.positions.shape[0] > 0:
            model.step(beta)
            episode_log.append(np.copy(model.positions))
            step += 1
            if log_interval and (step % log_interval == 0):
                print(f"[beta={beta:.3f}] step={step}, remaining={model.positions.shape[0]}")

    return np.array(episode_log, dtype=object), step, model.Q


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    learning_id = "Qlearning18"  # 既存のIDを踏襲（必要なら config 側で上書き）
    learning_dir = get_learning_dir(learning_id)
    save_config_path = os.path.join(learning_dir, "run_config_used.yaml")

    # Load config
    with open("config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Seeds
    seed = config.get("seed", None)
    seed_everything(seed)

    # Data & params
    map_array = np.load(config["map"])
    sff_path = config["sff"]
    full_N = int(config["N"])
    params = dict(config.get("params", {}))  # ensure dict

    # Optional: max steps (0 = unlimited to keep previous behavior)
    max_steps = int(params.get("max_steps", config.get("max_steps", 0)))

    # Training schedule (踏襲)
    num_episodes = 1200
    shared_Q: Dict[Any, np.ndarray] = {}

    print("👣 Warm-up: varying N for first 500 episodes (beta=1.0).")
    print("📉 After episode 500: start hybrid schedule where beta decreases to 0.0 (DFF stays ON).\n")

    for episode in range(num_episodes):
        # Determine N and beta
        N = compute_agent_count(episode, full_N)
        if episode < 500:
            beta = 1.0
        else:
            beta = compute_beta(episode - 500)

        # Run one episode
        episode_log, steps, shared_Q = run_episode(
            map_array=map_array,
            sff_path=sff_path,
            params=params,
            N=N,
            beta=beta,
            shared_Q=shared_Q,
            alpha=0.1,
            gamma=0.99,
            log_interval=100,
            max_steps=max_steps,
        )

        # Save episode log
        np.save(os.path.join(learning_dir, f"episode_{episode}.npy"), episode_log)
        print(f"✅ Episode {episode} finished in {steps} steps. Saved to {learning_dir}.")

        # Optional periodic Q save (safety checkpoint)
        if (episode + 1) % 50 == 0:
            q_path = os.path.join(learning_dir, f"Q_ep{episode+1}.pkl")
            try:
                import pickle
                with open(q_path, "wb") as f:
                    pickle.dump(shared_Q, f)
                print(f"💾 Q checkpoint saved: {q_path}")
            except Exception as e:
                print(f"⚠️ Failed to save Q checkpoint at episode {episode+1}: {e}")

    # Save the config actually used
    with open(save_config_path, "w") as f:
        yaml.safe_dump(config, f)

    # Final Q save via a temp model helper (reusing class method)
    dummy = FloorFieldModel(map_array, sff_path, 1, params)
    dummy.Q = shared_Q
    dummy.save_Q(os.path.join(learning_dir, "Q.pkl"))

    print(f"\n🎉 Training finished after {num_episodes} episodes.")
    print(f"📂 Results saved in: {learning_dir}")


if __name__ == "__main__":
    main()
