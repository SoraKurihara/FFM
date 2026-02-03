import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml

from model.ffm_learning_core import FloorFieldModel

# -----------------------------
# Utilities
# -----------------------------

def get_learning_dir(learning_id: str = "Qlearning_covA", base_dir: str = "output/logs") -> str:
    d = os.path.join(base_dir, learning_id)
    os.makedirs(d, exist_ok=True)
    return d


def seed_everything(seed: int | None) -> None:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)


def compute_beta(episode_step: int) -> float:
    """
    External beta schedule used after warm-up:
      - step <= 50      -> 1.0
      - 50 < step <=650 -> linearly decreases to 0.0
      - step > 650      -> 0.0
    """
    if episode_step <= 50:
        return 1.0
    elif episode_step <= 650:
        return 1.0 - (episode_step - 50) / 600.0
    else:
        return 0.0


def compute_agent_count(episode: int, full_N: int) -> int:
    """First 500 episodes: ramp N by 10% every 50 episodes. After 500: use full_N."""
    if episode < 500:
        ratio = (episode // 50 + 1)  # 1..10
        return max(1, full_N * ratio // 10)
    return full_N


# -----------------------------
# Coverage Pretrain (A: empty patterns)
# -----------------------------

# from_dir constants aligned with FloorFieldModel
FROM_UP = FloorFieldModel.FROM_UP
FROM_DOWN = FloorFieldModel.FROM_DOWN
FROM_LEFT = FloorFieldModel.FROM_LEFT
FROM_RIGHT = FloorFieldModel.FROM_RIGHT
FROM_SELF = FloorFieldModel.FROM_SELF

# (dir -> delta)
DIR_TO_DXY: Dict[int, Tuple[int, int]] = {
    FROM_UP:    (-1, 0),  # src = (tx-1, ty)
    FROM_DOWN:  (1, 0),   # src = (tx+1, ty)
    FROM_LEFT:  (0, -1),  # src = (tx, ty-1)
    FROM_RIGHT: (0, 1),   # src = (tx, ty+1)
    FROM_SELF:  (0, 0),   # src = (tx, ty)
}


def iter_free_targets(map_array: np.ndarray) -> Iterable[Tuple[int, int]]:
    """Yield all free-cell targets T (map==0)."""
    xs, ys = np.where(map_array == 0)
    for i in range(len(xs)):
        yield int(xs[i]), int(ys[i])


def valid_from_dirs_for_target(map_array: np.ndarray, tx: int, ty: int) -> List[int]:
    H, W = map_array.shape
    valids: List[int] = []
    for a_idx, (dx, dy) in DIR_TO_DXY.items():
        sx, sy = tx + dx, ty + dy
        if a_idx == FROM_SELF:
            # STOP always可（ただしT自体がfreeであることは呼び出し側で保証）
            valids.append(a_idx)
            continue
        if 0 <= sx < H and 0 <= sy < W and map_array[sx, sy] == 0:
            valids.append(a_idx)
    return valids


def force_first_step_and_roll(
    map_array: np.ndarray,
    sff_path: str,
    params: Dict,
    shared_Q: dict,
    T: Tuple[int, int],
    from_dir: int,
    step_buffer: int = 10,
    save_episode_path: str | None = None,
) -> None:
    """Run one mini-episode for target T with given from_dir.
    First action is forced (teacher-forced) then roll normally with beta=1.0 until exit or cap.
    Updates shared_Q in-place via model's MC backup.
    """
    H, W = map_array.shape
    tx, ty = T
    dx, dy = DIR_TO_DXY[from_dir]
    sx, sy = tx + dx, ty + dy

    # Safety: preconditions (caller should ensure these, but we guard lightly)
    if map_array[tx, ty] != 0:
        return  # target must be free
    if from_dir != FROM_SELF and not (0 <= sx < H and 0 <= sy < W and map_array[sx, sy] == 0):
        return

    model = FloorFieldModel(map_array, sff_path, N=1, params=params)
    model.alpha = params.get("alpha", 0.1)
    model.gamma = params.get("gamma", 0.99)
    model.Q = shared_Q  # share

    # Reset & place agent at src
    model.reset()
    model.positions[0, 0] = sx
    model.positions[0, 1] = sy

    # Build occupancy (only self at src)
    occ_grid = np.zeros((H, W), dtype=np.bool_)
    occ_grid[sx, sy] = True

    # Build target-centric state key at T and append first (forced) transition
    combined = model._combined3x3_at_target(tx, ty, occ_grid, (sx, sy))  # type: ignore (private access ok)
    state_key = (combined.tobytes(), model._block_index(tx, ty))  # type: ignore
    model._ensure_qvec(state_key)  # type: ignore

    step_pen = float(model.params["step_penalty"])
    stop_pen = float(model.params["stop_penalty"])

    if from_dir == FROM_SELF:
        reward = -stop_pen
        # no movement, no DFF
    else:
        reward = -step_pen
        # apply DFF at src and move to target
        model.dff[sx, sy] += 1.0
        model.positions[0, 0] = tx
        model.positions[0, 1] = ty
        model.prev_direction[0, 0] = tx - sx
        model.prev_direction[0, 1] = ty - sy

    model.paths[0].append((state_key, from_dir, reward))

    # Dynamic step cap from SFF at start location (with small buffer)
    sff_val = float(model.sff[sx, sy]) if (0 <= sx < H and 0 <= sy < W) else 0.0
    cap = int(min(200, max(1, sff_val + step_buffer)))

    # Roll normally with beta=1.0
    steps = 0
    ep_log = []
    while model.positions.shape[0] > 0 and steps < cap:
        model.step(beta=1.0)
        ep_log.append(np.copy(model.positions))
        steps += 1

    if model.positions.shape[0] > 0:
        # timeout finish (will MC-backup inside)
        model.finalize_timeouts()

    # Optional save (disabled by default for speed & disk)
    if save_episode_path is not None:
        np.save(save_episode_path, np.array(ep_log, dtype=object))


def coverage_pretrain_empty(
    map_array: np.ndarray,
    sff_path: str,
    params: Dict,
    shared_Q: dict,
    shuffle: bool = True,
    save_dir: str | None = None,
) -> None:
    """Run empty-state coverage pretraining (A):
    For each free target T, run mini-episodes for all valid FROM_* plus STOP.
    """
    targets = list(iter_free_targets(map_array))
    if shuffle:
        random.shuffle(targets)

    total = 0
    # Pre-count total patterns for nicer progress output
    for (tx, ty) in targets:
        total += len((map_array, tx, ty))

    done = 0
    print(f"\n🧭 Coverage pretrain (empty states): {total} patterns to run…")
    for (tx, ty) in targets:
        dirs = valid_from_dirs_for_target(map_array, tx, ty)
        random.shuffle(dirs)
        for a_idx in dirs:
            out_path = None
            if save_dir is not None and (done % 5000 == 0):
                # sparsely save a sample for sanity check
                out_path = os.path.join(save_dir, f"preA_T{tx}_{ty}_a{a_idx}.npy")
            force_first_step_and_roll(
                map_array=map_array,
                sff_path=sff_path,
                params=params,
                shared_Q=shared_Q,
                T=(tx, ty),
                from_dir=a_idx,
                step_buffer=10,
                save_episode_path=out_path,
            )
            done += 1
            if done % 10000 == 0:
                print(f"  … {done}/{total} patterns done")
    print(f"✅ Coverage pretrain finished: {done}/{total} patterns.")


# -----------------------------
# Standard training episodes
# -----------------------------

def run_episode(
    map_array: np.ndarray,
    sff_path: str,
    params: Dict,
    N: int,
    beta: float,
    shared_Q: dict,
    alpha: float = 0.1,
    gamma: float = 0.99,
    log_interval: int = 100,
):
    model = FloorFieldModel(map_array, sff_path, N, params)
    model.alpha = alpha
    model.gamma = gamma
    model.Q = shared_Q

    model.reset()
    step = 0
    episode_log = []

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

def main():
    learning_id = "Qlearning_covA_v3"
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
    params = dict(config.get("params", {}))

    # Let params carry alpha/gamma for pretrain helper (optional)
    params.setdefault("alpha", 0.1)
    params.setdefault("gamma", 0.99)

    # Shared Q across all runs
    shared_Q: dict = {}

    # -------------------------
    # A) Coverage pretrain (empty states, all patterns)
    # -------------------------
    cov_dir = os.path.join(learning_dir, "coverage_pretrain")
    os.makedirs(cov_dir, exist_ok=True)
    coverage_pretrain_empty(
        map_array=map_array,
        sff_path=sff_path,
        params=params,
        shared_Q=shared_Q,
        shuffle=True,
        save_dir=None,   # set to cov_dir to store sparse samples
    )

    # Optional checkpoint after pretrain
    try:
        import pickle
        with open(os.path.join(learning_dir, "Q_after_pretrain.pkl"), "wb") as f:
            pickle.dump(shared_Q, f)
        print("💾 Saved Q_after_pretrain.pkl")
    except Exception as e:
        print(f"⚠️ Failed to save Q_after_pretrain.pkl: {e}")

    # -------------------------
    # B) Standard training schedule (same as現行)
    # -------------------------
    num_episodes = 1200
    print("\n👣 Warm-up: varying N for first 500 episodes (beta=1.0).")
    print("📉 After episode 500: start hybrid schedule where beta decreases to 0.0.\n")

    for episode in range(num_episodes):
        N = compute_agent_count(episode, full_N)
        if episode < 500:
            beta = 1.0
        else:
            beta = compute_beta(episode - 500)

        episode_log, steps, shared_Q = run_episode(
            map_array=map_array,
            sff_path=sff_path,
            params=params,
            N=N,
            beta=beta,
            shared_Q=shared_Q,
            alpha=params["alpha"],
            gamma=params["gamma"],
            log_interval=100,
        )

        np.save(os.path.join(learning_dir, f"episode_{episode}.npy"), episode_log)
        print(f"✅ Episode {episode} finished in {steps} steps. Saved to {learning_dir}.")

        if (episode + 1) % 50 == 0:
            try:
                import pickle
                with open(os.path.join(learning_dir, f"Q_ep{episode+1}.pkl"), "wb") as f:
                    pickle.dump(shared_Q, f)
                print(f"💾 Q checkpoint saved: Q_ep{episode+1}.pkl")
            except Exception as e:
                print(f"⚠️ Failed to save Q checkpoint at episode {episode+1}: {e}")

    # Save config actually used
    with open(save_config_path, "w") as f:
        yaml.safe_dump(config, f)

    # Final Q save via helper instance
    dummy = FloorFieldModel(map_array, sff_path, 1, params)
    dummy.Q = shared_Q
    dummy.save_Q(os.path.join(learning_dir, "Q.pkl"))

    print(f"\n🎉 Training finished after {num_episodes} episodes.")
    print(f"📂 Results saved in: {learning_dir}")


if __name__ == "__main__":
    main()
