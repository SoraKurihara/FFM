import os
import random

import numpy as np
import yaml

from model.ffm_learning_core import FloorFieldModel


def get_learning_dir(learning_id="Qlearning1", base_dir="output/logs"):
    learning_dir = os.path.join(base_dir, learning_id)
    os.makedirs(learning_dir, exist_ok=True)
    return learning_dir

def compute_beta(episode_step):
    if episode_step <= 500:
        return 1.0
    elif episode_step <= 1500:
        return 1.0 - (episode_step - 500) / 1000.0
    else:
        return 0.0

def main():
    learning_id = "Qlearning1"  # ← ここを変えるだけでディレクトリ管理
    learning_dir = get_learning_dir(learning_id)
    save_config = os.path.join(learning_dir, "run_config_used.yaml")

    # 設定読み込み
    with open("config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # シード固定
    seed = config.get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # データ読み込み
    map_array = np.load(config["map"])
    sff_path = config["sff"]
    N = config["N"]
    params = config["params"]

    # モデル初期化
    model = FloorFieldModel(map_array, sff_path, N, params)
    model.Q = {}     # Qテーブル
    model.alpha = 0.1
    model.gamma = 0.9

    num_episodes = 10

    for episode in range(num_episodes):
        model.reset()
        step = 0
        episode_log = []

        while model.positions.shape[0] > 0:
            beta = compute_beta(step)
            model.step(beta)
            episode_log.append(np.copy(model.positions))
            step += 1

            if step % 100 == 0:
                print(f"[Episode {episode}] Step {step}, Remaining: {model.positions.shape[0]}, beta={beta:.3f}")

        # 各エピソードごとに保存
        np.save(os.path.join(learning_dir, f"episode_{episode}.npy"),
                np.array(episode_log, dtype=object))
        print(f"Episode {episode} finished in {step} steps and saved.")

    # 設定ファイルを保存
    with open(save_config, "w") as f:
        yaml.safe_dump(config, f)

    print(f"\nTraining finished after {num_episodes} episodes.")
    print(f"Results saved in directory: {learning_dir}")

if __name__ == "__main__":
    main()
