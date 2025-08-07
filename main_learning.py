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
    if episode_step <= 50:
        return 1.0
    elif episode_step <= 150:
        return 1.0 - (episode_step - 50) / 100.0
    else:
        return 0.0


def main():
    learning_id = "Qlearning6"
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
    full_N = config["N"]
    params = config["params"]

    num_episodes = 650
    model = None  # 最初はまだ生成しない（Nが変わるから）
    
    shared_Q = {}  # ← これを一番最初に用意！


    for episode in range(num_episodes):
        # 最初の500エピソードは割合を変える
        if episode < 500:
            ratio = (episode // 50 + 1)
            N = full_N * ratio // 10
            beta = 1.0
        else:
            N = full_N
            beta = compute_beta(episode - 500)

        model = FloorFieldModel(map_array, sff_path, N, params)
        model.alpha = 0.1
        model.gamma = 0.9
        model.Q = shared_Q  # ← ここで共有する！！

        if episode == 0:
            print(f"👣 Initial training with varying N for first 500 episodes.")
        elif episode == 500:
            print(f"📉 Now transitioning to mixed β Q-learning (beta < 1.0)")

        model.reset()
        step = 0
        episode_log = []

        while model.positions.shape[0] > 0:
            model.step(beta)
            episode_log.append(np.copy(model.positions))
            step += 1

            if step % 100 == 0:
                print(f"[Episode {episode}] Step {step}, Remaining: {model.positions.shape[0]}, beta={beta:.3f}")

        # 保存
        np.save(os.path.join(learning_dir, f"episode_{episode}.npy"),
                np.array(episode_log, dtype=object))
        print(f"Episode {episode} finished in {step} steps and saved.")

    # 設定保存
    with open(save_config, "w") as f:
        yaml.safe_dump(config, f)

    model.save_Q(f"output/logs/{learning_id}/Q.pkl")
    print(f"\n✅ Training finished after {num_episodes} episodes.")
    print(f"📂 Results saved in directory: {learning_dir}")



if __name__ == "__main__":
    main()
