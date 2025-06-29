import os
import random

import numpy as np
import yaml
from model.ffm_core import FloorFieldModel


def get_next_run_dir(base_dir="output/logs", prefix="run"):
    i = 1
    while os.path.exists(os.path.join(base_dir, f"{prefix}{i}")):
        i += 1
    run_dir = os.path.join(base_dir, f"{prefix}{i}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    # 設定読み込み
    with open("config/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # シード固定
    seed = config.get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # 保存先ディレクトリ自動作成
    run_dir = get_next_run_dir()
    save_log = os.path.join(run_dir, "positions.npy")
    save_config = os.path.join(run_dir, "run_config_used.yaml")

    # データ読み込み
    map_array = np.load(config["map"])
    sff_path = config["sff"]
    N = config["N"]
    params = config["params"]

    model = FloorFieldModel(map_array, sff_path, N, params)

    # シミュレーション実行
    positions_log = []
    step = 0
    while model.positions.shape[0] > 0:
        model.step()
        positions_log.append(np.copy(model.positions))
        step += 1
        if step % 100 == 0:
            print(f"Step {step}, Remaining: {model.positions.shape[0]}")

    # 保存
    np.save(save_log, np.array(positions_log, dtype=object))
    with open(save_config, "w") as f:
        yaml.safe_dump(config, f)

    print(f"\nSimulation finished in {step} steps.")
    print(f"Results saved in directory: {run_dir}")

if __name__ == "__main__":
    main()
