import numpy as np

from FFM import FloorFieldModel  # モジュール名に合わせてね


def main():
    # --- 1. データ読み込み ---
    map_array = np.load("Umeda_underground.npy")  # 0: 通路, 1: 壁
    sff_path = "Umeda_underground_Linf.npy"  # 事前に保存しておいた距離場

    # --- 2. パラメータ設定（必要なら） ---
    params = {
        "k_S": 3,
        "k_D": 1,
        "diffuse": 0.2,
        "decay": 0.2,
        "neighborhood": "moore"  # or "neumann"
    }

    # --- 3. モデル初期化 ---
    ffm = FloorFieldModel(map_array, sff_path, N=20000, params=params)

    # --- 4. 実行（保存間隔は100ステップごと） ---
    ffm.run(save_prefix="output/positions", save_interval=100)


if __name__ == "__main__":
    main()
