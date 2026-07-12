#!/usr/bin/env python3
"""
FFM+AC（集約Actor版）の実行スクリプト。
集約pklを参照しつつ Actor・Critic 両方学習。Critic は常に 0 から学習。
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np

try:
    from model.ffm_ac_aggregated import FloorFieldModelACAggregated
except ImportError:
    from ffm_ac_aggregated import FloorFieldModelACAggregated

# ==================== 設定 ====================
# 集約Actor pkl（inspect_actor_data.py の出力）
AGGREGATED_H_PATH = os.path.join(
    "output", "logs", "unified_actor_training", "H_actor_aggregated_3maps_by_distm_angle.pkl",
)
# Critic は常に 0 から学習（事前学習 V は使わない）
PRETRAINED_V_PATH = None

MAP_PATH = "data/maps/excel_map_009.npy"
SFF_PATH = "data/sff/excel_map_009_L1.npy"

# マップの約8割でキリの良い人数・単一設定
FILL_RATIO = 0.8
EPISODES = 1000
# 最大ステップ数: 最小100、最大 人数*3
MAX_STEPS_MIN = 100
MAX_STEPS_PER_AGENT = 3

OUTPUT_DIR = "output/logs/ffm_ac_aggregated"

MODEL_PARAMS = {
    "k_D": 1,
    "k_A": 10,
    "alpha_v": 0.1,
    "alpha_h": 0.1,
    "gamma": 0.99,
    "exit_reward": 100.0,
    "step_penalty": -1.0,
    "collision_penalty": -1.0,
    "neighborhood": "neumann",
    "block_size": 1,
}
# ================================================


def find_exits(map_array):
    """すべての出口の座標リストを返す"""
    exits = np.argwhere(map_array == 3)
    if len(exits) == 0:
        raise ValueError("出口が見つかりません")
    # [(x1, y1), (x2, y2), ...] の形式で返す
    return [(int(row), int(col)) for row, col in exits]


def count_available_cells(map_array, exit_positions, radius):
    """複数の出口に対応した有効セル数のカウント"""
    free = np.argwhere(map_array == 0)
    if len(free) == 0:
        return 0
    
    exits_array = np.array(exit_positions)
    # 各フリーセルから全出口へのL1距離を計算
    distances = np.abs(free[:, np.newaxis, 0] - exits_array[:, 0]) + \
                np.abs(free[:, np.newaxis, 1] - exits_array[:, 1])
    # 最も近い出口までの距離を取得
    min_distances = np.min(distances, axis=1)
    
    return int(np.sum(min_distances <= radius))


def walkable_cells_and_max_radius(map_array, exit_positions):
    """歩行可能セル数と、最も遠いセルから『最寄りの出口』までのL1距離を返す"""
    walkable = np.argwhere((map_array == 0) | (map_array == 3))
    if len(walkable) == 0:
        return 0, 0
    
    exits_array = np.array(exit_positions)
    # ブロードキャストで一気に全セル⇔全出口の距離を計算
    distances = np.abs(walkable[:, np.newaxis, 0] - exits_array[:, 0]) + \
                np.abs(walkable[:, np.newaxis, 1] - exits_array[:, 1])
    
    # 各セルから一番近い出口までの距離のリスト
    min_distances = np.min(distances, axis=1)
    
    # その中で最大のものがmax_radius
    return len(walkable), int(np.max(min_distances))


def round_to_nice(n):
    """キリの良い人数に丸める（5 or 10 の刻み）。"""
    if n <= 0:
        return 0
    if n <= 10:
        return max(1, int(round(n)))
    return int(round(n / 10.0) * 10)


def main():
    print("=" * 70)
    print("FFM+AC（集約Actor）実行 - Actor・Critic 両方学習")
    print("=" * 70)

    if not os.path.isfile(AGGREGATED_H_PATH):
        print("集約Actorが見つかりません: {}".format(AGGREGATED_H_PATH))
        print("先に inspect_actor_data.py で集約pklを生成してください。")
        return

    print("集約Actor: {}".format(AGGREGATED_H_PATH))
    print("Critic: 0 から学習")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, "run_{}".format(timestamp))
    os.makedirs(run_dir, exist_ok=True)

    map_array = np.load(MAP_PATH)
    print("マップ: {}".format(MAP_PATH))
    exit_positions = find_exits(map_array)
    print("出口リスト: {}".format(exit_positions))

    available, max_radius = walkable_cells_and_max_radius(map_array, exit_positions)
    n_raw = available * FILL_RATIO
    n_agents = round_to_nice(n_raw)
    n_agents = min(n_agents, available)
    if n_agents < 1:
        n_agents = 1
    max_steps = max(MAX_STEPS_MIN, n_agents * MAX_STEPS_PER_AGENT)
    print("歩行可能セル: {} | 半径: {} | 人数(約{:.0f}%): {} | "
          "max_steps: {}\n".format(
              available, max_radius, FILL_RATIO * 100, n_agents, max_steps))

    start_time = time.time()

    model = FloorFieldModelACAggregated(
        map_array=map_array,
        sff_path=SFF_PATH,
        N=n_agents,
        aggregated_h_path=AGGREGATED_H_PATH,
        pretrained_v_path=None,
        params=MODEL_PARAMS,
    )

    steps_list = []
    remaining_list = []
    trajectory_interval = 10
    for ep in range(1, EPISODES + 1):
        model.reset(exit_pos=exit_positions, radius=max_radius)
        if ep % trajectory_interval == 0:
            steps, trajectory = model.run(
                max_steps=max_steps, return_trajectory=True
            )
            traj_path = os.path.join(
                run_dir, "trajectory_ep{:04d}.pkl".format(ep)
            )
            with open(traj_path, "wb") as f:
                pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("  軌跡保存: {} ({} ステップ)".format(traj_path, len(trajectory)))
        else:
            steps = model.run(max_steps=max_steps)
        remaining = model.positions.shape[0]
        steps_list.append(steps)
        remaining_list.append(remaining)

        if ep == 1 or ep % 1 == 0 or ep == EPISODES:
            _, v_cur, _ = model.get_v_table_size()
            _, h_cur, _ = model.get_H_table_size()
            print(
                "Ep {:3d}/{} | Steps: {:3d} | 残り: {} 人 | "
                "V: {} 状態 | H: {} 状態".format(
                    ep, EPISODES, steps, remaining, v_cur, h_cur,
                )
            )

    elapsed = time.time() - start_time
    avg_steps = np.mean(steps_list)
    avg_remaining = np.mean(remaining_list)
    print("平均ステップ数: {:.2f} | 平均残り人数: {:.2f} 人".format(
        avg_steps, avg_remaining))

    v_table = model.get_v_table()
    v_path = os.path.join(run_dir, "V_aggregated_total{}ep.pkl".format(EPISODES))
    with open(v_path, "wb") as f:
        pickle.dump(v_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Vテーブル保存: {} ({} 状態)".format(v_path, len(v_table)))

    h_table = model.get_H_table()
    h_path = os.path.join(run_dir, "H_actor_aggregated_total{}ep.pkl".format(EPISODES))
    with open(h_path, "wb") as f:
        pickle.dump(h_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Hテーブル保存: {} ({} 状態)".format(h_path, len(h_table)))

    # 残り人数データを保存（1エピソード1行）
    remaining_path = os.path.join(run_dir, "remaining_per_episode.txt")
    with open(remaining_path, "w", encoding="utf-8") as f:
        f.write("# ep\tremaining\n")
        for ep, r in enumerate(remaining_list, start=1):
            f.write("{}\t{}\n".format(ep, r))
    print("残り人数データ保存: {} ({} エピソード)".format(
        remaining_path, len(remaining_list)))

    print("総エピソード数: {} | 所要時間: {:.1f}s".format(EPISODES, elapsed))
    print("=" * 70)


if __name__ == "__main__":
    main()
