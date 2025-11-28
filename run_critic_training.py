#!/usr/bin/env python3
"""
Critic学習の大規模実行プログラム
複数の人数パターンで大量のエピソードを実行し、学習の進捗を記録
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np

from model.ffm_ac_core import FloorFieldModel

# ==================== 設定パラメータ ====================
# 人数のパターン
# N=1は必ず実行し、その後は10刻み（10, 20, 30, ...）
N_FIXED = [1]  # 必ず実行する人数
N_START = 10  # 10刻みの開始
N_END = 100  # 10刻みの終了
N_STEP = 10  # 刻み幅

# エピソード数
EPISODES_PER_N = 1000

# 最大ステップ数
MAX_STEPS = 500

# 保存先ディレクトリ
OUTPUT_DIR = "output/logs/critic_training"

# モデルパラメータ
MODEL_PARAMS = {
    "k_S": 10,
    "k_D": 1,
    "alpha_v": 0.01,
    "gamma": 0.99,
    "exit_reward": 100.0,
    "step_penalty": -1.0,
    "collision_penalty": -1.0,
    "neighborhood": "neumann",
    "block_size": 5,  # 状態エンコーディングのブロックサイズ（粗い位置情報）
}

# マップとSFFのパス
MAP_PATH = "data/maps/simple_room.npy"
SFF_PATH = "data/sff/distance_L1.npy"
# ========================================================


def create_test_map():
    """テスト用のマップを作成（simple_room.npyがない場合）"""
    map_array = np.zeros((50, 50), dtype=np.uint8)
    map_array[0, :] = 1  # 上壁
    map_array[-1, :] = 1  # 下壁
    map_array[:, 0] = 1  # 左壁
    map_array[:, -1] = 1  # 右壁
    map_array[25, 49] = 3  # 出口（中央右端）
    return map_array


def create_test_sff(map_array):
    """テスト用のSFFを作成（distance_L1.npyがない場合）"""
    sff = np.ones(map_array.shape, dtype=np.float32) * 999
    exit_x, exit_y = 25, 49

    # L1距離場を作成
    for i in range(map_array.shape[0]):
        for j in range(map_array.shape[1]):
            if map_array[i, j] == 0 or map_array[i, j] == 3:
                sff[i, j] = abs(i - exit_x) + abs(j - exit_y)

    return sff


def run_training():
    """学習の実行"""
    print("=" * 80)
    print("Critic学習の大規模実行プログラム")
    print("=" * 80)

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # マップとSFFの読み込み
    if os.path.exists(MAP_PATH):
        map_array = np.load(MAP_PATH)
        print(f"✓ マップを読み込みました: {MAP_PATH}")
    else:
        map_array = create_test_map()
        print(f"⚠ テスト用マップを作成しました（{MAP_PATH}が見つかりません）")

    if os.path.exists(SFF_PATH):
        sff_path = SFF_PATH
        print(f"✓ SFFを読み込みました: {SFF_PATH}")
    else:
        sff = create_test_sff(map_array)
        sff_path = os.path.join(run_dir, "sff_temp.npy")
        np.save(sff_path, sff)
        print(f"⚠ テスト用SFFを作成しました（{SFF_PATH}が見つかりません）")

    # 人数のリスト（N=1 + 10刻み）
    n_list = N_FIXED + list(range(N_START, N_END + 1, N_STEP))
    total_patterns = len(n_list)

    print("\n設定:")
    print(f"  人数パターン: {n_list} = {total_patterns}パターン")
    print(f"  各パターンのエピソード数: {EPISODES_PER_N}")
    print(f"  総エピソード数: {total_patterns * EPISODES_PER_N}")
    print(f"  最大ステップ数: {MAX_STEPS}")
    print(f"  保存先: {run_dir}")
    print("=" * 80)

    # 全体の結果を記録
    all_results = []
    episode_results = []  # 全エピソードの結果
    start_time = time.time()

    # モデルを1回だけ初期化（Vテーブルを共有）
    print("\nモデルを初期化中...")
    model = FloorFieldModel(
        map_array=map_array,
        sff_path=sff_path,
        N=n_list[0],  # 初期人数（後で変更可能）
        params=MODEL_PARAMS,
    )
    print("✓ モデル初期化完了")

    # 総エピソード数のカウンター
    total_episodes = total_patterns * EPISODES_PER_N
    current_episode = 0

    # 各人数パターンで実行
    for pattern_idx, N in enumerate(n_list, 1):
        print("\n" + "=" * 80)
        print(f"人数パターン [{pattern_idx}/{total_patterns}]: N={N}")
        print("=" * 80)

        # このパターンの結果を記録
        pattern_results = {
            "N": N,
            "episodes": [],
            "v_table_sizes": [],
            "avg_steps": [],
        }

        # エピソードループ
        for episode in range(1, EPISODES_PER_N + 1):
            current_episode += 1
            episode_start = time.time()

            # 人数を設定してエピソード実行
            model.N = N
            model.reset()  # Vテーブルは保持、位置とDFFのみリセット
            steps = model.run(max_steps=MAX_STEPS)

            # 結果の記録
            v_table_size = model.get_v_table_size()
            episode_time = time.time() - episode_start

            pattern_results["episodes"].append(episode)
            pattern_results["v_table_sizes"].append(v_table_size)
            pattern_results["avg_steps"].append(steps)

            # エピソード結果を記録
            episode_results.append(
                {
                    "episode_num": current_episode,
                    "N": N,
                    "steps": steps,
                    "v_table_size": v_table_size,
                }
            )

            # 進捗表示（10エピソードごと、または最初と最後）
            if episode == 1 or episode % 10 == 0 or episode == EPISODES_PER_N:
                elapsed_total = time.time() - start_time
                progress = current_episode / total_episodes
                eta_seconds = (
                    (elapsed_total / progress) * (1 - progress)
                    if progress > 0
                    else 0
                )
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                print(
                    f"  N={N:3d} | Ep {episode:3d}/{EPISODES_PER_N} "
                    f"(全体{current_episode:4d}/{total_episodes}) | "
                    f"Steps: {steps:3d} | V-states: {v_table_size:5d} | "
                    f"Time: {episode_time:.2f}s | "
                    f"Progress: {progress*100:.1f}% | ETA: {eta_str}"
                )

        # パターンの統計情報
        avg_steps = np.mean(pattern_results["avg_steps"])
        avg_v_size = np.mean(pattern_results["v_table_sizes"])
        final_v_size = pattern_results["v_table_sizes"][-1]

        print(f"\n  パターンN={N}の統計:")
        print(f"    平均ステップ数: {avg_steps:.2f}")
        print(f"    平均V状態数: {avg_v_size:.1f}")
        print(f"    現在のV状態数: {final_v_size}")

        # 結果を全体に追加
        all_results.append(pattern_results)

    # 最終的なVテーブルを保存
    print("\n" + "=" * 80)
    print("学習完了 - Vテーブルを保存中...")
    print("=" * 80)

    v_table = model.get_v_table()
    final_v_size = len(v_table)

    # 統合Vテーブルの保存
    v_table_path = os.path.join(
        run_dir, f"V_integrated_total{total_episodes}ep.pkl"
    )
    with open(v_table_path, "wb") as f:
        pickle.dump(v_table, f)
    print(f"✓ 統合Vテーブル保存: {v_table_path}")
    print(f"  最終V状態数: {final_v_size}")

    # 価値の統計
    if len(v_table) > 0:
        values = list(v_table.values())
        print(f"  価値の範囲: [{min(values):.2f}, {max(values):.2f}]")
        print(f"  平均価値: {np.mean(values):.2f}")
        print(f"  標準偏差: {np.std(values):.2f}")

    # 全体の統計情報
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("全体の統計情報")
    print("=" * 80)
    print(f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"総エピソード数: {total_episodes}")
    print(f"平均エピソード時間: {total_time / total_episodes:.2f}秒")
    print(f"学習データ: {len(n_list)}種類の人数 × {EPISODES_PER_N}エピソード")

    # 結果の保存
    results_path = os.path.join(run_dir, "training_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "n_list": n_list,
                "episodes_per_n": EPISODES_PER_N,
                "model_params": MODEL_PARAMS,
                "results_by_n": all_results,
                "all_episodes": episode_results,  # 全エピソードの詳細
                "total_time": total_time,
                "final_v_table_size": final_v_size,
            },
            f,
        )
    print(f"\n✓ 学習結果を保存しました: {results_path}")

    # サマリーレポートの作成
    report_path = os.path.join(run_dir, "summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Critic学習の実行サマリー（統合Vテーブル）\n")
        f.write("=" * 80 + "\n")
        f.write(f"実行日時: {timestamp}\n")
        f.write(
            f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n"
        )
        f.write("\n統合Vテーブル:\n")
        f.write(f"  最終V状態数: {final_v_size}\n")
        if len(v_table) > 0:
            f.write(f"  価値の範囲: [{min(values):.2f}, {max(values):.2f}]\n")
            f.write(f"  平均価値: {np.mean(values):.2f}\n")
            f.write(f"  標準偏差: {np.std(values):.2f}\n")
        f.write("\n設定:\n")
        f.write(f"  人数パターン: {n_list}\n")
        f.write(f"  総エピソード数: {total_episodes}\n")
        f.write(f"  エピソード数/パターン: {EPISODES_PER_N}\n")
        f.write(f"  最大ステップ数: {MAX_STEPS}\n")
        f.write("\nモデルパラメータ:\n")
        f.write("  [FFMパラメータ]\n")
        f.write(f"    k_S: {MODEL_PARAMS['k_S']}\n")
        f.write(f"    k_D: {MODEL_PARAMS['k_D']}\n")
        f.write(f"    neighborhood: {MODEL_PARAMS['neighborhood']}\n")
        f.write("  [学習パラメータ]\n")
        f.write(f"    alpha_v (Critic学習率): {MODEL_PARAMS['alpha_v']}\n")
        f.write(f"    gamma (割引率): {MODEL_PARAMS['gamma']}\n")
        f.write(f"    block_size (状態ブロック): {MODEL_PARAMS['block_size']}\n")
        f.write("  [報酬設定]\n")
        f.write(f"    exit_reward: {MODEL_PARAMS['exit_reward']}\n")
        f.write(f"    step_penalty: {MODEL_PARAMS['step_penalty']}\n")
        f.write(
            f"    collision_penalty: {MODEL_PARAMS['collision_penalty']}\n"
        )
        f.write("\n人数別の結果:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            N = result["N"]
            avg_steps = np.mean(result["avg_steps"])
            v_size_start = result["v_table_sizes"][0]
            v_size_end = result["v_table_sizes"][-1]
            f.write(
                f"N={N:3d}: 平均ステップ={avg_steps:6.2f}, "
                f"V状態数 {v_size_start:5d}→{v_size_end:5d} "
                f"(+{v_size_end - v_size_start:4d})\n"
            )

    print(f"✓ サマリーレポートを保存しました: {report_path}")
    print("\n" + "=" * 80)
    print("✅ 全ての学習が完了しました！")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
