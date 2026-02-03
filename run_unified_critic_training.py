#!/usr/bin/env python3
"""
統合Critic学習の実行プログラム（初期配置を半径で制御）
- 出口から半径10, 15, 20, ... と5マス単位で増やしていく
- 各半径に対して、N=1, 10, 20, 30, ..., 100と人数を増やしていく
- 人数増加が優先（半径10マス以内の1人～100人、半径15マス以内の1人～100人、...）
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np

from model.ffm_unified import FloorFieldModelUnified

# ==================== 設定パラメータ ====================
# 半径の設定
RADIUS_START = 3  # 開始半径
RADIUS_END = 15  # 終了半径
RADIUS_STEP = 2  # 半径の刻み幅

# 人数の設定
# 人数リストは [1, 10, 20, 30, ..., N_END] となる（1だけ特別、それ以外は10の倍数）
N_START = 1  # 未使用（互換性のため残している）
N_END = 90  # 終了人数（10の倍数）
N_STEP = 10  # 人数の刻み幅（10の倍数）

# エピソード数
EPISODES_PER_CONFIG = 1000  # 各設定（半径×人数）あたりのエピソード数

# 最大ステップ数
MAX_STEPS = 300

# 保存先ディレクトリ
OUTPUT_DIR = "output/logs/unified_critic_training"

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
    "block_size": 1,  # 状態エンコーディングのブロックサイズ（粗い位置情報）
}

# マップとSFFのパス
MAP_PATH = "data/maps/simple_room_12x12.npy"
SFF_PATH = "data/sff/distance_L1_12x12.npy"
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
    exit_positions = np.argwhere(map_array == 3)
    if len(exit_positions) == 0:
        exit_x, exit_y = 25, 49
    else:
        exit_x, exit_y = exit_positions[0]

    # L1距離場を作成
    for i in range(map_array.shape[0]):
        for j in range(map_array.shape[1]):
            if map_array[i, j] == 0 or map_array[i, j] == 3:
                sff[i, j] = abs(i - exit_x) + abs(j - exit_y)

    return sff


def find_exit_position(map_array):
    """出口位置を検出"""
    exit_positions = np.argwhere(map_array == 3)
    if len(exit_positions) == 0:
        raise ValueError("出口が見つかりません")
    # 最初の出口を使用
    exit_x, exit_y = exit_positions[0]
    return (int(exit_x), int(exit_y))


def count_available_cells(map_array, exit_pos, radius):
    """指定半径内の空きセル数をカウント"""
    exit_x, exit_y = exit_pos
    free_cells = np.argwhere(map_array == 0)
    radius_mask = (
        np.abs(free_cells[:, 0] - exit_x) + np.abs(free_cells[:, 1] - exit_y)
        <= radius
    )
    return np.sum(radius_mask)


def run_training():
    """学習の実行"""
    print("=" * 80)
    print("統合Critic学習の実行プログラム（初期配置を半径で制御）")
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

    # 出口位置を検出
    exit_pos = find_exit_position(map_array)
    print(f"✓ 出口位置: {exit_pos}")

    # 半径と人数のリストを生成
    radius_list = list(range(RADIUS_START, RADIUS_END + 1, RADIUS_STEP))
    # 人数リスト: 1だけ特別で、それ以外は10の倍数（10, 20, 30, ...）
    n_list = [1] + list(range(10, N_END + 1, N_STEP))

    print("\n設定:")
    print(f"  半径パターン: {radius_list}")
    print(f"  人数パターン: {n_list}")
    print(f"  各設定のエピソード数: {EPISODES_PER_CONFIG}")
    total_configs = len(radius_list) * len(n_list)
    print(f"  総設定数: {total_configs}")
    print(f"  総エピソード数: {total_configs * EPISODES_PER_CONFIG}")
    print(f"  最大ステップ数: {MAX_STEPS}")
    print(f"  保存先: {run_dir}")
    print("=" * 80)

    # 全体の結果を記録
    all_results = []
    episode_results = []  # 全エピソードの結果
    start_time = time.time()

    # モデルを1回だけ初期化（Vテーブルを共有）
    print("\nモデルを初期化中...")
    model = FloorFieldModelUnified(
        map_array=map_array,
        sff_path=sff_path,
        N=n_list[0],  # 初期人数（後で変更可能）
        learning_mode="critic_only",
        params=MODEL_PARAMS,
    )
    print("✓ モデル初期化完了")

    # 総エピソード数のカウンター
    total_episodes = total_configs * EPISODES_PER_CONFIG
    current_episode = 0
    config_idx = 0

    # 各半径パターンで実行
    for radius in radius_list:
        print("\n" + "=" * 80)
        print(f"半径パターン: radius={radius}")
        print("=" * 80)

        # この半径内の利用可能セル数を確認
        available_cells = count_available_cells(map_array, exit_pos, radius)
        print(f"  半径{radius}内の利用可能セル数: {available_cells}")

        # 各人数パターンで実行
        for N in n_list:
            config_idx += 1

            # 利用可能セル数を超える場合はスキップ
            if N > available_cells:
                print(
                    f"  ⚠ スキップ: N={N} > 利用可能セル数={available_cells} (radius={radius})"
                )
                continue

            print(
                f"\n  設定 [{config_idx}/{total_configs}]: radius={radius}, N={N}"
            )

            # この設定の結果を記録
            config_results = {
                "radius": radius,
                "N": N,
                "episodes": [],
                "v_table_sizes": [],
                "avg_steps": [],
            }

            # 人数を設定
            model.N = N

            # エピソードループ
            for episode in range(1, EPISODES_PER_CONFIG + 1):
                current_episode += 1
                episode_start = time.time()

                # エピソード実行（半径を指定してリセット）
                model.reset(exit_pos=exit_pos, radius=radius)
                steps = model.run(max_steps=MAX_STEPS)

                # 結果の記録
                v_table_size = model.get_v_table_size()
                episode_time = time.time() - episode_start

                config_results["episodes"].append(episode)
                config_results["v_table_sizes"].append(v_table_size)
                config_results["avg_steps"].append(steps)

                # エピソード結果を記録
                episode_results.append(
                    {
                        "episode_num": current_episode,
                        "config_idx": config_idx,
                        "radius": radius,
                        "N": N,
                        "steps": steps,
                        "v_table_size": v_table_size,
                    }
                )

                # 進捗表示（10エピソードごと、または最初と最後）
                if (
                    episode == 1
                    or episode % 10 == 0
                    or episode == EPISODES_PER_CONFIG
                ):
                    elapsed_total = time.time() - start_time
                    progress = current_episode / total_episodes
                    eta_seconds = (
                        (elapsed_total / progress) * (1 - progress)
                        if progress > 0
                        else 0
                    )
                    eta_str = time.strftime(
                        "%H:%M:%S", time.gmtime(eta_seconds)
                    )

                    print(
                        f"    radius={radius:2d} | N={N:3d} | "
                        f"Ep {episode:3d}/{EPISODES_PER_CONFIG} "
                        f"(全体{current_episode:4d}/{total_episodes}) | "
                        f"Steps: {steps:3d} | V-states: {v_table_size:5d} | "
                        f"Time: {episode_time:.2f}s | "
                        f"Progress: {progress*100:.1f}% | ETA: {eta_str}"
                    )

            # 設定の統計情報
            avg_steps = np.mean(config_results["avg_steps"])
            avg_v_size = np.mean(config_results["v_table_sizes"])
            final_v_size = config_results["v_table_sizes"][-1]

            print(
                f"\n  設定統計 (radius={radius}, N={N}):"
                f" 平均ステップ数={avg_steps:.2f}, "
                f"平均V状態数={avg_v_size:.1f}, "
                f"最終V状態数={final_v_size}"
            )

            # 結果を全体に追加
            all_results.append(config_results)

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
    print(
        f"学習データ: {len(radius_list)}種類の半径 × {len(n_list)}種類の人数 × {EPISODES_PER_CONFIG}エピソード"
    )

    # 結果の保存
    results_path = os.path.join(run_dir, "training_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "radius_list": radius_list,
                "n_list": n_list,
                "episodes_per_config": EPISODES_PER_CONFIG,
                "model_params": MODEL_PARAMS,
                "exit_pos": exit_pos,
                "results_by_config": all_results,
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
        f.write("統合Critic学習の実行サマリー（初期配置を半径で制御）\n")
        f.write("=" * 80 + "\n")
        f.write(f"実行日時: {timestamp}\n")
        f.write(
            f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n"
        )
        f.write(f"\n出口位置: {exit_pos}\n")
        f.write("\n統合Vテーブル:\n")
        f.write(f"  最終V状態数: {final_v_size}\n")
        if len(v_table) > 0:
            f.write(f"  価値の範囲: [{min(values):.2f}, {max(values):.2f}]\n")
            f.write(f"  平均価値: {np.mean(values):.2f}\n")
            f.write(f"  標準偏差: {np.std(values):.2f}\n")
        f.write("\n設定:\n")
        f.write(f"  半径パターン: {radius_list}\n")
        f.write(f"  人数パターン: {n_list}\n")
        f.write(f"  総エピソード数: {total_episodes}\n")
        f.write(f"  エピソード数/設定: {EPISODES_PER_CONFIG}\n")
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
        f.write("\n設定別の結果:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            radius = result["radius"]
            N = result["N"]
            avg_steps = np.mean(result["avg_steps"])
            v_size_start = result["v_table_sizes"][0]
            v_size_end = result["v_table_sizes"][-1]
            f.write(
                f"radius={radius:2d}, N={N:3d}: "
                f"平均ステップ={avg_steps:6.2f}, "
                f"V状態数 {v_size_start:5d}→{v_size_end:5d} "
                f"(+{v_size_end - v_size_start:4d})\n"
            )

    print(f"✓ サマリーレポートを保存しました: {report_path}")
    print("\n" + "=" * 80)
    print("✅ 全ての学習が完了しました！")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
