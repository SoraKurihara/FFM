#!/usr/bin/env python3
"""
Actor学習の実行プログラム（事前学習済みCriticを使用）
事前に学習されたCriticデータを読み込み、Actorのみを学習する

学習の流れ:
1. 事前学習済みCriticデータ（V_integrated_total*.pkl）を読み込み
2. Actorを0から学習（最初はランダムウォーク）
3. シミュレーション中に新しく遭遇した状態のV値のみ追加学習
4. 学習されたH値と新規V値をlogsに保存
"""

import os
import pickle
import time
from datetime import datetime

import numpy as np

from model.ffm_actor_only import FloorFieldModelActorOnly

# ==================== 設定パラメータ ====================
# 事前学習済みCriticデータのパス
PRETRAINED_V_PATH = "output/logs/critic_training/run_20251206_153157/V_integrated_total11000ep.pkl"
# PRETRAINED_V_PATH = "output/logs/critic_training/alpha_v_0.01_gamma_0.99/V_integrated_total11000ep.pkl"

# 人数のパターン
N_FIXED = [1]  # 必ず実行する人数
N_START = 1  # 10刻みの開始
N_END = 1  # 10刻みの終了
N_STEP = 10  # 刻み幅

# エピソード数
EPISODES_PER_N = 10000

# 最大ステップ数
MAX_STEPS = 1000

# ε-greedy探索率（線形減衰）
EPSILON_START = 0.2
EPSILON_END = 0.01

# 保存先ディレクトリ
OUTPUT_DIR = "output/logs/actor_only_training"

# モデルパラメータ
MODEL_PARAMS = {
    "k_D": 1,
    "k_A": 10,
    "alpha_v": 0.1,  # Criticの学習率（新規状態のみ）
    "alpha_h": 0.1,  # Actorの学習率
    "gamma": 0.95,
    "exit_reward": 100.0,
    "step_penalty": 0.0,
    "collision_penalty": -1.0,
    "neighborhood": "neumann",
    "epsilon": EPSILON_START,
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
    print("Actor学習の実行プログラム（事前学習済みCriticを使用）")
    print("=" * 80)

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    # 軌跡保存用ディレクトリ
    trajectories_dir = os.path.join(run_dir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)

    # 事前学習済みCriticの確認
    if not os.path.exists(PRETRAINED_V_PATH):
        print(f"⚠ 警告: 事前学習済みCriticが見つかりません: {PRETRAINED_V_PATH}")
        print("  Criticなしで学習を開始します")
        pretrained_v_path = None
    else:
        pretrained_v_path = PRETRAINED_V_PATH
        print(f"✓ 事前学習済みCritic: {PRETRAINED_V_PATH}")

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
    print(f"  エピソード数/パターン: {EPISODES_PER_N}")
    print(f"  総エピソード数: {total_patterns * EPISODES_PER_N}")
    print(f"  最大ステップ数: {MAX_STEPS}")
    print(f"  保存先: {run_dir}")
    print("=" * 80)

    # 全体の結果を記録
    all_results = []
    episode_results = []  # 全エピソードの結果
    start_time = time.time()

    # 共有モデルを1回だけ初期化（Hテーブルは全人数で共有）
    print("\nモデルを初期化中...")
    model = FloorFieldModelActorOnly(
        map_array=map_array,
        sff_path=sff_path,
        N=n_list[0],  # 初期人数（後で変更可能）
        pretrained_v_path=pretrained_v_path,
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

        # 人数を設定（Hテーブルは共有）
        model.N = N

        # このパターンの結果を記録
        pattern_results = {
            "N": N,
            "episodes": [],
            "v_initial_size": [],
            "v_current_size": [],
            "v_new_states": [],
            "h_table_sizes": [],
            "avg_steps": [],
            "epsilons": [],
        }

        # エピソードループ
        for episode in range(1, EPISODES_PER_N + 1):
            current_episode += 1
            episode_start = time.time()

            if total_episodes > 1:
                progress = (current_episode - 1) / (total_episodes - 1)
            else:
                progress = 0.0
            epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * progress
            epsilon = float(np.clip(epsilon, 0.0, 1.0))
            model.set_epsilon(epsilon)

            # エピソード実行
            model.reset()
            # 10エピソードに1回軌跡を保存
            save_trajectory = episode % 100 == 0
            if save_trajectory:
                steps, trajectory = model.run(
                    max_steps=MAX_STEPS, return_trajectory=True
                )
                # 軌跡を保存
                trajectory_filename = os.path.join(
                    trajectories_dir,
                    f"trajectory_N{N}_ep{episode:05d}_total{current_episode:05d}.npz",
                )
                np.savez_compressed(
                    trajectory_filename,
                    positions=trajectory,
                    episode=episode,
                    N=N,
                    total_episode=current_episode,
                    steps=steps,
                )
            else:
                steps = model.run(max_steps=MAX_STEPS, return_trajectory=False)

            # 結果の記録
            (
                v_initial_size,
                v_current_size,
                v_new_states,
            ) = model.get_v_table_size()
            h_states, h_total_actions = model.get_h_table_size()
            episode_time = time.time() - episode_start

            pattern_results["episodes"].append(episode)
            pattern_results["v_initial_size"].append(v_initial_size)
            pattern_results["v_current_size"].append(v_current_size)
            pattern_results["v_new_states"].append(v_new_states)
            pattern_results["h_table_sizes"].append(
                (h_states, h_total_actions)
            )
            pattern_results["avg_steps"].append(steps)
            pattern_results["epsilons"].append(epsilon)

            # エピソード結果を記録
            episode_results.append(
                {
                    "episode_num": current_episode,
                    "N": N,
                    "steps": steps,
                    "v_initial_size": v_initial_size,
                    "v_current_size": v_current_size,
                    "v_new_states": v_new_states,
                    "h_states": h_states,
                    "h_total_actions": h_total_actions,
                    "epsilon": epsilon,
                }
            )

            # 進捗表示（10エピソードごと、または最初と最後）
            if episode == 1 or episode % 1 == 0 or episode == EPISODES_PER_N:
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
                    f"Steps: {steps:3d} | V-states: {v_current_size:5d} "
                    f"(初期:{v_initial_size}, 増加:{v_new_states}) | "
                    f"H-states: {h_states:5d} | "
                    f"ε: {epsilon:.3f} | "
                    f"Time: {episode_time:.2f}s | "
                    f"Progress: {progress*100:.1f}% | ETA: {eta_str}"
                )

        # パターンの統計情報
        avg_steps = np.mean(pattern_results["avg_steps"])
        final_v_initial = pattern_results["v_initial_size"][-1]
        final_v_current = pattern_results["v_current_size"][-1]
        final_v_new = pattern_results["v_new_states"][-1]
        final_h_states, final_h_total = pattern_results["h_table_sizes"][-1]

        print(f"\n  パターンN={N}の統計:")
        print(f"    平均ステップ数: {avg_steps:.2f}")
        print(
            f"    最終V状態数: {final_v_current} (初期:{final_v_initial}, 増加:{final_v_new})"
        )
        print(f"    最終H状態数: {final_h_states} (総行動数: {final_h_total})")

        # この人数でのHテーブルを保存（共有Hテーブルのスナップショット）
        h_table_n = model.get_h_table()
        h_states_n, h_total_actions_n = model.get_h_table_size()
        h_table_path_n = os.path.join(
            run_dir, f"H_actor_N{N}_total{EPISODES_PER_N}ep.pkl"
        )
        with open(h_table_path_n, "wb") as f:
            pickle.dump(h_table_n, f)
        print(f"\n✓ N={N}でのHテーブル保存: {h_table_path_n}")
        print(f"  H状態数: {h_states_n} (総行動数: {h_total_actions_n})")

        # 結果を全体に追加
        all_results.append(pattern_results)

    # 最終的なテーブルを保存
    print("\n" + "=" * 80)
    print("学習完了 - テーブルを保存中...")
    print("=" * 80)

    # 共有モデルからVテーブルを取得
    v_initial_size, v_current_size, v_new_states = model.get_v_table_size()
    v_table = model.get_v_table()
    h_table = model.get_h_table()
    h_states, h_total_actions = model.get_h_table_size()

    # 更新されたVテーブルの保存
    v_table_path = os.path.join(
        run_dir, f"V_updated_total{total_episodes}ep.pkl"
    )
    with open(v_table_path, "wb") as f:
        pickle.dump(v_table, f)
    print(f"✓ 更新済みVテーブル保存: {v_table_path}")
    print(f"  V状態数: {v_current_size} (初期:{v_initial_size}, 増加:{v_new_states})")

    # V値の統計
    if len(v_table) > 0:
        v_values = list(v_table.values())
        print(f"  V価値の範囲: [{min(v_values):.2f}, {max(v_values):.2f}]")
        print(f"  V平均価値: {np.mean(v_values):.2f}")
        print(f"  V標準偏差: {np.std(v_values):.2f}")

    # 最終的なHテーブルを保存（全人数で共有されたHテーブル）
    h_table_path = os.path.join(
        run_dir, f"H_actor_total{total_episodes}ep.pkl"
    )
    with open(h_table_path, "wb") as f:
        pickle.dump(h_table, f)
    print(f"\n✓ 最終Hテーブル保存: {h_table_path}")
    print(f"  H状態数: {h_states} (総行動数: {h_total_actions})")

    # Hロジットの統計
    if len(h_table) > 0:
        all_h_values = []
        for state_actions in h_table.values():
            all_h_values.extend(state_actions)
        if all_h_values:
            print(
                f"  Hロジットの範囲: [{min(all_h_values):.2f}, {max(all_h_values):.2f}]"
            )
            print(f"  H平均ロジット: {np.mean(all_h_values):.2f}")
            print(f"  H標準偏差: {np.std(all_h_values):.2f}")

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
                "pretrained_v_path": PRETRAINED_V_PATH,
                "n_list": n_list,
                "episodes_per_n": EPISODES_PER_N,
                "model_params": MODEL_PARAMS,
                "results_by_n": all_results,
                "all_episodes": episode_results,  # 全エピソードの詳細
                "total_time": total_time,
                "final_v_initial_size": v_initial_size,
                "final_v_current_size": v_current_size,
                "final_v_new_states": v_new_states,
                "final_h_states": h_states,
                "final_h_total_actions": h_total_actions,
                "epsilon_start": EPSILON_START,
                "epsilon_end": EPSILON_END,
            },
            f,
        )
    print(f"\n✓ 学習結果を保存しました: {results_path}")

    # サマリーレポートの作成
    report_path = os.path.join(run_dir, "summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Actor学習の実行サマリー（事前学習済みCriticを使用）\n")
        f.write("=" * 80 + "\n")
        f.write(f"実行日時: {timestamp}\n")
        f.write(
            f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n"
        )
        f.write(f"\n事前学習済みCritic: {PRETRAINED_V_PATH}\n")
        f.write("\n学習結果:\n")
        f.write(f"  初期V状態数: {v_initial_size}\n")
        f.write(f"  最終V状態数: {v_current_size}\n")
        f.write(f"  新規追加状態数: {v_new_states}\n")
        if len(v_table) > 0:
            f.write(f"  V価値の範囲: [{min(v_values):.2f}, {max(v_values):.2f}]\n")
            f.write(f"  V平均価値: {np.mean(v_values):.2f}\n")
        f.write(f"  Actor H状態数: {h_states} (総行動数: {h_total_actions})\n")
        if len(h_table) > 0:
            all_h_values = []
            for state_actions in h_table.values():
                all_h_values.extend(state_actions)
            if all_h_values:
                f.write(
                    f"  Hロジットの範囲: [{min(all_h_values):.2f}, {max(all_h_values):.2f}]\n"
                )
                f.write(f"  H平均ロジット: {np.mean(all_h_values):.2f}\n")
        f.write(
            f"  εスケジュール: start={EPSILON_START:.3f}, end={EPSILON_END:.3f}\n"
        )
        f.write("\n設定:\n")
        f.write(f"  人数パターン: {n_list}\n")
        f.write(f"  総エピソード数: {total_episodes}\n")
        f.write(f"  エピソード数/パターン: {EPISODES_PER_N}\n")
        f.write(f"  最大ステップ数: {MAX_STEPS}\n")
        f.write("\nモデルパラメータ:\n")
        for key, value in MODEL_PARAMS.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n人数別の結果:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            N = result["N"]
            avg_steps = np.mean(result["avg_steps"])
            v_current_start = result["v_current_size"][0]
            v_current_end = result["v_current_size"][-1]
            v_new_added = v_current_end - v_current_start
            h_states_end, h_total_end = result["h_table_sizes"][-1]
            f.write(
                f"N={N:3d}: 平均ステップ={avg_steps:6.2f}, "
                f"V状態数 {v_current_start:5d}→{v_current_end:5d} "
                f"(+{v_new_added:4d}), "
                f"H状態数 {h_states_end:5d} (総行動数: {h_total_end})\n"
            )

    print(f"✓ サマリーレポートを保存しました: {report_path}")
    print("\n" + "=" * 80)
    print("✅ 全ての学習が完了しました！")
    print("=" * 80)


if __name__ == "__main__":
    run_training()
