#!/usr/bin/env python3
"""
統合Actor学習の実行プログラム（初期配置を半径で制御）
- 事前学習済みCriticを読み込み、Actorのみを学習
- 出口から半径を徐々に広げ、人数を徐々に増やしていくカリキュラム学習
- 【変更点】イプシロン（探索率）は各設定（半径×人数）の開始時にリセットし、
  その設定のエピソードループ内で減衰させる。
"""

import os
import pickle
import time
import csv
from datetime import datetime

import numpy as np

# modelフォルダにある ffm_unified.py からクラスを読み込む想定
# 同じ階層に置く場合は `from ffm_unified import ...` に変更してください
try:
    from model.ffm_unified import FloorFieldModelUnified
except ImportError:
    # 簡易的なフォールバック（同じディレクトリにある場合）
    from ffm_unified import FloorFieldModelUnified

# ==================== 設定パラメータ ====================
# 事前学習済みCriticデータのパス
# ※実際のパスに合わせて書き換えてください
# PRETRAINED_V_PATH = "output/logs/unified_critic_training/run_20260115_094311/V_integrated_total70000ep.pkl"
PRETRAINED_V_PATH = "output/logs/unified_critic_training/run_20260117_101523/V_integrated_total70000ep.pkl"

# 半径の設定
RADIUS_START = 3   # 開始半径
RADIUS_END = 15    # 終了半径
RADIUS_STEP = 2    # 半径の刻み幅

# 人数の設定
# 人数リストは [1, 10, 20, 30, ..., N_END] となる
N_START = 1   # 未使用（互換性のため残している）
N_END = 90    # 終了人数（10の倍数）
N_STEP = 10   # 人数の刻み幅（10の倍数）

# エピソード数
EPISODES_PER_CONFIG = 100  # 各設定（半径×人数）あたりのエピソード数

# 最大ステップ数
MAX_STEPS = 300

# ε-greedy探索率（各設定内での減衰）
# 設定が切り替わるたびに START に戻り、1000エピソードかけて END まで下がる
EPSILON_START = 0.2
EPSILON_END = 0.01

# 保存先ディレクトリ
OUTPUT_DIR = "output/logs/unified_actor_training"

# モデルパラメータ
MODEL_PARAMS = {
    "k_S": 10,  # SFF係数
    "k_D": 1,   # DFF係数
    "k_A": 10,  # Actor係数
    "alpha_v": 0.01,  # Criticの学習率（新規状態のみ）
    "alpha_h": 0.1,   # Actorの学習率
    "gamma": 0.99,
    "exit_reward": 100.0,
    "step_penalty": -1.0,
    "collision_penalty": -1.0,
    "neighborhood": "neumann",
    "block_size": 1,  # 状態エンコーディングのブロックサイズ
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
    print("統合Actor学習の実行プログラム（カリキュラム学習・局所的ε減衰）")
    print("=" * 80)

    # 事前学習済みCriticの確認
    if not os.path.exists(PRETRAINED_V_PATH):
        print(f"⚠ 事前学習済みCriticが見つかりません: {PRETRAINED_V_PATH}")
        print("  先にCritic学習を実行してください。")
        return
    print(f"✓ 事前学習済みCritic: {PRETRAINED_V_PATH}")

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
    # 人数リスト: 1だけ特別で、それ以外は10の倍数
    n_list = [1] + list(range(10, N_END + 1, N_STEP))

    # 実行予定の総エピソード数を概算（スキップ分があるので実際はこれより少ない）
    estimated_total_episodes = len(radius_list) * len(n_list) * EPISODES_PER_CONFIG

    print("\n設定:")
    print(f"  半径パターン: {radius_list}")
    print(f"  人数パターン: {n_list}")
    print(f"  各設定のエピソード数: {EPISODES_PER_CONFIG}")
    print(f"  概算総エピソード数: {estimated_total_episodes} (実際は配置不可設定を除外)")
    print(f"  最大ステップ数: {MAX_STEPS}")
    print(f"  ε探索率: {EPSILON_START} → {EPSILON_END} (設定変更ごとにリセット)")
    print(f"  保存先: {run_dir}")
    print("=" * 80)

    # 全体の結果を記録
    all_results = []
    episode_results = []  # 全エピソードの結果
    start_time = time.time()

    # モデルを1回だけ初期化（V/Hテーブルを共有し、設定変更で使い回す）
    print("\nモデルを初期化中...")
    model = FloorFieldModelUnified(
        map_array=map_array,
        sff_path=sff_path,
        N=n_list[0],  # 初期人数（ループ内で変更）
        learning_mode="actor_only",
        pretrained_v_path=PRETRAINED_V_PATH,
        params=MODEL_PARAMS,
    )
    print("✓ モデル初期化完了")

    total_episodes_counter = 0  # 実際に実行した累計エピソード数
    config_idx = 0
    total_configs_estimated = len(radius_list) * len(n_list)

    # === カリキュラム学習ループ ===
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
                f"\n  設定 [{config_idx}/{total_configs_estimated}]: radius={radius}, N={N}"
            )

            # この設定の結果を記録する辞書
            config_results = {
                "radius": radius,
                "N": N,
                "episodes": [],
                "v_table_sizes": [],
                "h_table_sizes": [],
                "avg_steps": [],
            }

            # モデルの人数を更新
            model.N = N

            # === エピソードループ ===
            for episode in range(1, EPISODES_PER_CONFIG + 1):
                total_episodes_counter += 1
                episode_start = time.time()

                # --- 【重要】イプシロンのローカル減衰処理 ---
                # 現在の設定内での進捗率 (0.0付近 -> 1.0)
                local_progress = episode / EPISODES_PER_CONFIG
                
                # 線形減衰: START(0.2) から END(0.01) へ
                epsilon = (
                    EPSILON_START + (EPSILON_END - EPSILON_START) * local_progress
                )
                model.set_epsilon(epsilon)
                # ------------------------------------------

                # エピソード実行（半径を指定してリセット）
                model.reset(exit_pos=exit_pos, radius=radius)
                steps = model.run(max_steps=MAX_STEPS)

                # 結果の記録
                v_table_size = model.get_v_table_size()
                h_table_size_info = model.get_h_table_size()
                h_table_size = h_table_size_info[0] if h_table_size_info else 0
                episode_time = time.time() - episode_start

                config_results["episodes"].append(episode)
                config_results["v_table_sizes"].append(v_table_size)
                config_results["h_table_sizes"].append(h_table_size)
                config_results["avg_steps"].append(steps)

                # エピソード詳細結果を記録
                episode_results.append(
                    {
                        "episode_num": total_episodes_counter,
                        "config_idx": config_idx,
                        "radius": radius,
                        "N": N,
                        "steps": steps,
                        "v_table_size": v_table_size,
                        "h_table_size": h_table_size,
                        "epsilon": epsilon,
                    }
                )

                # 進捗表示
                if (
                    episode == 1
                    or episode % 50 == 0
                    or episode == EPISODES_PER_CONFIG
                ):
                    # V状態数の表示整形
                    if isinstance(v_table_size, tuple):
                        v_display = f"V:{v_table_size[0]}+{v_table_size[2]}"
                    else:
                        v_display = f"V:{v_table_size}"

                    print(
                        f"    Ep {episode:4d}/{EPISODES_PER_CONFIG} "
                        f"| Steps: {steps:3d} | {v_display} | H:{h_table_size:5d} | "
                        f"ε:{epsilon:.4f} | "  # イプシロンの現在値を表示
                        f"Time: {episode_time:.2f}s"
                    )

            # --- 設定終了後の処理 ---
            
            # 統計情報
            avg_steps = np.mean(config_results["avg_steps"])
            final_h_size = config_results["h_table_sizes"][-1]

            print(
                f"\n  設定完了 (radius={radius}, N={N}):"
                f" 平均ステップ数={avg_steps:.2f}, "
                f"最終H状態数={final_h_size}"
            )

            # 人数が変化するごとに途中経過のテーブルを保存
            print(
                f"  途中経過テーブルを保存中 (radius={radius}, N={N})..."
            )

            # Vテーブルを保存
            v_table = model.get_v_table()
            v_filename = (
                f"V_actor_radius{radius}_N{N}_"
                f"total{total_episodes_counter}ep.pkl"
            )
            v_intermediate_path = os.path.join(run_dir, v_filename)
            with open(v_intermediate_path, "wb") as f:
                pickle.dump(v_table, f)

            # Hテーブルを保存
            h_table = model.get_h_table()
            h_filename = (
                f"H_actor_radius{radius}_N{N}_"
                f"total{total_episodes_counter}ep.pkl"
            )
            h_intermediate_path = os.path.join(run_dir, h_filename)
            with open(h_intermediate_path, "wb") as f:
                pickle.dump(h_table, f)
            print(f"    ✓ 保存完了: {h_filename}")

            # 結果を全体に追加
            all_results.append(config_results)

    # === 全学習終了後の処理 ===
    print("\n" + "=" * 80)
    print("学習完了 - 最終データを保存中...")
    print("=" * 80)

    # 最終Vテーブル
    v_table = model.get_v_table()
    final_v_size = model.get_v_table_size()
    v_table_path = os.path.join(
        run_dir, f"V_actor_FINAL_total{total_episodes_counter}ep.pkl"
    )
    with open(v_table_path, "wb") as f:
        pickle.dump(v_table, f)
    print(f"✓ Vテーブル保存: {v_table_path}")

    # 最終Hテーブル
    h_table = model.get_h_table()
    final_h_size_info = model.get_h_table_size()
    final_h_size = final_h_size_info[0] if final_h_size_info else 0
    h_table_path = os.path.join(
        run_dir, f"H_actor_FINAL_total{total_episodes_counter}ep.pkl"
    )
    with open(h_table_path, "wb") as f:
        pickle.dump(h_table, f)
    print(f"✓ Hテーブル保存: {h_table_path}")

    # 全体の統計情報
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("全体の統計情報")
    print("=" * 80)
    print(f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"総実行エピソード数: {total_episodes_counter}")
    print(f"平均エピソード時間: {total_time / total_episodes_counter:.2f}秒")

    # 結果オブジェクトの保存
    results_path = os.path.join(run_dir, "training_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "radius_list": radius_list,
                "n_list": n_list,
                "episodes_per_config": EPISODES_PER_CONFIG,
                "model_params": MODEL_PARAMS,
                "exit_pos": exit_pos,
                "pretrained_v_path": PRETRAINED_V_PATH,
                "results_by_config": all_results,
                "all_episodes": episode_results,
                "total_time": total_time,
                "final_v_table_size": final_v_size,
                "final_h_table_size": final_h_size,
            },
            f,
        )
    print(f"\n✓ 学習結果オブジェクトを保存しました: {results_path}")

    # CSVファイル保存
    steps_csv_path = os.path.join(run_dir, "steps_per_episode.csv")
    with open(steps_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_num", "config_idx", "radius", "N",
            "steps", "v_table_size", "h_table_size", "epsilon"
        ])
        for ep_result in episode_results:
            v_size = ep_result["v_table_size"]
            if isinstance(v_size, tuple):
                v_size_str = f"{v_size[0]}+{v_size[2]}"
            else:
                v_size_str = str(v_size)

            writer.writerow([
                ep_result["episode_num"],
                ep_result["config_idx"],
                ep_result["radius"],
                ep_result["N"],
                ep_result["steps"],
                v_size_str,
                ep_result["h_table_size"],
                f"{ep_result['epsilon']:.6f}"
            ])
    print(f"✓ ステップ数ログ(CSV)を保存しました: {steps_csv_path}")

    print("\n" + "=" * 80)
    print("✅ 全ての学習が完了しました！")
    print("=" * 80)


if __name__ == "__main__":
    run_training()