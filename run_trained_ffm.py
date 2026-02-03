#!/usr/bin/env python3
"""
学習済みHテーブルを使用するFFM実行プログラム

training_results.pklから学習済みのHテーブルを読み込み、
その移動確率を使ってFFMシミュレーションを実行します。
"""

import os
import pickle
import glob
import re
import time
import csv
from datetime import datetime

import numpy as np

from model.ffm_trained_core import FloorFieldModel

# ==================== 設定パラメータ ====================
# training_results.pklのパス
TRAINING_RESULTS_PATH = (
    "output/logs/unified_actor_training/"
    "run_20260115_234109/training_results.pkl"
)

# マップとSFFのパス
MAP_PATH = "data/maps/simple_room_12x12.npy"
SFF_PATH = "data/sff/distance_L1_12x12.npy"

# 人数の設定
# 人数リストは [1, 10, 20, 30, ..., N_END] となる（1だけ特別、それ以外は10の倍数）
N_END = 90  # 終了人数（10の倍数）
N_STEP = 10  # 人数の刻み幅（10の倍数）

# エピソード数
EPISODES_PER_N = 100  # 各人数あたりのエピソード数

# シミュレーション設定
MAX_STEPS = 300  # 最大ステップ数

# 出力ディレクトリ
OUTPUT_DIR = "output/logs/trained_ffm"
# ========================================================


def find_h_table_path(training_results_path):
    """
    training_results.pklと同じディレクトリからHテーブルファイルを探す
    
    Args:
        training_results_path: training_results.pklのパス
    
    Returns:
        str: Hテーブルファイルのパス（H_actor_total*.pkl）
    """
    base_dir = os.path.dirname(training_results_path)
    
    # H_actor_total*.pklを探す
    pattern = os.path.join(base_dir, "H_actor_total*.pkl")
    h_files = glob.glob(pattern)
    
    if not h_files:
        raise FileNotFoundError(
            f"Hテーブルファイルが見つかりません: {pattern}"
        )
    
    # 最新のファイルを選択（エピソード数が最大のもの）
    def extract_episode_num(filepath):
        """ファイル名からエピソード数を抽出"""
        basename = os.path.basename(filepath)
        # H_actor_total70000ep.pkl から 70000 を抽出
        match = re.search(r'total(\d+)ep', basename)
        if match:
            return int(match.group(1))
        # フォールバック: 数字を探す
        match = re.search(r'(\d+)ep', basename)
        if match:
            return int(match.group(1))
        return 0
    
    h_file = max(h_files, key=extract_episode_num)
    
    return h_file


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


def main():
    """メイン実行関数"""
    print("=" * 80)
    print("学習済みHテーブルを使用するFFM実行プログラム")
    print("=" * 80)
    
    # training_results.pklの確認
    if not os.path.exists(TRAINING_RESULTS_PATH):
        print(f"⚠ training_results.pklが見つかりません: {TRAINING_RESULTS_PATH}")
        return
    print(f"✓ training_results.pkl: {TRAINING_RESULTS_PATH}")
    
    # training_results.pklから設定を読み込み
    with open(TRAINING_RESULTS_PATH, "rb") as f:
        training_data = pickle.load(f)
    
    model_params = training_data.get("model_params", {})
    print("✓ 学習パラメータを読み込みました")
    print(f"  k_D: {model_params.get('k_D', 'N/A')}")
    print(f"  k_A: {model_params.get('k_A', 'N/A')}")
    print(f"  block_size: {model_params.get('block_size', 'N/A')}")
    print(f"  neighborhood: {model_params.get('neighborhood', 'N/A')}")
    
    # Hテーブルファイルを探す
    h_table_path = find_h_table_path(TRAINING_RESULTS_PATH)
    print(f"✓ Hテーブルファイル: {h_table_path}")
    
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
        sff_path = os.path.join(OUTPUT_DIR, "sff_temp.npy")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.save(sff_path, sff)
        print(f"⚠ テスト用SFFを作成しました（{SFF_PATH}が見つかりません）")
    
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # パラメータの設定（training_results.pklから取得したものを使用）
    params = {
        "k_D": model_params.get("k_D", 1),
        "k_A": model_params.get("k_A", 10),
        "diffuse": 0.2,
        "decay": 0.2,
        "neighborhood": model_params.get("neighborhood", "neumann"),
        "block_size": model_params.get("block_size", 5),
    }
    
    # 人数リストを生成: [1, 10, 20, 30, ..., N_END]
    n_list = [1] + list(range(10, N_END + 1, N_STEP))
    
    print("\n" + "=" * 80)
    print("シミュレーション設定")
    print("=" * 80)
    print(f"人数パターン: {n_list}")
    print(f"各人数のエピソード数: {EPISODES_PER_N}")
    print(f"最大ステップ数: {MAX_STEPS}")
    print(f"保存先: {run_dir}")
    total_episodes = len(n_list) * EPISODES_PER_N
    print(f"総エピソード数: {total_episodes}")
    print("=" * 80)
    
    # 全体の結果を記録
    all_results = []
    episode_results = []  # 全エピソードの結果
    start_time = time.time()
    
    # FFMモデルを1回だけ初期化（Hテーブルを共有）
    print("\nFFMモデルを初期化中...")
    model = FloorFieldModel(
        map_array=map_array,
        sff_path=sff_path,
        N=n_list[0],  # 初期人数（後で変更可能）
        h_table_path=h_table_path,
        params=params,
    )
    print("✓ モデル初期化完了")
    
    # 総エピソード数のカウンター
    current_episode = 0
    config_idx = 0
    
    # 各人数パターンで実行
    for N in n_list:
        config_idx += 1
        print("\n" + "=" * 80)
        print(f"人数パターン: N={N} [{config_idx}/{len(n_list)}]")
        print("=" * 80)
        
        # この設定の結果を記録
        config_results = {
            "N": N,
            "episodes": [],
            "avg_steps": [],
        }
        
        # 人数を設定
        model.N = N
        
        # エピソードループ
        for episode in range(1, EPISODES_PER_N + 1):
            current_episode += 1
            episode_start = time.time()
            
            # エージェントの初期配置をリセット
            model.positions = model.initialize_agents()
            model.dff = np.zeros_like(model.map_array, dtype=np.float32)
            
            # シミュレーション実行
            steps = model.run(
                save_prefix=None,  # 個別の軌跡は保存しない
                save_interval=100,
                max_steps=MAX_STEPS,
            )
            
            # 結果の記録
            episode_time = time.time() - episode_start
            config_results["episodes"].append(episode)
            config_results["avg_steps"].append(steps)
            
            # エピソード結果を記録
            episode_results.append(
                {
                    "episode_num": current_episode,
                    "config_idx": config_idx,
                    "N": N,
                    "steps": steps,
                }
            )
            
            # 進捗表示（10エピソードごと、または最初と最後）
            if (
                episode == 1
                or episode % 10 == 0
                or episode == EPISODES_PER_N
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
                    f"  N={N:3d} | Ep {episode:3d}/{EPISODES_PER_N} "
                    f"(全体{current_episode:4d}/{total_episodes}) | "
                    f"Steps: {steps:3d} | "
                    f"Time: {episode_time:.2f}s | "
                    f"Progress: {progress*100:.1f}% | ETA: {eta_str}"
                )
        
        # 設定の統計情報
        avg_steps = np.mean(config_results["avg_steps"])
        std_steps = np.std(config_results["avg_steps"])
        
        print(
            f"\n  設定統計 (N={N}):"
            f" 平均ステップ数={avg_steps:.2f}±{std_steps:.2f}"
        )
        
        # 結果を全体に追加
        all_results.append(config_results)
    
    # 全体の統計情報
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("全体の統計情報")
    print("=" * 80)
    print(f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"総エピソード数: {total_episodes}")
    print(f"平均エピソード時間: {total_time / total_episodes:.2f}秒")
    
    # 結果の保存
    results_path = os.path.join(run_dir, "simulation_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(
            {
                "n_list": n_list,
                "episodes_per_n": EPISODES_PER_N,
                "model_params": params,
                "results_by_n": all_results,
                "all_episodes": episode_results,
                "total_time": total_time,
            },
            f,
        )
    print(f"\n✓ シミュレーション結果を保存しました: {results_path}")
    
    # 各エピソードのステップ数をCSVファイルに保存
    steps_csv_path = os.path.join(run_dir, "steps_per_episode.csv")
    with open(steps_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # ヘッダー
        writer.writerow([
            "episode_num", "config_idx", "N", "steps"
        ])
        # データ
        for ep_result in episode_results:
            writer.writerow([
                ep_result["episode_num"],
                ep_result["config_idx"],
                ep_result["N"],
                ep_result["steps"],
            ])
    print(f"✓ 各エピソードのステップ数を保存しました: {steps_csv_path}")
    
    # サマリーレポートの作成
    report_path = os.path.join(run_dir, "summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("学習済みHテーブルを使用するFFM実行サマリー\n")
        f.write("=" * 80 + "\n")
        f.write(f"実行日時: {timestamp}\n")
        f.write(
            f"総実行時間: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n"
        )
        f.write("\n入力ファイル:\n")
        f.write(f"  training_results.pkl: {TRAINING_RESULTS_PATH}\n")
        f.write(f"  Hテーブル: {h_table_path}\n")
        map_str = MAP_PATH if os.path.exists(MAP_PATH) else "テスト用マップ"
        sff_str = SFF_PATH if os.path.exists(SFF_PATH) else "テスト用SFF"
        f.write(f"  マップ: {map_str}\n")
        f.write(f"  SFF: {sff_str}\n")
        f.write("\n設定:\n")
        f.write(f"  人数パターン: {n_list}\n")
        f.write(f"  総エピソード数: {total_episodes}\n")
        f.write(f"  エピソード数/人数: {EPISODES_PER_N}\n")
        f.write(f"  最大ステップ数: {MAX_STEPS}\n")
        f.write("\nパラメータ:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n人数別の結果:\n")
        f.write("-" * 80 + "\n")
        for result in all_results:
            N = result["N"]
            avg_steps = np.mean(result["avg_steps"])
            std_steps = np.std(result["avg_steps"])
            f.write(
                f"N={N:3d}: "
                f"平均ステップ={avg_steps:6.2f}±{std_steps:6.2f}\n"
            )
    
    print(f"✓ サマリーレポートを保存しました: {report_path}")
    print("\n" + "=" * 80)
    print("✅ 全ての処理が完了しました！")
    print("=" * 80)


if __name__ == "__main__":
    main()

