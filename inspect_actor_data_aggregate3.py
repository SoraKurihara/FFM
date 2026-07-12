"""
3つの統合Actor学習データ（H_actor_FINAL_*.pkl）を読み込み、それぞれ異なるマップで距離[m]・角度を計算した上で、
同じ (周囲の状況, 距離ビン, 角度ビン) でまとめて1つに集約し、1つのpklに保存するスクリプト。

・各データは対応するマップで出口位置を計算する。
・距離は 1セル=0.5m とし、DIST_BIN_M[m] 刻みの距離ビンに変換する。
・(ranks_tuple, 距離ビン, 角度ビン) が同じキー同士のH値をプールして平均化する。
・集約結果を「(周囲の状況, 距離ビン, 角度ビン) -> 上下左右待機の価値」形式で別pklに保存する。
"""
import os
import pickle
from collections import defaultdict

import numpy as np

# 元スクリプトの共通処理を利用
from inspect_actor_data import (
    load_h_table,
    get_exit_and_max_distance,
    distance_bin_m,
    angle_to_exit_deg,
    angle_bin_index,
    ACTION_LABELS,
    format_h_values,
    top2_actions,
    ANGLE_BIN_DEG,
    CELL_SIZE_M,
    DIST_BIN_M,
)

# ---------------------------------------------------------------------------
# 3つのデータセット: それぞれ (Actor pkl のパス, 対応するマップのパス)
# マップが異なるため、距離帯・角度は各マップの出口・最大距離で計算する。
# ---------------------------------------------------------------------------
DATASETS = [
    (
        os.path.join(
            "output", "logs", "unified_actor_training",
            "learning_center_12", "H_actor_FINAL_total4800ep.pkl",
        ),
        "data/maps/learning_center_12.npy",
    ),
    (
        os.path.join(
            "output", "logs", "unified_actor_training",
            "learning_corner_ul_12", "H_actor_FINAL_total4800ep.pkl",  # 例: 2つ目
        ),
        "data/maps/learning_corner_ul_12.npy",  # 例: 別マップ
    ),
    (
        os.path.join(
            "output", "logs", "unified_actor_training",
            "learning_corner_ur_12", "H_actor_FINAL_total4800ep.pkl",  # 例: 3つ目
        ),
        "data/maps/learning_corner_ur_12.npy",  # 例: 別マップ
    ),
]

# 集約結果の保存先（1つのpklにまとめる）
OUTPUT_PKL_PATH = os.path.join(
    "output", "logs", "unified_actor_training",
    "H_actor_aggregated_3maps_by_distm_angle.pkl",
)


def process_one_file(actor_path, map_path):
    """
    1つのActor pkl とそのマップから、(ranks_tuple, d_bin, a_bin) -> list of H vectors を構築。
    戻り値: (groups_detail, n_states, exit_pos, max_l1)
    """
    exit_pos, max_l1 = get_exit_and_max_distance(map_path)
    ex, ey = exit_pos
    data = load_h_table(actor_path)
    if not isinstance(data, dict):
        return None, 0, exit_pos, max_l1

    n_bins = 360 // ANGLE_BIN_DEG
    groups_detail = defaultdict(list)

    for key, h_values in data.items():
        if isinstance(key, bytes):
            try:
                key = pickle.loads(key)
            except Exception:
                continue
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        ranks_tuple, block_idx = key
        if not isinstance(block_idx, (list, tuple)) or len(block_idx) < 2:
            continue
        x, y = int(block_idx[0]), int(block_idx[1])
        l1 = abs(x - ex) + abs(y - ey)
        d_bin = distance_bin_m(l1)
        ang = angle_to_exit_deg(x, y, ex, ey)
        a_bin = angle_bin_index(ang, ANGLE_BIN_DEG)
        h_vec = np.asarray(h_values, dtype=np.float64)
        if h_vec.size >= 5:
            groups_detail[(ranks_tuple, d_bin, a_bin)].append(h_vec)

    return groups_detail, len(data), exit_pos, max_l1


def main():
    print("=" * 70)
    print("3データ集約: 統合Actor学習データ（距離[m]ビン・角度でマージ・平均化）")
    print("=" * 70)

    # 各ファイルを読み、同じキーでプール
    combined = defaultdict(list)
    total_states = 0
    n_bins = 360 // ANGLE_BIN_DEG

    for i, (actor_path, map_path) in enumerate(DATASETS, 1):
        if not os.path.isfile(actor_path):
            print("スキップ（ファイルなし）: {}".format(actor_path))
            continue
        if not os.path.isfile(map_path):
            print("スキップ（マップなし）: {} -> {}".format(actor_path, map_path))
            continue

        groups_detail, n_states, exit_pos, max_l1 = process_one_file(actor_path, map_path)
        if groups_detail is None:
            print("スキップ（データ型不正）: {}".format(actor_path))
            continue

        print("データ{}: {} (マップ: {}, 出口: {}, 最大L1: {}), 状態数: {}".format(
            i, actor_path, map_path, exit_pos, max_l1, n_states))
        total_states += n_states

        for key, vecs in groups_detail.items():
            combined[key].extend(vecs)

    if not combined:
        print("有効なデータがありません。パスとマップを確認してください。")
        return

    print("\n距離ビン: 1セル={}m, 幅={}m (bin=0,1,2,...)".format(CELL_SIZE_M, DIST_BIN_M))
    print("角度: 正面(出口方向)=0度, ビン幅={}度\n".format(ANGLE_BIN_DEG))

    # (d_bin, a_bin) で表示用サマリを計算
    groups_display = defaultdict(list)
    for (ranks_tuple, d_bin, a_bin), vecs in combined.items():
        groups_display[(d_bin, a_bin)].extend(vecs)

    summary = []
    for (d_bin, a_bin), vecs in sorted(groups_display.items()):
        arr = np.array(vecs)
        mean_h = np.mean(arr, axis=0)
        summary.append((d_bin, a_bin, len(vecs), mean_h))

    print("総状態数（3データ合計）: {}".format(total_states))
    print("集約グループ数: {} (距離ビン x 角度ビン{})\n".format(len(summary), n_bins))
    print("-" * 100)
    print("距離[m] | 角度ビン(中心度) | 状態数 | 平均H (上, 下, 左, 右, 待機) | 1位 | 2位")
    print("-" * 100)

    for d_bin, a_bin, count, mean_h in summary:
        center_deg = (a_bin + 0.5) * ANGLE_BIN_DEG - 180
        d_from = d_bin * DIST_BIN_M
        d_to = (d_bin + 1) * DIST_BIN_M
        h_str = format_h_values(mean_h.tolist())
        (lbl1, val1), (lbl2, val2) = top2_actions(mean_h)
        print("{:4.1f}-{:<4.1f} | {:7.1f}          | {:5d} | {} | {}={:.4f} | {}={:.4f}".format(
            d_from, d_to, center_deg, count, h_str, lbl1, val1, lbl2, val2))

    print("-" * 100)

    # 保存用: (ranks_tuple, d_bin, a_bin) -> 平均H
    out_dict = {}
    for key, vecs in combined.items():
        mean_h = np.mean(np.array(vecs), axis=0)
        out_dict[key] = mean_h.tolist()

    out_dir = os.path.dirname(OUTPUT_PKL_PATH)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        with open(OUTPUT_PKL_PATH, "wb") as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("集約データを保存しました: {} ({} エントリ)".format(OUTPUT_PKL_PATH, len(out_dict)))
    except Exception as e:
        print("保存に失敗しました: {}".format(e))

    print("処理完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
