#!/usr/bin/env python3
"""
指定フォルダ内の trajectory_ep*.pkl を列挙し、各セルにいた人数の累積回数を集計して
マップ .npy 上にヒートマップとして保存する。
"""

import os
import pickle
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ログディレクトリ（この中の trajectory_ep*.pkl を対象）
LOG_DIR = r"output/logs/ffm_ac_aggregated/ob1"
# 一括出力するエピソード番号の範囲（両端含む）
EPISODE_MIN = 10
EPISODE_MAX = 1000
# マップ（壁=1 は黒で表示）
MAP_NPY = r"output/logs/ffm_ac_aggregated/ob1/excel_map_009.npy"


class CompatibleUnpickler(pickle.Unpickler):
    """numpy 2.x 互換で pickle を読むための Unpickler"""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.umath":
            module = "numpy.core.umath"
        return super().find_class(module, name)


def load_trajectory(path):
    with open(path, "rb") as f:
        data = CompatibleUnpickler(f).load()
    return list(data)


def build_visit_count_map(positions, height, width):
    """
    positions: list of (N, 2) array
    戻り値: shape=(height, width) の累積人数カウント配列
    """
    counts = np.zeros((height, width), dtype=np.int64)
    for step_positions in positions:
        step_positions = np.asarray(step_positions)
        for agent_pos in step_positions:
            x, y = int(agent_pos[0]), int(agent_pos[1])
            if 0 <= x < height and 0 <= y < width:
                counts[x, y] += 1
    return counts


def find_trajectory_pkls(log_dir, ep_min, ep_max):
    """log_dir 内の trajectory_ep{N}.pkl で ep_min <= N <= ep_max のものを (ep, path) で返す"""
    if not os.path.isdir(log_dir):
        return []
    pattern = re.compile(r"^trajectory_ep(\d+)\.pkl$", re.IGNORECASE)
    out = []
    for name in os.listdir(log_dir):
        m = pattern.match(name)
        if not m:
            continue
        ep = int(m.group(1))
        if ep_min <= ep <= ep_max:
            out.append((ep, os.path.join(log_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def render_heatmap(map_array, counts, output_path, episode):
    """
    マップを背景に、歩行者累積人数をヒートマップ表示して保存する。
    """
    # 通行不可セル（例: 壁=1）をマスクし、壁にヒートが乗らないようにする
    wall_mask = map_array == 1
    plot_counts = np.ma.array(counts, mask=wall_mask)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 背景マップ（白: 通路, 黒: 壁, 緑: 出口）
    base = np.ones((*map_array.shape, 3), dtype=np.float32)
    base[map_array == 1] = [0.0, 0.0, 0.0]
    base[map_array == 3] = [0.2, 0.8, 0.2]
    ax.imshow(base, interpolation="nearest")

    # 人数ヒートマップ（人数0は透過）
    vmax = int(plot_counts.max()) if plot_counts.count() > 0 else 0
    white_blue = LinearSegmentedColormap.from_list(
        "white_blue",
        ["#ffffff", "#0000ff"],
    )
    heat = ax.imshow(
        plot_counts,
        cmap=white_blue,
        alpha=0.75,
        interpolation="nearest",
        vmin=1 if vmax > 0 else 0,
        vmax=vmax if vmax > 0 else 1,
    )

    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative pedestrian count")

    ax.set_title(f"Trajectory density heatmap (episode {episode})")
    ax.set_xlabel("Y (column)")
    ax.set_ylabel("X (row)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, map_array.shape[1] - 0.5)
    ax.set_ylim(map_array.shape[0] - 0.5, -0.5)

    # 参考値を表示
    total_visits = int(counts.sum())
    max_visits = int(counts.max()) if counts.size else 0
    ax.text(
        0.01,
        0.99,
        f"total visits: {total_visits}\nmax per cell: {max_visits}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    if not os.path.isfile(MAP_NPY):
        raise FileNotFoundError(f"map file not found: {MAP_NPY}")

    pairs = find_trajectory_pkls(LOG_DIR, EPISODE_MIN, EPISODE_MAX)
    if not pairs:
        raise FileNotFoundError(
            f"no trajectory_ep*.pkl in [{EPISODE_MIN}, {EPISODE_MAX}] "
            f"under: {LOG_DIR}"
        )

    map_array = np.load(MAP_NPY)
    if map_array.ndim != 2:
        raise ValueError(
            f"map array must be 2D, but got shape={map_array.shape}"
        )
    h, w = map_array.shape

    for ep, pkl_path in pairs:
        positions = load_trajectory(pkl_path)
        counts = build_visit_count_map(positions, h, w)
        stem = os.path.splitext(os.path.basename(pkl_path))[0]
        output_png = os.path.join(LOG_DIR, f"{stem}_density_heatmap.png")
        render_heatmap(map_array, counts, output_png, episode=ep)
        print(f"saved: {output_png}")

    print(f"done: {len(pairs)} file(s)")


if __name__ == "__main__":
    main()
