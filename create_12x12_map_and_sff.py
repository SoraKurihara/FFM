#!/usr/bin/env python3
"""
12x12のマップとL1ノルムのSFFを作成するスクリプト
data/maps/simple_room.npyを参考に作成
"""

import numpy as np
import os

# ディレクトリの作成
os.makedirs("data/maps", exist_ok=True)
os.makedirs("data/sff", exist_ok=True)

# 12x12のマップを作成
map_array = np.zeros((12, 12), dtype=np.uint8)

# 外周を壁（値2）で囲む
map_array[0, :] = 2   # 上壁
map_array[-1, :] = 2  # 下壁
map_array[:, 0] = 2   # 左壁
map_array[:, -1] = 2  # 右壁

# 出口を上端の中央に配置（値3）
exit_x, exit_y = 0, 6  # 12x12の場合、中央はインデックス6
map_array[exit_x, exit_y] = 3

# マップを保存
map_path = "data/maps/simple_room_12x12.npy"
np.save(map_path, map_array)
print(f"✓ マップを保存しました: {map_path}")
print(f"  サイズ: {map_array.shape}")
print(f"  出口位置: ({exit_x}, {exit_y})")
print(f"  値の種類: {np.unique(map_array)}")

# L1距離のSFFを作成
exit_positions = np.argwhere(map_array == 3)
height, width = map_array.shape

# 初期化（無限大で埋める）
dist_L1 = np.full((height, width), np.inf, dtype=np.float32)

# すべてのセルに対して、距離を計算
for i in range(height):
    for j in range(width):
        if (map_array[i, j] == 0) | (map_array[i, j] == 3):  # 通路のみ対象
            for ex, ey in exit_positions:
                # L1距離（マンハッタン）
                d1 = abs(i - ex) + abs(j - ey)
                # 最小値を保持
                dist_L1[i, j] = min(dist_L1[i, j], d1)

# SFFを保存
sff_path = "data/sff/distance_L1_12x12.npy"
np.save(sff_path, dist_L1)
print(f"\n✓ SFFを保存しました: {sff_path}")
print(f"  サイズ: {dist_L1.shape}")
print(f"  距離の範囲: [{np.nanmin(dist_L1[dist_L1 != np.inf]):.1f}, {np.nanmax(dist_L1[dist_L1 != np.inf]):.1f}]")
print(f"  有効セル数: {np.sum(dist_L1 != np.inf)}")

# 可視化用の情報を表示
print("\nマップの構造:")
print(map_array)
print("\nSFF（L1距離）の一部（通路部分）:")
print(dist_L1)


