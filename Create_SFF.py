import matplotlib.pyplot as plt
import numpy as np

# ファイルの読み込み
cell_map = np.load(r"data/maps/simple_room.npy")

# 出口（値が3）の座標を取得
exit_positions = np.argwhere(cell_map == 3)

# マップのサイズ
height, width = cell_map.shape

# 初期化（無限大で埋める）
dist_L1 = np.full((height, width), np.inf)
dist_L2 = np.full((height, width), np.inf)
dist_Linf = np.full((height, width), np.inf)

# すべてのセルに対して、距離を計算
for i in range(height):
    for j in range(width):
        if (cell_map[i, j] == 0) | (cell_map[i, j] == 3):  # 通路のみ対象
            for ex, ey in exit_positions:
                # L1距離（マンハッタン）
                d1 = abs(i - ex) + abs(j - ey)
                # L2距離（ユークリッド）
                d2 = np.hypot(i - ex, j - ey)
                # L∞距離（チェビシェフ）
                d_inf = max(abs(i - ex), abs(j - ey))

                # 最小値を保持
                dist_L1[i, j] = min(dist_L1[i, j], d1)
                dist_L2[i, j] = min(dist_L2[i, j], d2)
                dist_Linf[i, j] = min(dist_Linf[i, j], d_inf)



# 保存（npy形式）
np.save("distance_L1.npy", dist_L1)
np.save("distance_L2.npy", dist_L2)
np.save("distance_Linf.npy", dist_Linf)

# 可視化
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].imshow(dist_L1, cmap='viridis')
axs[0].set_title("L1 Distance")
axs[1].imshow(dist_L2, cmap='viridis')
axs[1].set_title("L2 Distance")
axs[2].imshow(dist_Linf, cmap='viridis')
axs[2].set_title("L∞ Distance")
plt.suptitle("Distance Maps from Exits")
plt.tight_layout()
plt.show()
