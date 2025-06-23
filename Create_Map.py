import matplotlib.pyplot as plt
import numpy as np

# サイズの定義
height = 50
width = 50

# すべて0で初期化（通路）
cell_map = np.zeros((height, width), dtype=int)

# 壁を2で設定（上下左右）
cell_map[0, :] = 2      # 上端
cell_map[-1, :] = 2     # 下端
cell_map[:, 0] = 2      # 左端
cell_map[:, -1] = 2     # 右端

# 出口を3で設定（一番上の中央に1マス）
center = width // 2
cell_map[0, center] = 3

# npyファイルとして保存
np.save("simple_room.npy", cell_map)

# 確認用の表示
plt.imshow(cell_map, cmap='gray_r')
plt.title("Cell Map")
plt.colorbar(label='Cell Type (0: path, 2: wall, 3: exit)')
plt.show()
