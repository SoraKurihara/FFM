"""
統合Actor学習データ（H_actor_FINAL_*.pkl）を読み込み、内容を表示するスクリプト。
・出口からの距離帯（0: 1/3以内, 1: 1/3~2/3, 2: 2/3~）
・出口に対する角度（正面=0度）でグループ化し、H値を平均化して表示。
・集約結果を「(周囲の状況, 距離帯, 角度ビン) -> 上下左右待機の価値」形式で別pklに保存する。
"""
import math
import pickle
import os
from collections import defaultdict

import numpy as np

# 使用するActorデータのパス
H_ACTOR_PATH = os.path.join(
    "output", "logs", "unified_actor_training",
    "learning_corner_12", "H_actor_FINAL_total6600ep.pkl",
)
# マップ（出口位置・最大距離の計算用、学習時と同じ）
MAP_PATH = "data/maps/excel_map_001.npy"
# 角度のビン幅（度）。例: 45 → 8方向に集約
ANGLE_BIN_DEG = 45
# 集約結果の保存先（入力と同じディレクトリに同名ベースで _by_dist_angle.pkl を付与）
OUTPUT_PKL_SUFFIX = "_by_dist_angle.pkl"


class CompatibleUnpickler(pickle.Unpickler):
    """numpy 2.0以降で保存されたpickleをnumpy 1.xで読み込むためのUnpickler"""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.umath":
            module = "numpy.core.umath"
        return super().find_class(module, name)


def load_h_table(path):
    """Hテーブルを互換Unpicklerで読み込む"""
    with open(path, "rb") as f:
        return CompatibleUnpickler(f).load()


def get_exit_and_max_distance(map_path):
    """
    マップから出口位置と歩行可能セルにおける出口への最大L1距離を取得。
    戻り値: (exit_x, exit_y), max_l1_distance
    """
    if not os.path.isfile(map_path):
        return (0, 6), 17  # 12x12想定のフォールバック
    map_array = np.load(map_path)
    exits = np.argwhere(map_array == 3)
    if len(exits) == 0:
        return (0, 6), 17
    ex, ey = int(exits[0, 0]), int(exits[0, 1])
    walkable = np.argwhere((map_array == 0) | (map_array == 3))
    if len(walkable) == 0:
        return (ex, ey), 0
    l1 = np.abs(walkable[:, 0] - ex) + np.abs(walkable[:, 1] - ey)
    max_d = int(np.max(l1))
    return (ex, ey), max_d


def distance_band(l1_dist, max_l1):
    """
    出口からのL1距離を距離帯に変換。
    0: 最大距離の1/3以内, 1: 1/3~2/3, 2: 2/3超
    """
    if max_l1 <= 0:
        return 0
    r = l1_dist / max_l1
    if r <= 1.0 / 3.0:
        return 0
    if r <= 2.0 / 3.0:
        return 1
    return 2


def angle_to_exit_deg(x, y, ex, ey):
    """
    セル(x,y)から出口(ex,ey)への方向を角度で返す。正面=0度。
    出口が上側(行が小さい)なので、正面＝上方向＝0度とする。
    戻り値: -180 ~ 180 の範囲で正規化。
    """
    dx = ex - x
    dy = ey - y
    if dx == 0 and dy == 0:
        return 0.0
    # atan2(dy, dx): 右=0°, 上=-90°。出口方向(-1,0)は180°なので、180°引いて正面=0°
    rad = math.atan2(dy, dx)
    deg = math.degrees(rad) - 180.0
    while deg > 180:
        deg -= 360
    while deg <= -180:
        deg += 360
    return deg


def angle_bin_index(angle_deg, bin_deg):
    """角度をビン番号に。0 <= index < 360/bin_deg。"""
    # -180..180 を 0..360 相当にずらしてビン分割
    a = (angle_deg + 180) % 360
    return int(a // bin_deg) % (360 // bin_deg)


ACTION_LABELS = ("上", "下", "左", "右", "待機")


def format_h_values(h_list, labels=ACTION_LABELS):
    """行動価値リストをラベル付きで表示"""
    if not isinstance(h_list, (list, tuple, np.ndarray)):
        return str(h_list)
    parts = []
    for i, v in enumerate(h_list):
        lbl = labels[i] if i < len(labels) else "行動{}".format(i)
        parts.append("{}={:.4f}".format(lbl, float(v)))
    return " ".join(parts)


def top2_actions(mean_h, labels=ACTION_LABELS):
    """
    平均Hベクトルから価値が高い順に1位・2位の(ラベル, 値)を返す。
    戻り値: (1位のラベル, 1位の値), (2位のラベル, 2位の値)
    """
    arr = np.asarray(mean_h).ravel()
    n = min(len(arr), len(labels))
    if n == 0:
        return ("-", 0.0), ("-", 0.0)
    indices = np.argsort(-arr[:n])  # 降順
    i1, i2 = indices[0], indices[1] if n > 1 else indices[0]
    return (
        (labels[i1], float(arr[i1])),
        (labels[i2], float(arr[i2])),
    )


def main():
    path = H_ACTOR_PATH
    if not os.path.isfile(path):
        print("ファイルが見つかりません: {}".format(path))
        return

    print("=" * 70)
    print("統合Actor学習データ（距離帯・角度で集約・平均化）")
    print("=" * 70)
    print("ファイル: {}\n".format(path))

    # マップから出口と最大距離を取得（block_size=1 で block_idx = セル座標）
    exit_pos, max_l1 = get_exit_and_max_distance(MAP_PATH)
    ex, ey = exit_pos
    print("マップ: {}, 出口: {}, 最大L1距離: {}".format(MAP_PATH, exit_pos, max_l1))
    print("距離帯: 0=1/3以内, 1=1/3~2/3, 2=2/3~")
    print("角度: 正面(出口方向)=0度, ビン幅={}度\n".format(ANGLE_BIN_DEG))

    data = load_h_table(path)

    if not isinstance(data, dict):
        print("型: {}, 想定外のデータです".format(type(data)))
        return

    n_states = len(data)
    if n_states == 0:
        print("（エントリがありません）")
        return

    # (dist_band, angle_bin) -> list of H vectors（表示用）
    # (ranks_tuple, dist_band, angle_bin) -> list of H vectors（保存用）
    n_bins = 360 // ANGLE_BIN_DEG
    groups = defaultdict(list)
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
        d_band = distance_band(l1, max_l1)
        ang = angle_to_exit_deg(x, y, ex, ey)
        a_bin = angle_bin_index(ang, ANGLE_BIN_DEG)
        h_vec = np.asarray(h_values, dtype=np.float64)
        if h_vec.size >= 5:
            groups[(d_band, a_bin)].append(h_vec)
            # 保存用: 周囲の状況(ranks_tuple), 距離帯, 角度ビン でグループ化
            groups_detail[(ranks_tuple, d_band, a_bin)].append(h_vec)

    # 各グループで平均を計算
    summary = []
    for (d_band, a_bin), vecs in sorted(groups.items()):
        arr = np.array(vecs)
        mean_h = np.mean(arr, axis=0)
        summary.append((d_band, a_bin, len(vecs), mean_h))

    print("総状態数: {}".format(n_states))
    print("集約グループ数: {} (距離帯0,1,2 x 角度ビン{})\n".format(len(summary), n_bins))
    print("-" * 100)
    print("距離帯 | 角度ビン(中心度) | 状態数 | 平均H (上, 下, 左, 右, 待機) | 1位 | 2位")
    print("-" * 100)

    for d_band, a_bin, count, mean_h in summary:
        center_deg = (a_bin + 0.5) * ANGLE_BIN_DEG - 180
        h_str = format_h_values(mean_h.tolist())
        (lbl1, val1), (lbl2, val2) = top2_actions(mean_h)
        print("   {}   | {:7.1f}          | {:5d} | {} | {}={:.4f} | {}={:.4f}".format(
            d_band, center_deg, count, h_str, lbl1, val1, lbl2, val2))

    print("-" * 100)

    # 集約結果を元pklと同様の key->value 形式で別ファイルに保存
    # key = (周囲の状況=ranks_tuple, 距離帯, 角度ビン), value = [上, 下, 左, 右, 待機] の平均価値
    out_dict = {}
    for (ranks_tuple, d_band, a_bin), vecs in groups_detail.items():
        mean_h = np.mean(np.array(vecs), axis=0)
        out_dict[(ranks_tuple, d_band, a_bin)] = mean_h.tolist()

    out_dir = os.path.dirname(path)
    base = os.path.basename(path)
    if base.lower().endswith(".pkl"):
        out_name = base[:-4] + OUTPUT_PKL_SUFFIX
    else:
        out_name = base + OUTPUT_PKL_SUFFIX
    out_path = os.path.join(out_dir, out_name)

    try:
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("集約データを保存しました: {} ({} エントリ)".format(out_path, len(out_dict)))
    except Exception as e:
        print("保存に失敗しました: {}".format(e))

    print("処理完了")
    print("=" * 70)


if __name__ == "__main__":
    main()
