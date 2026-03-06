"""
指定したマップの .npy から Lp ノルム（静的フロアフィールド）を計算し、
data/sff に <元のnpyのベース名>_L1.npy 等で保存する。
障害物・壁は通過せず、歩行可能セル(0)と出口(3)のみで距離を計算する。

対応ノルム:
  - L1: 実装済み（4近傍 BFS による最短経路長、障害物対応）
  - L2: 枠のみ（未実装）
  - Linf: 枠のみ（未実装）
"""
import warnings
import numpy as np
from pathlib import Path
from collections import deque

# マップのセル種別（createMap と一致）
FLOOR = 0
WALL = 2
EXIT = 3


def _load_map(npy_path):
    """.npy を読み込み uint8 のマップ配列を返す。"""
    path = Path(npy_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    arr = np.load(path)
    return np.asarray(arr, dtype=np.uint8)


def _find_exit(map_array):
    """出口(3)の座標を返す。複数ある場合は最初の一つ。"""
    exits = np.argwhere(map_array == EXIT)
    if len(exits) == 0:
        raise ValueError("マップに出口(3)がありません")
    return (int(exits[0, 0]), int(exits[0, 1]))


def compute_L1(map_array, exit_pos):
    """
    出口から各歩行可能セルまでの L1 距離（4近傍での最短経路長）を計算する。
    障害物・壁(2)は通過しない。到達不能は np.inf。
    戻り値: float の 2次元配列（出口は 0、壁・障害物は inf）
    """
    h, w = map_array.shape
    walkable = (map_array == FLOOR) | (map_array == EXIT)
    dist = np.full((h, w), np.inf, dtype=np.float64)
    ex, ey = exit_pos
    if not (0 <= ex < h and 0 <= ey < w) or map_array[ex, ey] != EXIT:
        return dist
    dist[ex, ey] = 0.0
    q = deque([(ex, ey)])
    while q:
        r, c = q.popleft()
        d = dist[r, c]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and np.isinf(dist[nr, nc]):
                dist[nr, nc] = d + 1.0
                q.append((nr, nc))
    return dist


def compute_L2(map_array, exit_pos):
    """
    L2（ユークリッド）に基づく距離場。障害物対応は未実装。
    枠のみ用意し、中身は未実装。
    """
    # TODO: 障害物を考慮した測地距離（Fast Marching 等）
    raise NotImplementedError("L2 は未実装です")


def compute_Linf(map_array, exit_pos):
    """
    L∞（チェビシェフ）に基づく距離場。障害物対応は未実装。
    枠のみ用意し、中身は未実装。
    """
    # TODO: 障害物を考慮した測地距離
    raise NotImplementedError("Linf は未実装です")


class ComputeSFFLp:
    """
    マップ .npy を読み込み、Lp ノルムの SFF を計算して data/sff に保存する。
    CreateMap と同様に、第一引数にマップの .npy パスを指定する。
    """

    def __init__(self, map_npy_path):
        """
        map_npy_path: マップの .npy ファイルパス（例: data/maps/excel_map_000.npy）
        """
        path = Path(map_npy_path)
        if not path.suffix.lower() == ".npy":
            raise TypeError("マップは .npy ファイルを指定してください")
        self._map_path = path if path.is_absolute() else Path(__file__).resolve().parent / path
        self.map_array = _load_map(self._map_path)
        self.exit_pos = _find_exit(self.map_array)

    def _get_output_dir(self):
        """保存先: data/sff（スクリプト基準）。"""
        return Path(__file__).resolve().parent / "data" / "sff"

    def _get_stem(self):
        """元の npy の拡張子を除いたファイル名（例: excel_map_000）。"""
        return self._map_path.stem

    def _output_path(self, norm_name):
        """例: data/sff/excel_map_000_L1.npy"""
        return self._get_output_dir() / f"{self._get_stem()}_{norm_name}.npy"

    def run_L1(self, save=True):
        """L1 を計算し、data/sff/<stem>_L1.npy に保存する。"""
        sff = compute_L1(self.map_array, self.exit_pos)
        if save:
            out_dir = self._get_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            path = self._output_path("L1")
            np.save(path, sff)
            self._last_saved_L1_path = path
            return path
        return sff

    def run_L2(self, save=True):
        """L2 は未実装。枠のみ。"""
        # compute_L2(...) 実装後に保存パスは self._output_path("L2")
        raise NotImplementedError("L2 は未実装です")

    def run_Linf(self, save=True):
        """Linf は未実装。枠のみ。"""
        # compute_Linf(...) 実装後に保存パスは self._output_path("Linf")
        raise NotImplementedError("Linf は未実装です")

    def run(self, norms=("L1",), save=True):
        """
        指定したノルムを計算して保存する。
        norms: 計算するノルムのリスト。例: ("L1",) または ("L1", "L2", "Linf")
        L2 / Linf は未実装のため指定してもスキップする。
        """
        saved = []
        if "L1" in norms:
            p = self.run_L1(save=save)
            if save:
                saved.append(p)
        if "L2" in norms:
            warnings.warn("L2 は未実装のためスキップしました", UserWarning)
        if "Linf" in norms:
            warnings.warn("Linf は未実装のためスキップしました", UserWarning)
        return saved


def main():
    import sys
    if len(sys.argv) < 2:
        print("使い方: python compute_sff_lp.py <マップの.npyパス>")
        print("例: python compute_sff_lp.py data/maps/excel_map_000.npy")
        sys.exit(1)
    map_path = sys.argv[1]
    cm = ComputeSFFLp(map_path)
    saved = cm.run(norms=("L1",), save=True)
    print("exit_pos:", cm.exit_pos)
    print("saved:", saved)


if __name__ == "__main__":
    main()
