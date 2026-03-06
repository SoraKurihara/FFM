"""
マップ生成: 壁・出口・障害物を配置する。
障害物は「歩行可能な field」のセルのみから選ばれ、壁と同じ状態 2 で表現する。
出口(3)から到達できない 0（2で囲まれた島）ができないよう、接続性を保って配置する。

生成方法は2通り（第一引数が .xlsx かどうかで判定）:
  - Excel参照:  CreateMap("data/maps/maps.xlsx") または CreateMap("data/maps/maps.xlsx", sheet=0)
    → 第一引数が .xlsx なら Excel から field を読み込み。size/exit_pos は不要。
  - サイズ指定: CreateMap(size=5, exit_pos=[0, 2], obstacle_mode="random")
    → 第一引数を省略し、size と exit_pos はキーワードで指定（位置引数で size を渡すのは避ける）。
"""
import re
import numpy as np
from collections import deque
from pathlib import Path

# セル種別（障害物は壁と同じ 2）
FLOOR = 0
WALL = 2
EXIT = 3

# Excel 読み込み用（xlsx は openpyxl が必要: pip install openpyxl）
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _load_field_from_excel(excel_path, sheet=0):
    """data/maps/maps.xlsx 等の xlsx からマップ行列を読み込む。セル値は 0/2/3 を想定。"""
    if not _HAS_PANDAS:
        raise ImportError("Excel 読み込みには pandas が必要です。pip install pandas openpyxl")
    path = Path(excel_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    field = np.array(df, dtype=np.float64)
    # 欠損や小数を整数に
    field = np.nan_to_num(field, nan=0.0)
    return np.clip(field.astype(np.int64), 0, 3).astype(np.uint8)


class CreateMap:
    """
    コンストラクタは第一引数で判定:
      - 第一引数が .xlsx のパス → Excel から field を読み込み
      - 第一引数なし → size= と exit_pos= をキーワードで指定（サイズ指定）
    """

    def __init__(
        self,
        path=None,
        sheet=0,
        *,
        size=None,
        exit_pos=None,
        obstacle_mode=False,
    ):
        self.rng = np.random.default_rng()
        self.obstacle_mode = obstacle_mode

        use_excel = path is not None
        if use_excel:
            if isinstance(path, str) and path.lower().endswith(".xlsx"):
                self._from_excel = True
                self._init_from_excel(path, sheet)
            else:
                raise TypeError(
                    "第一引数は .xlsx のパスを指定するか、省略して size= と exit_pos= をキーワードで指定してください"
                )
        else:
            self._from_excel = False
            if size is None or exit_pos is None:
                raise TypeError("サイズ指定時は size= と exit_pos= をキーワードで指定してください")
            self._init_from_params(size, exit_pos)

    def _init_from_params(self, size: int, exit_pos):
        """サイズ・出口指定でフィールドを生成する。"""
        self.size = size
        self.exit_pos = np.asarray(exit_pos)
        field = np.full((size, size), FLOOR, dtype=np.uint8)
        field[0, :] = WALL
        field[-1, :] = WALL
        field[:, 0] = WALL
        field[:, -1] = WALL
        self.field = field
        self.locate_obstacles = np.empty((0, 2), dtype=np.intp)

    def _init_from_excel(self, excel_path, sheet):
        """Excel から field を読み込み、size / exit_pos / locate_obstacles を設定する。"""
        path = Path(excel_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        self._excel_path = path.resolve()
        self._excel_sheet = sheet
        self.field = _load_field_from_excel(str(self._excel_path), sheet)
        self.size = self.field.shape[0]  # 行数（縦）
        exits = np.argwhere(self.field == EXIT)
        if len(exits) == 0:
            self.exit_pos = np.array([0, 0])
        else:
            self.exit_pos = np.asarray(exits[0])
        # 外周以外の 2 を「障害物として置いたセル」として記録（任意）
        h, w = self.field.shape
        interior = np.argwhere(self.field == WALL)
        interior = interior[
            (interior[:, 0] > 0) & (interior[:, 0] < h - 1)
            & (interior[:, 1] > 0) & (interior[:, 1] < w - 1)
        ]
        self.locate_obstacles = interior if len(interior) else np.empty((0, 2), dtype=np.intp)

    def run(self):
        if self._from_excel:
            return  # Excel の内容をそのまま使うため何もしない
        self._assign_exit()
        self._assign_obstacles()

    def _assign_exit(self):
        """出口を field に設定する。"""
        self.field[tuple(self.exit_pos)] = EXIT

    def _get_field_cells(self):
        """
        障害物を置けるセル（壁・出口以外＝歩行可能な field）の (row, col) のリストを返す。
        """
        candidates = np.argwhere(self.field == FLOOR)
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=np.intp)
        return candidates

    def _is_connected(self):
        """
        出口(3)から、すべての FLOOR(0) に到達可能か（2で囲まれた0が無いか）を判定する。
        4近傍で BFS。歩行可能は 0 と 3。
        """
        h, w = self.field.shape
        walkable = (self.field == FLOOR) | (self.field == EXIT)
        n_walkable = int(np.sum(walkable))
        if n_walkable == 0:
            return True
        ex, ey = int(self.exit_pos[0]), int(self.exit_pos[1])
        if not (0 <= ex < h and 0 <= ey < w) or self.field[ex, ey] != EXIT:
            return False
        seen = np.zeros_like(self.field, dtype=bool)
        q = deque([(ex, ey)])
        seen[ex, ey] = True
        count = 1
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and walkable[nr, nc]:
                    seen[nr, nc] = True
                    count += 1
                    q.append((nr, nc))
        return count == n_walkable

    def _assign_obstacles(self):
        """
        障害物を壁(2)として配置する。
        出口から到達できない 0 ができないよう、1つ置くたびに接続性を確認し、
        接続が壊れる位置には置かない。
        """
        if self.obstacle_mode != "random":
            return

        candidates = self._get_field_cells()
        n_candidates = len(candidates)
        if n_candidates == 0:
            self.locate_obstacles = np.empty((0, 2), dtype=np.intp)
            return

        # 置く数を決め、候補をシャッフルして先頭から順に「置いても接続」なら置く
        num_to_try = int(self.rng.integers(0, n_candidates + 1))
        order = self.rng.permutation(n_candidates)
        placed = []

        for idx in order[:num_to_try]:
            i, j = int(candidates[idx, 0]), int(candidates[idx, 1])
            self.field[i, j] = WALL
            if self._is_connected():
                placed.append([i, j])
            else:
                self.field[i, j] = FLOOR

        self.locate_obstacles = np.array(placed, dtype=np.intp) if placed else np.empty((0, 2), dtype=np.intp)

    def _get_save_dir(self):
        """npy の保存先ディレクトリ（xlsx と同じ data/maps 等）を返す。"""
        if self._from_excel:
            return self._excel_path.parent
        return Path(__file__).resolve().parent / "data" / "maps"

    def _next_excel_map_number(self, directory):
        """保存先にある excel_map_*.npy の連番の次を返す（0始まり）。"""
        directory = Path(directory)
        pattern = re.compile(r"^excel_map_(\d{3})\.npy$")
        numbers = []
        if directory.exists():
            for p in directory.glob("excel_map_*.npy"):
                m = pattern.match(p.name)
                if m:
                    numbers.append(int(m.group(1)))
        return (max(numbers) + 1) if numbers else 0

    def _get_npy_basename(self, save_dir=None):
        """保存ファイル名を返す。Excel: 連番で excel_map_000.npy、自動: auto_map_12.npy / auto_map_12_ob.npy"""
        if self._from_excel:
            directory = save_dir if save_dir is not None else self._get_save_dir()
            n = self._next_excel_map_number(directory)
            return f"excel_map_{n:03d}.npy"
        if len(self.locate_obstacles) > 0:
            return f"auto_map_{self.size}_ob.npy"
        return f"auto_map_{self.size}.npy"

    def save_npy(self, save_dir=None):
        """セル情報（field）を .npy で保存する。保存先は xlsx と同じディレクトリ。"""
        directory = Path(save_dir) if save_dir is not None else self._get_save_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self._get_npy_basename(directory)
        np.save(path, self.field)
        self._last_saved_npy_path = path
        return path

    def run(self, save_npy=True):
        """マップを確定し、オプションで .npy に保存する。"""
        if self._from_excel:
            if save_npy:
                self.save_npy()
            return
        self._assign_exit()
        self._assign_obstacles()
        if save_npy:
            self.save_npy()


def main():
    import sys
    use_excel = "--excel" in sys.argv or "-e" in sys.argv

    if use_excel:
        # 第一引数が .xlsx → Excel 参照 → excel_map_000.npy 等で保存
        cm = CreateMap("data/maps/maps.xlsx", 0)
        cm.run()
        print("(Excel) size:", cm.size, "exit_pos:", cm.exit_pos)
    else:
        # 第一引数なし → auto_map_12.npy / auto_map_12_ob.npy 等で保存
        cm = CreateMap(size=12, exit_pos=[0, 6], obstacle_mode="random")
        cm.run()
        print("(size) locate_obstacles:", cm.locate_obstacles)
    saved = getattr(cm, "_last_saved_npy_path", None) or (cm._get_save_dir() / cm._get_npy_basename())
    print("saved:", saved)
    print("field:\n", cm.field)


if __name__ == "__main__":
    main()
