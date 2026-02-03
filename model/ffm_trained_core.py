import pickle
import random
import sys

import numpy as np

# numpy互換性のための設定
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray


class FloorFieldModel:
    """
    学習済みHテーブルを使用するFloor Field Model
    
    学習済みのActor（Hテーブル）から移動確率を計算して使用します。
    """
    
    def __init__(self, map_array, sff_path, N, h_table_path, params=None):
        """
        Args:
            map_array: マップ配列（0:歩行可能, 1:壁, 3:出口）
            sff_path: SFFファイルのパス
            N: エージェント数
            h_table_path: 学習済みHテーブルのパス（.pklファイル）
            params: パラメータ辞書
        """
        default_params = {
            "k_D": 1,   # DFF係数
            "k_A": 10,  # Actor係数
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "neumann",
            "block_size": 5,  # 状態エンコーディングのブロックサイズ
        }
        self.params = default_params if params is None else {**default_params, **params}
        self.map_array = map_array.astype(np.uint8)
        
        # SFFの読み込み
        sff_loaded = np.load(sff_path, mmap_mode="r")
        # SFFのinf値を0に置き換え
        self.sff = np.where(np.isinf(sff_loaded), 0.0, sff_loaded).astype(np.float32)
        
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)
        self.N = N
        self.positions = self.initialize_agents()
        self.neighbors = self.get_neighbors()
        self.block_size = self.params["block_size"]
        
        # 学習済みHテーブルの読み込みとキーの正規化
        with open(h_table_path, "rb") as f:
            h_table_pickled = pickle.load(f)
            
        self.H = {}
        for k_bytes, v in h_table_pickled.items():
            # 1. 保存されていたバイト列キーを復元（中身はNumpy型を含んでいる可能性あり）
            # 構造は ((r, r, r, r), (bx, by)) を想定
            original_key_structure = pickle.loads(k_bytes)
            
            # 2. 中身を強制的に Python 標準の int に変換してタプル化
            # original_key_structure[0] は ranks, [1] は block_idx
            ranks_tuple = tuple(int(r) for r in original_key_structure[0])
            block_idx_tuple = (int(original_key_structure[1][0]), int(original_key_structure[1][1]))
            
            # 3. 新しいキーとして保存（バイト列ではなくタプルそのものをキーにする）
            clean_key = (ranks_tuple, block_idx_tuple)
            self.H[clean_key] = v
            
        print(f"✓ 学習済みHテーブルを読み込みました: {len(self.H)}状態")
        
    def initialize_agents(self):
        """エージェントの初期配置を生成"""
        free_cells = np.argwhere(self.map_array == 0)
        selected = free_cells[np.random.choice(len(free_cells), self.N, replace=False)]
        return selected
    
    def get_neighbors(self):
        """近傍セルのオフセットを取得"""
        if self.params["neighborhood"] == "neumann":
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            return [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
    
    def _encode_state(self, x, y, state_map):
        """
        上下左右ランク状態エンコーディング
        
        Returns:
            tuple: (ranks_tuple, block_idx_tuple) の形式でハッシュ可能なキー
        """
        height, width = state_map.shape
        ranks = []
        
        # 上下左右の4方向をチェック
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        
        for dx, dy in directions:
            rank = 3  # デフォルトはランク3
            
            # 1歩先（隣接セル）をチェック
            nx1, ny1 = x + dx, y + dy
            if 0 <= nx1 < height and 0 <= ny1 < width:
                val1 = state_map[nx1, ny1]
                # 移動不可（壁(2)または人(1)）ならランク0
                if val1 == 2 or val1 == 1:
                    rank = 0
                else:
                    # 2. 斜め前をチェック（上方向なら、上のセルの左右）
                    # 斜め前の座標を計算
                    if dx != 0:  # 上下方向
                        diag_left = (nx1, ny1 - 1)  # 斜め左
                        diag_right = (nx1, ny1 + 1)  # 斜め右
                    else:  # 左右方向
                        diag_left = (nx1 - 1, ny1)  # 斜め上
                        diag_right = (nx1 + 1, ny1)  # 斜め下
                    
                    # 斜め前に人がいるかチェック（左右問わない）
                    has_person_diag = False
                    for diag_x, diag_y in [diag_left, diag_right]:
                        if 0 <= diag_x < height and 0 <= diag_y < width:
                            if state_map[diag_x, diag_y] == 1:  # 人がいる
                                has_person_diag = True
                                break
                    
                    if has_person_diag:
                        rank = 1
                    else:
                        # 3. 2歩先をチェック
                        nx2, ny2 = x + 2 * dx, y + 2 * dy
                        if 0 <= nx2 < height and 0 <= ny2 < width:
                            val2 = state_map[nx2, ny2]
                            # 壁(2)または人(1)ならランク2
                            if val2 == 2 or val2 == 1:
                                rank = 2
                        else:
                            # 境界外は壁として扱う
                            rank = 2
            else:
                # 境界外は移動不可としてランク0
                rank = 0
            
            ranks.append(rank)
        
        # 粗い位置情報（ブロックインデックス）
        # ここで必ず Python の int に変換する
        block_idx = (int(x // self.block_size), int(y // self.block_size))
        
        # ranks も int のタプルに変換
        ranks_tuple = tuple(int(r) for r in ranks)
        
        # pickle.dumps ではなく、タプルをそのまま返す
        return (ranks_tuple, block_idx)
    
    def step(self):
        """1ステップ実行"""
        move_requests = {}
        next_positions = np.copy(self.positions)
        
        # マップ値配列を作成（0:歩行可能, 1:歩行者, 2:壁, 3:出口）
        state_map = self.map_array.copy()
        for pos in self.positions:
            state_map[pos[0], pos[1]] = 1  # 歩行者を1に設定
        
        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]
            
            # 現在の状態をエンコード
            state = self._encode_state(x, y, state_map)
            
            # 移動候補に[0,0]を最初から含める
            offsets = np.array(self.neighbors + [(0, 0)])
            all_coords = self.positions[idx] + offsets
            
            # occupied set（自分を除外）
            other_positions = np.delete(self.positions, idx, axis=0)
            occupied = {(int(p[0]), int(p[1])) for p in other_positions}
            
            # 有効性マスクを計算（境界内、移動可能（0または3）、占有されていない）
            height, width = self.map_array.shape
            valid_boundary = np.zeros(len(all_coords), dtype=bool)
            valid_map = np.zeros(len(all_coords), dtype=bool)
            valid_occupied = np.zeros(len(all_coords), dtype=bool)
            
            for i, coord in enumerate(all_coords):
                # 境界チェック
                if 0 <= coord[0] < height and 0 <= coord[1] < width:
                    valid_boundary[i] = True
                    # 移動可能セル（0または3）
                    map_val = self.map_array[coord[0], coord[1]]
                    valid_map[i] = (map_val == 0) or (map_val == 3)
                    # 占有チェック（現在地[0,0]は常に有効）
                    if i == len(all_coords) - 1:  # 現在地（最後の要素）
                        valid_occupied[i] = True
                    else:
                        valid_occupied[i] = tuple(coord) not in occupied
            
            all_valid_mask = valid_boundary & valid_map & valid_occupied
            
            # 出口チェック（現在地を除く）
            exit_mask = np.zeros(len(all_coords), dtype=bool)
            for i in range(len(all_coords) - 1):  # 現在地を除く
                if (
                    valid_boundary[i]
                    and self.map_array[all_coords[i][0], all_coords[i][1]] == 3
                ):
                    exit_mask[i] = True
            
            if np.any(exit_mask):
                chosen_coord = all_coords[exit_mask][0]
                if tuple(chosen_coord) not in move_requests:
                    move_requests[tuple(chosen_coord)] = []
                move_requests[tuple(chosen_coord)].append(idx)
                continue
            
            # Hテーブルから移動確率を計算
            # DFF値を取得（全方向に対して）
            dff_vals = np.array(
                [self.dff[coord[0], coord[1]] for coord in all_coords]
            )
            
            # Hテーブルから行動価値を取得
            state_key = state
            fixed_action_size = len(all_coords)
            if (
                state_key not in self.H
                or len(self.H[state_key]) != fixed_action_size
            ):
                # Hテーブルに状態がない場合、0で初期化
                h_vals = np.zeros(fixed_action_size, dtype=np.float32)
            else:
                h_vals = np.array(self.H[state_key], dtype=np.float32)
            
            # H値をSFFスケールに規格化（Hテーブル自体は変更しない）
            if len(self.H) > 0:
                all_h_values = []
                for h_list in self.H.values():
                    if isinstance(h_list, list):
                        all_h_values.extend(h_list)
                    else:
                        all_h_values.append(h_list)
                
                if all_h_values:
                    h_table_min = float(np.min(all_h_values))
                    h_table_max = float(np.max(all_h_values))
                    sff_min = float(np.min(self.sff))
                    sff_max = float(np.max(self.sff))
                    
                    if not (
                        np.isnan(h_table_min)
                        or np.isnan(h_table_max)
                        or np.isinf(h_table_min)
                        or np.isinf(h_table_max)
                    ):
                        if h_table_max - h_table_min > 1e-6:
                            h_vals_normalized = (
                                (h_table_max - h_vals)
                                / (h_table_max - h_table_min)
                            ) * (sff_max - sff_min) + sff_min
                            h_vals = h_vals_normalized
            
            # 全方向のスコアを計算（無効な方向も含む）
            score = (
                -self.params["k_A"] * h_vals
                + self.params["k_D"] * dff_vals
            )
            
            # NaN/Infチェック
            if np.any(np.isnan(score)) or np.any(np.isinf(score)):
                # 不正な値がある場合は有効な行動のみで均一に
                valid_indices = np.where(all_valid_mask)[0]
                if len(valid_indices) > 0:
                    score = np.zeros_like(score)
                    score[valid_indices] = 1.0
                else:
                    score = np.zeros_like(score)
            
            # 全方向のスコアから確率分布を計算
            score_max = np.max(score)
            probs = np.exp(score - score_max)
            
            # 無効な行動の確率を0に設定（mask適用）
            probs[~all_valid_mask] = 0.0
            
            # 正規化
            probs_sum = probs.sum()
            if np.isfinite(probs_sum) and probs_sum > 0:
                probs /= probs_sum
            else:
                # フォールバック: 有効な行動から均一に選択
                valid_indices = np.where(all_valid_mask)[0]
                if len(valid_indices) > 0:
                    probs = np.zeros_like(score)
                    probs[valid_indices] = 1.0 / len(valid_indices)
                else:
                    probs = np.zeros_like(score)
            
            # 確率分布に従って行動を選択
            chosen_idx = np.random.choice(len(all_coords), p=probs)
            chosen = tuple(all_coords[chosen_idx])
            
            if chosen not in move_requests:
                move_requests[chosen] = []
            move_requests[chosen].append(idx)
        
        # 衝突処理
        for target, agents in move_requests.items():
            if len(agents) == 1:
                next_positions[agents[0]] = target
                self.dff[
                    self.positions[agents[0]][0], self.positions[agents[0]][1]
                ] += 1
            else:
                # 衝突発生: 必ずだれか一人がそのセルに入れる
                chosen = random.choice(agents)
                next_positions[chosen] = target
                self.dff[
                    self.positions[chosen][0], self.positions[chosen][1]
                ] += 1
        
        # 出口に到達した歩行者を除外
        keep_mask = (
            self.map_array[next_positions[:, 0], next_positions[:, 1]] != 3
        )
        self.positions = next_positions[keep_mask]
        
        self.update_dff()
    
    def update_dff(self):
        """DFFの更新"""
        diffuse = self.params["diffuse"]
        decay = self.params["decay"]
        new_dff = (1 - decay) * (1 - diffuse) * self.dff
        
        padded = np.pad(new_dff, 1, mode="constant")
        for dx, dy in self.neighbors:
            new_dff += (
                decay
                * (1 - diffuse)
                / len(self.neighbors)
                * padded[
                    1 + dx : new_dff.shape[0] + 1 + dx,
                    1 + dy : new_dff.shape[1] + 1 + dy,
                ]
            )
        
        self.dff = new_dff
        threshold = 1e-4
        self.dff[self.dff < threshold] = 0
    
    def run(self, save_prefix=None, save_interval=100, max_steps=None):
        """
        シミュレーションを実行
        
        Args:
            save_prefix: 保存用プレフィックス
            save_interval: 保存間隔
            max_steps: 最大ステップ数（Noneの場合は無制限）
        
        Returns:
            int: 実行ステップ数
        """
        buffer = []
        step = 0
        
        while self.positions.shape[0] > 0:
            if max_steps is not None and step >= max_steps:
                break
            
            self.step()
            buffer.append(np.copy(self.positions))
            step += 1
            
            if save_prefix and (step % save_interval == 0):
                np.savez_compressed(
                    f"{save_prefix}_{step}.npz",
                    positions=np.array(buffer, dtype=np.int32),
                )
                buffer = []
        
        if save_prefix and buffer:
            np.savez_compressed(
                f"{save_prefix}_final.npz",
                positions=np.array(buffer, dtype=np.int32),
            )
        
        return step