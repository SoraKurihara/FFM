"""
FFM+AC（集約Actor版）: 集約pklの (周囲の状況, 距離帯, 角度ビン) -> 行動価値 を用いて
確率計算。Actor・Criticともに学習（HはTD誤差で更新、VはTD学習）。
当てはまるkeyがなければランダムウォーク。
"""
import math
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

# 集約pklの角度ビン幅（inspect_actor_data.py と一致）
ANGLE_BIN_DEG = 45


class CompatibleUnpickler(pickle.Unpickler):
    """numpy 2.x で保存された pickle を読み込むための Unpickler"""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        elif module == "numpy._core.umath":
            module = "numpy.core.umath"
        return super().find_class(module, name)


def _distance_band(l1_dist, max_l1):
    """出口からのL1距離を距離帯に。0: 1/3以内, 1: 1/3~2/3, 2: 2/3~"""
    if max_l1 <= 0:
        return 0
    r = l1_dist / max_l1
    if r <= 1.0 / 3.0:
        return 0
    if r <= 2.0 / 3.0:
        return 1
    return 2


def _angle_to_exit_deg(x, y, ex, ey):
    """セル(x,y)から出口(ex,ey)への方向を角度で。正面=0度。"""
    dx = ex - x
    dy = ey - y
    if dx == 0 and dy == 0:
        return 0.0
    rad = math.atan2(dy, dx)
    deg = math.degrees(rad) - 180.0
    while deg > 180:
        deg -= 360
    while deg <= -180:
        deg += 360
    return deg


def _angle_bin_index(angle_deg, bin_deg=ANGLE_BIN_DEG):
    """角度をビン番号に。0 <= index < 360/bin_deg。"""
    a = (angle_deg + 180) % 360
    return int(a // bin_deg) % (360 // bin_deg)


def _exits_and_max_l1(map_array):
    """マップからすべての出口位置と、歩行可能セルでの最寄り出口までの最大L1距離を取得。"""
    exits = np.argwhere(map_array == 3)
    if len(exits) == 0:
        return [(0, 6)], 17
    exit_list = [(int(r), int(c)) for r, c in exits]
    
    walkable = np.argwhere((map_array == 0) | (map_array == 3))
    if len(walkable) == 0:
        return exit_list, 0
        
    exits_array = np.array(exit_list)
    distances = np.abs(walkable[:, np.newaxis, 0] - exits_array[:, 0]) + \
                np.abs(walkable[:, np.newaxis, 1] - exits_array[:, 1])
    min_distances = np.min(distances, axis=1)
    
    return exit_list, int(np.max(min_distances))

def _get_nearest_exit(x, y, exits):
    """セル(x, y)から最も近い出口の座標と距離を返す"""
    min_dist = float('inf')
    nearest = exits[0]
    for ex, ey in exits:
        dist = abs(x - ex) + abs(y - ey)
        if dist < min_dist:
            min_dist = dist
            nearest = (ex, ey)
    return nearest, min_dist


def _normalize_aggregated_key(key):
    """(ranks_tuple, d_band, a_bin) を Python int のタプルに正規化。"""
    ranks_tuple, d_band, a_bin = key
    ranks = tuple(int(r) for r in ranks_tuple)
    return (ranks, int(d_band), int(a_bin))


class FloorFieldModelACAggregated:
    """
    FFM+AC（集約Actor版）。
    Actor: 集約pklの (ranks_tuple, 距離帯, 角度ビン) で lookup、あればその価値で行動、なければランダムウォーク。
    Critic: TD学習で V(s) を更新。
    """

    def __init__(
        self,
        map_array,
        sff_path,
        N,
        aggregated_h_path,
        pretrained_v_path=None,
        params=None,
    ):
        default_params = {
            "k_D": 1,
            "k_A": 10,
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "neumann",
            "alpha_v": 0.1,
            "alpha_h": 0.1,
            "gamma": 0.95,
            "exit_reward": 100.0,
            "step_penalty": 0.0,
            "collision_penalty": -1.0,
            "block_size": 1,
        }
        self.params = (
            default_params if params is None else {**default_params, **params}
        )

        self.map_array = map_array.astype(np.uint8)
        self.exit_positions, self.max_l1 = _exits_and_max_l1(self.map_array)

        sff_loaded = np.load(sff_path, mmap_mode="r")
        self.sff = np.where(np.isinf(sff_loaded), 0.0, sff_loaded).astype(
            np.float32
        )
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)
        self.N = N
        self.positions = self.initialize_agents()
        self.neighbors = self.get_neighbors()
        self.block_size = self.params["block_size"]
        self.alpha_v = self.params["alpha_v"]
        self.alpha_h = self.params["alpha_h"]
        self.gamma = self.params["gamma"]

        # 集約pkl: データがないときの参照用（key=周囲+距離+角度、更新しない）
        with open(aggregated_h_path, "rb") as f:
            raw = CompatibleUnpickler(f).load()
        self.H_aggregated = {}
        for k, v in raw.items():
            try:
                nk = _normalize_aggregated_key(k)
                self.H_aggregated[nk] = list(v)[:5]
            except Exception:
                continue
        # H は既存データ使用時・新規学習時にのみ書き込む
        self.H = {}
        self.initial_h_size = 0
        print("✓ 集約Actorを読み込みました（参照用）: {} エントリ".format(
            len(self.H_aggregated)))

        # Critic
        if pretrained_v_path:
            with open(pretrained_v_path, "rb") as f:
                pv = pickle.load(f)
            self.V = defaultdict(lambda: 0.0)
            for k, val in pv.items():
                try:
                    key = pickle.loads(k) if isinstance(k, bytes) else k
                    if not isinstance(key, (tuple, list)) or len(key) != 2:
                        continue
                    r = tuple(int(x) for x in key[0])
                    b = (int(key[1][0]), int(key[1][1]))
                    self.V[(r, b)] = float(val)
                except Exception:
                    continue
            self.initial_v_size = len(self.V)
            print("✓ 事前学習済みCritic: {} 状態".format(self.initial_v_size))
        else:
            self.V = defaultdict(lambda: 0.0)
            self.initial_v_size = 0

    def initialize_agents(self, exit_positions=None, radius=None):
        if exit_positions is None or radius is None:
            free_cells = np.argwhere(self.map_array == 0)
            n = min(self.N, len(free_cells))
            if n == 0:
                return np.empty((0, 2), dtype=np.int32)
            idx = np.random.choice(len(free_cells), n, replace=False)
            return free_cells[idx]
        
        free_cells = np.argwhere(self.map_array == 0)
        if len(free_cells) == 0:
            return np.empty((0, 2), dtype=np.int32)

        # リストで受け取った複数の出口をNumPy配列に変換
        exits_array = np.array(exit_positions)
        # 全フリーセル ⇔ 全出口の距離を一気に計算
        distances = np.abs(free_cells[:, np.newaxis, 0] - exits_array[:, 0]) + \
                    np.abs(free_cells[:, np.newaxis, 1] - exits_array[:, 1])
        min_distances = np.min(distances, axis=1)
        
        # 最も近い出口からの距離がradius以下のセルを抽出
        mask = min_distances <= radius
        rc = free_cells[mask]
        n = min(self.N, len(rc))
        if n == 0:
            return np.empty((0, 2), dtype=np.int32)
        return rc[np.random.choice(len(rc), n, replace=False)]

    def get_neighbors(self):
        if self.params["neighborhood"] == "neumann":
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [
            (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]

    def _encode_state(self, x, y, state_map):
        """上下左右ランクとブロックインデックス。ffm_unified と同じ仕様。"""
        height, width = state_map.shape
        ranks = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            rank = 3
            nx1, ny1 = x + dx, y + dy
            if 0 <= nx1 < height and 0 <= ny1 < width:
                val1 = state_map[nx1, ny1]
                if val1 == 2 or val1 == 1:
                    rank = 0
                else:
                    if dx != 0:
                        diag_left, diag_right = (nx1, ny1 - 1), (nx1, ny1 + 1)
                    else:
                        diag_left, diag_right = (nx1 - 1, ny1), (nx1 + 1, ny1)
                    has_person_diag = any(
                        0 <= ax < height and 0 <= ay < width
                        and state_map[ax, ay] == 1
                        for ax, ay in [diag_left, diag_right]
                    )
                    if has_person_diag:
                        rank = 1
                    else:
                        nx2, ny2 = x + 2 * dx, y + 2 * dy
                        if 0 <= nx2 < height and 0 <= ny2 < width:
                            if state_map[nx2, ny2] in (2, 1):
                                rank = 2
                        else:
                            rank = 2
            else:
                rank = 0
            ranks.append(rank)
        ranks_tuple = tuple(int(r) for r in ranks)
        block_idx = (int(x // self.block_size), int(y // self.block_size))
        return (ranks_tuple, block_idx)

    def step(self):
        move_requests = {}
        next_positions = np.copy(self.positions)
        state_map = self.map_array.copy()
        for pos in self.positions:
            state_map[pos[0], pos[1]] = 1

        states = {}
        will_exit = {}
        actions_idx = {}
        action_valid_mask = {}

        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]
            state = self._encode_state(x, y, state_map)
            states[idx] = state
            ranks_tuple, block_idx = state

            offsets = np.array(self.neighbors + [(0, 0)])
            all_coords = self.positions[idx] + offsets
            other_positions = np.delete(self.positions, idx, axis=0)
            occupied = {(int(p[0]), int(p[1])) for p in other_positions}

            height, width = self.map_array.shape
            all_valid_mask = np.ones(len(all_coords), dtype=bool)
            for i, coord in enumerate(all_coords):
                if not (0 <= coord[0] < height and 0 <= coord[1] < width):
                    all_valid_mask[i] = False
                else:
                    if self.map_array[coord[0], coord[1]] not in (0, 3):
                        all_valid_mask[i] = False
                    if i < len(all_coords) - 1 and tuple(coord) in occupied:
                        all_valid_mask[i] = False

            exit_mask = np.zeros(len(all_coords), dtype=bool)
            for i in range(len(all_coords) - 1):
                if (all_valid_mask[i]
                        and self.map_array[all_coords[i][0], all_coords[i][1]]
                        == 3):
                    exit_mask[i] = True

            if np.any(exit_mask):
                will_exit[idx] = True
                chosen_coord = all_coords[exit_mask][0]
                if tuple(chosen_coord) not in move_requests:
                    move_requests[tuple(chosen_coord)] = []
                move_requests[tuple(chosen_coord)].append(idx)
                continue

            dff_vals = np.array([
                self.dff[coord[0], coord[1]] for coord in all_coords
            ])

            # 参照順: 学習テーブル H(周囲+座標) → 無ければ集約(周囲+距離+角度) → ランダム
            state_key = (ranks_tuple, (int(block_idx[0]), int(block_idx[1])))
            h_vals = None  # このエージェント用
            if state_key in self.H:
                h_list = self.H[state_key]
                h_vals = np.array(h_list, dtype=np.float64)
                if len(h_vals) < len(all_coords):
                    h_vals = np.resize(h_vals, len(all_coords))
            else:
                # エージェントから一番近い出口を計算
                (nearest_ex, nearest_ey), nearest_dist = _get_nearest_exit(x, y, self.exit_positions)
                agg_key = (
                    ranks_tuple,
                    _distance_band(nearest_dist, self.max_l1),
                    _angle_bin_index(
                        _angle_to_exit_deg(x, y, nearest_ex, nearest_ey)
                    ),
                )
                if agg_key in self.H_aggregated:
                    h_list = list(self.H_aggregated[agg_key])[:5]
                    self.H[state_key] = h_list
                    h_vals = np.array(h_list, dtype=np.float64)
                    if len(h_vals) < len(all_coords):
                        h_vals = np.resize(h_vals, len(all_coords))
                else:
                    score = np.ones(len(all_coords), dtype=np.float64)
                    score[~all_valid_mask] = -1e9

            if h_vals is not None:
                # 最大値を0に、最小値を最大値に: 正規化してからスコア計算
                h_max = np.max(h_vals)
                h_min = np.min(h_vals)
                if h_max - h_min > 1e-9:
                    h_vals = h_max - h_vals
                score = (
                    -self.params["k_A"] * h_vals
                    + self.params["k_D"] * dff_vals
                )
            # h_vals is None のときは score は既にランダム用に設定済み

            if np.any(np.isnan(score)) or np.any(np.isinf(score)):
                valid_indices = np.where(all_valid_mask)[0]
                score = np.zeros_like(score)
                if len(valid_indices) > 0:
                    score[valid_indices] = 1.0

            score[~all_valid_mask] = -1e9
            score_max = np.max(score)
            probs = np.exp(score - score_max)
            probs[~all_valid_mask] = 0.0
            s = probs.sum()
            if np.isfinite(s) and s > 0:
                probs = probs / s
            else:
                valid_indices = np.where(all_valid_mask)[0]
                probs = np.zeros_like(score)
                if len(valid_indices) > 0:
                    probs[valid_indices] = 1.0 / len(valid_indices)

            chosen_idx = int(np.random.choice(len(all_coords), p=probs))
            chosen = tuple(all_coords[chosen_idx])
            if chosen not in move_requests:
                move_requests[chosen] = []
            move_requests[chosen].append(idx)

            # Actor学習用に記録（保存は 周囲+座標 の H で行う）
            actions_idx[idx] = chosen_idx
            action_valid_mask[idx] = np.array(all_valid_mask, dtype=bool)

        collision_counts = {}
        for target, agents in move_requests.items():
            if len(agents) == 1:
                next_positions[agents[0]] = target
                self.dff[
                    self.positions[agents[0]][0],
                    self.positions[agents[0]][1],
                ] += 1
                collision_counts[agents[0]] = 0
            else:
                collision_count = len(agents) - 1
                chosen = random.choice(agents)
                next_positions[chosen] = target
                self.dff[
                    self.positions[chosen][0], self.positions[chosen][1]
                ] += 1
                collision_counts[chosen] = collision_count
                for agent in agents:
                    if agent != chosen:
                        collision_counts[agent] = collision_count

        state_map_next = self.map_array.copy()
        for pos in next_positions:
            if self.map_array[pos[0], pos[1]] != 3:
                state_map_next[pos[0], pos[1]] = 1

        # Critic 更新 & TD誤差を収集（Actor更新用）
        td_errors = {}
        for idx in states:
            state = states[idx]
            reward = self.params["step_penalty"]
            if idx in will_exit and will_exit[idx]:
                reward += self.params["exit_reward"]
            if idx in collision_counts:
                reward += collision_counts[idx] * self.params["collision_penalty"]
            if idx in will_exit and will_exit[idx]:
                v_next = 0.0
            else:
                x_next, y_next = next_positions[idx]
                state_next = self._encode_state(
                    x_next, y_next, state_map_next
                )
                v_next = self.V[state_next]
            v_cur = self.V[state]
            td_error = reward + self.gamma * v_next - v_cur
            td_errors[idx] = td_error
            self.V[state] = v_cur + self.alpha_v * td_error

        # Actor 更新（H の key は 周囲の状況+座標）
        self._update_actor(states, actions_idx, action_valid_mask, td_errors)

        keep_mask = (
            self.map_array[next_positions[:, 0], next_positions[:, 1]] != 3
        )
        self.positions = next_positions[keep_mask]
        self.update_dff()

    def _update_actor(
        self, states, actions_idx, action_valid_mask, td_errors
    ):
        """方策勾配: H(周囲+座標) の選択した行動を TD誤差で更新。"""
        fixed_action_size = 5
        for idx in list(actions_idx.keys()):
            if idx not in td_errors or idx not in states:
                continue
            state = states[idx]
            ranks_tuple, block_idx = state
            state_key = (ranks_tuple, (int(block_idx[0]), int(block_idx[1])))
            chosen_idx = actions_idx[idx]
            td_error = td_errors[idx]
            valid_mask = action_valid_mask.get(idx)
            if valid_mask is None or chosen_idx >= len(valid_mask):
                continue
            if not valid_mask[chosen_idx]:
                continue
            if state_key not in self.H:
                self.H[state_key] = [0.0] * fixed_action_size
            h_list = self.H[state_key]
            if len(h_list) < fixed_action_size:
                h_list.extend([0.0] * (fixed_action_size - len(h_list)))
            if chosen_idx < len(self.H[state_key]):
                self.H[state_key][chosen_idx] += self.alpha_h * td_error

    def update_dff(self):
        diffuse = self.params["diffuse"]
        decay = self.params["decay"]
        new_dff = (1 - decay) * (1 - diffuse) * self.dff
        padded = np.pad(new_dff, 1, mode="constant")
        for dx, dy in self.neighbors:
            new_dff += (
                decay * (1 - diffuse) / len(self.neighbors)
                * padded[
                    1 + dx : new_dff.shape[0] + 1 + dx,
                    1 + dy : new_dff.shape[1] + 1 + dy,
                ]
            )
        self.dff = new_dff
        self.dff[self.dff < 1e-4] = 0

    def reset(self, exit_pos=None, radius=None):
        # 実行スクリプトから渡された出口リスト(exit_pos)を、
        # exit_positions引数としてinitialize_agentsに渡す
        self.positions = self.initialize_agents(
            exit_positions=exit_pos, radius=radius
        )
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)

    def get_v_table(self):
        return dict(self.V)

    def get_v_table_size(self):
        current = len(self.V)
        return (self.initial_v_size, current, current - self.initial_v_size)

    def get_H_table(self):
        return dict(self.H)

    def get_H_table_size(self):
        current = len(self.H)
        return (self.initial_h_size, current, current - self.initial_h_size)

    def run(self, max_steps=None, return_trajectory=False):
        step = 0
        trajectory = [] if return_trajectory else None
        while self.positions.shape[0] > 0:
            if max_steps is not None and step >= max_steps:
                break
            self.step()
            if return_trajectory:
                trajectory.append(np.copy(self.positions))
            step += 1
        if return_trajectory:
            return step, trajectory
        return step
