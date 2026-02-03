import pickle
import random
from collections import defaultdict

import numpy as np


class FloorFieldModel:
    def __init__(self, map_array, sff_path, N, params=None):
        default_params = {
            "k_S": 10,
            "k_D": 1,
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "neumann",
            # Critic学習用パラメータ
            "alpha_v": 0.1,  # Criticの学習率
            "gamma": 0.95,  # 割引率
            "exit_reward": 100.0,  # 出口到達報酬
            "step_penalty": 0.0,  # ステップ罰
            "collision_penalty": -1.0,  # 衝突ペナルティ（1人あたり）
            "block_size": 3,  # 状態エンコーディングのブロックサイズ
        }
        self.params = (
            default_params if params is None else {**default_params, **params}
        )
        self.map_array = map_array.astype(np.uint8)
        self.sff = np.load(sff_path, mmap_mode="r")
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)
        self.N = N
        self.positions = self.initialize_agents()
        self.neighbors = self.get_neighbors()

        # Critic学習用の変数
        self.V = defaultdict(lambda: 0.0)  # 状態価値テーブル（初期値0）
        self.alpha_v = self.params["alpha_v"]
        self.gamma = self.params["gamma"]
        self.block_size = self.params["block_size"]

    def initialize_agents(self):
        free_cells = np.argwhere(self.map_array == 0)
        selected = free_cells[
            np.random.choice(len(free_cells), self.N, replace=False)
        ]
        return selected

    def get_neighbors(self):
        if self.params["neighborhood"] == "neumann":
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            return [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]

    def _encode_state(self, x, y, state_map):
        """
        13セル状態エンコーディング
        - 中心を含む3x3（9セル）
        - 上下左右に1セル多く伸ばした4セル（合計13セル）

        Args:
            x, y: エージェントの位置
            state_map: マップ値配列（0:歩行可能, 1:歩行者, 2:壁, 3:出口）

        Returns:
            bytes: 状態のハッシュ可能な表現
        """
        # 1. 中心を含む3x3（9セル）を取得
        local_3x3 = state_map[max(0, x - 1) : x + 2, max(0, y - 1) : y + 2]

        # パディングして3x3に統一（境界外は2=壁として扱う）
        padded_3x3 = np.full((3, 3), 2, dtype=np.uint8)  # デフォルトは壁
        start_x = max(0, 1 - x) if x < 1 else 0
        start_y = max(0, 1 - y) if y < 1 else 0
        end_x = start_x + local_3x3.shape[0]
        end_y = start_y + local_3x3.shape[1]
        padded_3x3[start_x:end_x, start_y:end_y] = local_3x3

        # 2. 上下左右に1セル多く伸ばした4セル
        # 上方向（x-2, y）、下方向（x+2, y）、左方向（x, y-2）、右方向（x, y+2）
        ahead_cells = []
        ahead_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # U2, D2, L2, R2
        for dx, dy in ahead_offsets:
            ahead_x, ahead_y = x + dx, y + dy
            if (
                0 <= ahead_x < state_map.shape[0]
                and 0 <= ahead_y < state_map.shape[1]
            ):
                ahead_cells.append(int(state_map[ahead_x, ahead_y]))
            else:
                ahead_cells.append(2)  # 境界外は壁（2）

        # 3. 13セルの状態を結合（9セル + 4セル）
        state_13 = np.concatenate(
            [padded_3x3.flatten().astype(int), ahead_cells]
        )

        # 4. 粗い位置情報（ブロックインデックス）
        block_idx = (x // self.block_size, y // self.block_size)

        # 5. ハッシュ可能な形式に変換
        return pickle.dumps((tuple(state_13), block_idx))

    def step(self):
        move_requests = {}
        next_positions = np.copy(self.positions)

        # Critic学習用の変数
        states = {}  # エージェントの現在状態
        will_exit = {}  # 出口到達フラグ

        # マップ値配列を作成（0:歩行可能, 1:歩行者, 2:壁, 3:出口）
        state_map = self.map_array.copy()
        for pos in self.positions:
            state_map[pos[0], pos[1]] = 1  # 歩行者を1に設定

        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]
            current_pos = np.array([x, y])

            # 現在の状態をエンコード
            state = self._encode_state(x, y, state_map)
            states[idx] = state

            offsets = np.array(self.neighbors)
            neighbor_coords = self.positions[idx] + offsets

            # occupied set の numpy版（自分を除外）
            other_positions = np.delete(self.positions, idx, axis=0)
            occupied = other_positions.view(
                [("", other_positions.dtype)] * 2
            ).reshape(-1)

            # 移動可能セル（壁・出口）
            mask = (
                self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]]
                == 0
            ) | (
                self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]]
                == 3
            )
            neighbor_coords = neighbor_coords[mask]

            # occupied 判定
            if neighbor_coords.shape[0] > 0:
                neighbor_struct = neighbor_coords.view(
                    [("", neighbor_coords.dtype)] * 2
                ).reshape(-1)
                mask = ~np.isin(neighbor_struct, occupied)
                neighbor_coords = neighbor_coords[mask]

            # ★現在地を追加して「動かない」という選択肢を入れる
            if neighbor_coords.shape[0] > 0:
                neighbor_coords = np.vstack([neighbor_coords, current_pos])

                exit_mask = (
                    self.map_array[
                        neighbor_coords[:, 0], neighbor_coords[:, 1]
                    ]
                    == 3
                )
                if np.any(exit_mask):
                    chosen_coord = neighbor_coords[exit_mask][0]
                    will_exit[idx] = True  # 出口到達をマーク
                    if tuple(chosen_coord) not in move_requests:
                        move_requests[tuple(chosen_coord)] = []
                    move_requests[tuple(chosen_coord)].append(idx)
                    continue

                sff_vals = self.sff[
                    neighbor_coords[:, 0], neighbor_coords[:, 1]
                ]
                dff_vals = self.dff[
                    neighbor_coords[:, 0], neighbor_coords[:, 1]
                ]

                score = (
                    -self.params["k_S"] * sff_vals
                    + self.params["k_D"] * dff_vals
                )
                score_max = np.max(score)

                probs = np.exp(score - score_max)
                probs_sum = probs.sum()
                if np.isfinite(probs_sum) and probs_sum != 0:
                    probs /= probs_sum
                    chosen_idx = np.random.choice(
                        len(neighbor_coords), p=probs
                    )
                    chosen = tuple(neighbor_coords[chosen_idx])
                    if chosen not in move_requests:
                        move_requests[chosen] = []
                    move_requests[chosen].append(idx)

        # 衝突カウントを記録
        collision_counts = {}  # エージェントごとの衝突人数

        for target, agents in move_requests.items():
            if len(agents) == 1:
                next_positions[agents[0]] = target
                self.dff[
                    self.positions[agents[0]][0], self.positions[agents[0]][1]
                ] += 1
                collision_counts[agents[0]] = 0  # 衝突なし
            else:
                # 衝突発生: 必ずだれか一人がそのセルに入れる
                collision_count = len(agents) - 1  # 自分以外の衝突人数
                chosen = random.choice(agents)
                next_positions[chosen] = target
                self.dff[
                    self.positions[chosen][0], self.positions[chosen][1]
                ] += 1
                collision_counts[chosen] = collision_count
                # 負けたエージェントにも衝突を記録
                for agent in agents:
                    if agent != chosen:
                        collision_counts[agent] = collision_count

        # 出口に到達した歩行者を除外する前にTD更新を実行
        # 次のマップ値配列を作成（0:歩行可能, 1:歩行者, 2:壁, 3:出口）
        state_map_next = self.map_array.copy()
        for pos in next_positions:
            if self.map_array[pos[0], pos[1]] != 3:  # 出口以外
                state_map_next[pos[0], pos[1]] = 1  # 歩行者を1に設定

        # TD更新を実行
        self._update_critic(
            states, will_exit, collision_counts, next_positions, state_map_next
        )

        # 出口に到達した歩行者を除外
        keep_mask = (
            self.map_array[next_positions[:, 0], next_positions[:, 1]] != 3
        )
        self.positions = next_positions[keep_mask]

        self.update_dff()

    def _update_critic(
        self,
        states,
        will_exit,
        collision_counts,
        next_positions,
        state_map_next,
    ):
        """
        TD学習によるCriticの更新

        Args:
            states: エージェントの現在状態
            will_exit: 出口到達フラグ
            collision_counts: 衝突人数
            next_positions: 次の位置
            state_map_next: 次のマップ値配列（0:歩行可能, 1:歩行者, 2:壁, 3:出口）
        """
        for idx in states:
            state = states[idx]

            # 報酬計算
            reward = self.params["step_penalty"]  # ステップ罰

            # 出口到達報酬
            if idx in will_exit and will_exit[idx]:
                reward += self.params["exit_reward"]

            # 衝突ペナルティ
            if idx in collision_counts:
                collision_penalty = (
                    collision_counts[idx] * self.params["collision_penalty"]
                )
                reward += collision_penalty

            # 次の状態の価値を計算
            if idx in will_exit and will_exit[idx]:
                # 終端状態（出口到達）
                v_next = 0.0
            else:
                # 次の状態をエンコード
                x_next, y_next = next_positions[idx]
                state_next = self._encode_state(x_next, y_next, state_map_next)
                v_next = self.V[state_next]

            # TD誤差の計算
            v_current = self.V[state]
            td_error = reward + self.gamma * v_next - v_current

            # 価値関数の更新
            self.V[state] = v_current + self.alpha_v * td_error

    def update_dff(self):
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

    def reset(self):
        """
        エピソードをリセット（学習用）
        位置とDFFをリセットするが、Vテーブルは保持
        """
        self.positions = self.initialize_agents()
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)

    def get_v_table(self):
        """
        現在のVテーブルを取得

        Returns:
            dict: 状態価値テーブル
        """
        return dict(self.V)

    def set_v_table(self, v_table):
        """
        Vテーブルを設定

        Args:
            v_table: 状態価値テーブル
        """
        self.V = defaultdict(lambda: -1.0, v_table)

    def get_v_table_size(self):
        """
        Vテーブルのサイズを取得

        Returns:
            int: 状態数
        """
        return len(self.V)

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
