import pickle
import random
from collections import defaultdict

import numpy as np


class FloorFieldModelActorOnly:
    """
    Actor学習を実装したFloor Field Model（Criticは事前学習済みを使用）
    - Critic: 事前学習済みのV(s)を読み込み（read-only + 新規状態のみ学習）
    - Actor: 状態-行動ロジットH(s,a)を0から学習
    - ロジット = kA * H(s,a) + kD * DFF（SFFは使わない）
    """

    def __init__(
        self, map_array, sff_path, N, pretrained_v_path=None, params=None
    ):
        default_params = {
            "k_D": 1,
            "k_A": 10,  # Actorのロジットスケール
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "neumann",
            # Critic学習用パラメータ（新規状態のみ）
            "alpha_v": 0.1,  # Criticの学習率
            "gamma": 0.95,  # 割引率
            "exit_reward": 100.0,  # 出口到達報酬
            "step_penalty": 0.0,  # ステップ罰
            "collision_penalty": -1.0,  # 衝突ペナルティ（1人あたり）
            # Actor学習用パラメータ
            "alpha_h": 0.1,  # Actorの学習率
            "epsilon": 0.0,  # 探索率
        }
        self.params = (
            default_params if params is None else {**default_params, **params}
        )
        self.map_array = map_array.astype(np.uint8)
        sff_loaded = np.load(sff_path, mmap_mode="r")
        # SFFのinf値を0に置き換え
        self.sff = np.where(np.isinf(sff_loaded), 0.0, sff_loaded).astype(
            np.float32
        )
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)
        self.N = N
        self.positions = self.initialize_agents()
        self.neighbors = self.get_neighbors()

        # Critic: 事前学習済みを読み込み、全状態を更新可能
        if pretrained_v_path:
            with open(pretrained_v_path, "rb") as f:
                pretrained_v = pickle.load(f)
            self.V = defaultdict(lambda: 0.0, pretrained_v)
            self.initial_v_size = len(pretrained_v)
            print(f"✓ 事前学習済みCriticを読み込みました: {self.initial_v_size}状態")
        else:
            self.V = defaultdict(lambda: 0.0)
            self.initial_v_size = 0
            print("⚠ 事前学習済みCriticなしで開始します")

        self.alpha_v = self.params["alpha_v"]
        self.gamma = self.params["gamma"]

        # Actor学習用の変数
        self.H = defaultdict(lambda: [])  # 状態ごとの行動価値リスト
        self.alpha_h = self.params["alpha_h"]
        self.epsilon = self.params.get("epsilon", 0.0)

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

    def _encode_state(self, x, y, occupancy):
        """
        13セル状態エンコーディング
        - 中心を含む3x3（9セル）
        - 上下左右に1セル多く伸ばした4セル（合計13セル）

        Args:
            x, y: エージェントの位置
            occupancy: 占有マップ（0=空き, 1=歩行者, 2=壁, 3=出口）

        Returns:
            bytes: 状態のハッシュ可能な表現
        """
        # 1. 中心を含む3x3（9セル）を取得
        local_3x3 = occupancy[max(0, x - 1) : x + 2, max(0, y - 1) : y + 2]

        # パディングして3x3に統一
        padded_3x3 = np.zeros((3, 3), dtype=np.uint8)
        start_x = max(0, 1 - x) if x < 1 else 0
        start_y = max(0, 1 - y) if y < 1 else 0
        end_x = start_x + local_3x3.shape[0]
        end_y = start_y + local_3x3.shape[1]
        padded_3x3[start_x:end_x, start_y:end_y] = local_3x3

        # 2. 上下左右に1セル多く伸ばした4セル
        ahead_cells = []
        ahead_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]  # U2, D2, L2, R2
        for dx, dy in ahead_offsets:
            ahead_x, ahead_y = x + dx, y + dy
            if (
                0 <= ahead_x < occupancy.shape[0]
                and 0 <= ahead_y < occupancy.shape[1]
            ):
                ahead_cells.append(int(occupancy[ahead_x, ahead_y]))
            else:
                ahead_cells.append(0)  # 境界外は0（空きとして扱う）

        # 3. 13セルの状態を結合（9セル + 4セル）
        state_13 = np.concatenate([padded_3x3.flatten(), ahead_cells])

        # 4. 粗い位置情報（ブロックインデックス）
        block_size = 5
        block_idx = (x // block_size, y // block_size)

        # 5. ハッシュ可能な形式に変換
        return pickle.dumps((tuple(state_13), block_idx))

    def step(self):
        move_requests = {}
        next_positions = np.copy(self.positions)

        # 学習用の変数
        states = {}  # エージェントの現在状態
        will_exit = {}  # 出口到達フラグ
        actions = {}  # エージェントが選択した行動
        action_probs = {}  # 行動選択確率
        action_choices = {}  # 各エージェントの利用可能な行動リスト

        # 現在の占有状況を作成（マップの値: 0=空き, 1=歩行者, 2=壁, 3=出口）
        occupancy = np.copy(self.map_array)
        for pos in self.positions:
            occupancy[pos[0], pos[1]] = 1  # 歩行者がいるセルは1
        # INSERT_YOUR_CODE
        # Hテーブルは変更せず、計算時にのみSFFに規格化する

        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]
            current_pos = np.array([x, y])

            # 現在の状態をエンコード
            state = self._encode_state(x, y, occupancy)
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
            if neighbor_coords.shape[0] != 4:
                pass
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
                    # 出口強制時もアクション記録
                    actions[idx] = tuple(chosen_coord)
                    # INSERT_YOUR_CODE
                    # 出口強制時も action_probs, action_choice を記録
                    # 強制選択時もchoiceとして保存しないと後でtd_error計算で困る
                    action_probs[idx] = np.zeros(len(neighbor_coords))
                    action_probs[idx][0] = 1.0  # 最後の選択肢が現在地(=出口)で必ず選ばれる
                    action_choices[idx] = [
                        tuple(coord) for coord in neighbor_coords
                    ]
                    continue

                dff_vals = self.dff[
                    neighbor_coords[:, 0], neighbor_coords[:, 1]
                ]

                # Actorのロジット計算: kA * H(s,a) + kD * DFF
                # 状態をkeyとして行動価値リストを取得
                state_key = state
                if state_key not in self.H or len(self.H[state_key]) != len(
                    neighbor_coords
                ):
                    # 初回または行動数が変わった場合、0で初期化
                    self.H[state_key] = [0.0] * len(neighbor_coords)
                h_vals = np.array(self.H[state_key])

                # デバッグ: 配列サイズの確認
                if len(h_vals) != len(dff_vals):
                    print(
                        f"警告: h_vals長さ={len(h_vals)}, dff_vals長さ={len(dff_vals)}"
                    )
                    print(f"neighbor_coords長さ={len(neighbor_coords)}")
                    # サイズを合わせる
                    if len(h_vals) < len(dff_vals):
                        h_vals = np.pad(
                            h_vals,
                            (0, len(dff_vals) - len(h_vals)),
                            "constant",
                        )
                    else:
                        h_vals = h_vals[: len(dff_vals)]

                # H値をSFFスケールに規格化（Hテーブル自体は変更しない）
                if hasattr(self, "sff") and len(self.H) > 0:
                    # Hテーブル全体の最大値と最小値を取得
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

                        # NaN/Infチェック
                        if not (
                            np.isnan(h_table_min)
                            or np.isnan(h_table_max)
                            or np.isinf(h_table_min)
                            or np.isinf(h_table_max)
                        ):
                            # 反転マッピング: h_max -> sff_min, h_min -> sff_max
                            if h_table_max - h_table_min > 1e-6:
                                h_vals_normalized = (
                                    (h_table_max - h_vals)
                                    / (h_table_max - h_table_min)
                                ) * (sff_max - sff_min) + sff_min
                                h_vals = h_vals_normalized

                score = (
                    -self.params["k_A"] * h_vals
                    + self.params["k_D"] * dff_vals
                )

                # NaN/Infチェック
                if np.any(np.isnan(score)) or np.any(np.isinf(score)):
                    # 不正な値が含まれている場合はランダム選択
                    score = np.zeros_like(score)

                score_max = np.max(score)

                probs = np.exp(score - score_max)
                probs_sum = probs.sum()
                if np.isfinite(probs_sum) and probs_sum != 0:
                    probs /= probs_sum
                    # ε-greedy: εの確率で完全にランダム移動
                    if self.epsilon > 0 and random.random() < self.epsilon:
                        chosen_idx = np.random.randint(len(neighbor_coords))
                        probs_record = np.zeros(len(neighbor_coords))
                        probs_record[chosen_idx] = 1.0
                    else:
                        chosen_idx = np.random.choice(
                            len(neighbor_coords), p=probs
                        )
                        probs_record = probs
                    chosen = tuple(neighbor_coords[chosen_idx])
                    if chosen not in move_requests:
                        move_requests[chosen] = []
                    move_requests[chosen].append(idx)

                    # Actor学習用の記録
                    actions[idx] = chosen
                    action_probs[idx] = probs_record
                    action_choices[idx] = [
                        tuple(coord) for coord in neighbor_coords
                    ]

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
        # 次の占有状況を作成（マップの値: 0=空き, 1=歩行者, 2=壁, 3=出口）
        occupancy_next = np.copy(self.map_array)
        for pos in next_positions:
            if self.map_array[pos[0], pos[1]] != 3:  # 出口以外
                occupancy_next[pos[0], pos[1]] = 1  # 歩行者がいるセルは1

        # TD更新を実行（Critic: 新規状態のみ）
        td_errors = self._update_critic(
            states, will_exit, collision_counts, next_positions, occupancy_next
        )

        # Actor更新
        self._update_actor(
            states, actions, action_choices, action_probs, td_errors
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
        occupancy_next,
    ):
        """
        TD学習によるCriticの更新（新規状態のみ）

        Args:
            states: エージェントの現在状態
            will_exit: 出口到達フラグ
            collision_counts: 衝突人数
            next_positions: 次の位置
            occupancy_next: 次の占有状況

        Returns:
            dict: エージェントごとのTD誤差
        """
        td_errors = {}

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
                state_next = self._encode_state(x_next, y_next, occupancy_next)
                v_next = self.V[state_next]

            # TD誤差の計算
            v_current = self.V[state]
            td_error = reward + self.gamma * v_next - v_current

            # 価値関数の更新（全ての状態を更新）
            self.V[state] = v_current + self.alpha_v * td_error

            # TD誤差を記録
            td_errors[idx] = td_error

        return td_errors

    def _update_actor(
        self, states, actions, action_choices, action_probs, td_errors
    ):
        """
        方策勾配法によるActorの更新

        Args:
            states: エージェントの現在状態
            actions: エージェントが選択した行動
            action_choices: 各エージェントの利用可能な行動リスト
            action_probs: 行動選択確率
            td_errors: TD誤差
        """
        for idx in states:
            if (
                idx not in actions
                or idx not in td_errors
                or idx not in action_choices
            ):
                continue

            state = states[idx]
            action = actions[idx]
            td_error = td_errors[idx]
            choices = action_choices[idx]
            probs = action_probs[idx]

            # 選択された行動のインデックスを見つける
            try:
                chosen_idx = choices.index(action)
            except ValueError:
                continue

            # 状態をkeyとして行動価値リストを更新
            state_key = state
            if state_key not in self.H or len(self.H[state_key]) != len(
                choices
            ):
                self.H[state_key] = [0.0] * len(choices)
            # 全ての行動についてH値を更新
            for i, choice in enumerate(choices):
                if i == chosen_idx:
                    # 選択された行動: H(s,a) += α_h * δ * (1 - π(a|s))
                    self.H[state_key][i] += self.alpha_h * td_error
                # else:
                #     # 選択されなかった行動: H(s,a') -= α_h * δ * π(a'|s)
                #     self.H[state_key][i] -= self.alpha_h * td_error * probs[i]

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
        位置とDFFをリセットするが、V/Hテーブルは保持
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

    def get_v_table_size(self):
        """
        Vテーブルのサイズを取得

        Returns:
            tuple: (初期サイズ, 現在サイズ, 増加数)
        """
        current_size = len(self.V)
        new_states = current_size - self.initial_v_size
        return (self.initial_v_size, current_size, new_states)

    def get_h_table(self):
        """
        現在のHテーブルを取得

        Returns:
            dict: 状態ごとの行動価値リスト
        """
        return dict(self.H)

    def set_epsilon(self, epsilon):
        """
        ε-greedyのεを設定

        Args:
            epsilon (float): 0.0〜1.0の探索率
        """
        self.epsilon = float(np.clip(epsilon, 0.0, 1.0))

    def get_h_table_size(self):
        """
        Hテーブルのサイズを取得

        Returns:
            tuple: (状態数, 総行動数)
        """
        total_actions = sum(len(actions) for actions in self.H.values())
        return (len(self.H), total_actions)

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
