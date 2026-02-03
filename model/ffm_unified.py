import pickle
import random
import sys
from collections import defaultdict

import numpy as np

if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray


class FloorFieldModelUnified:
    """
    統合Floor Field Model（Critic学習とActor学習を統合）

    学習モード:
    - "critic_only": Criticのみ学習（SFF + DFFを使用）
    - "actor_only": Actorのみ学習（事前学習済みCriticを使用、H + DFFを使用）
    - "both": CriticとActorの両方を学習

    Args:
        learning_mode: 学習モード（"critic_only", "actor_only", "both"）
        pretrained_v_path: 事前学習済みCriticのパス（actor_onlyまたはbothモードで使用可能）
    """

    def __init__(
        self,
        map_array,
        sff_path,
        N,
        learning_mode="critic_only",
        pretrained_v_path=None,
        params=None,
    ):
        default_params = {
            "k_S": 10,  # SFFの重み（critic_onlyモードで使用）
            "k_D": 1,  # DFFの重み
            "k_A": 10,  # Actorのロジットスケール（actor_only/bothモードで使用）
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "neumann",
            # Critic学習用パラメータ
            "alpha_v": 0.1,  # Criticの学習率
            "gamma": 0.95,  # 割引率
            "exit_reward": 100.0,  # 出口到達報酬
            "step_penalty": 0.0,  # ステップ罰
            "collision_penalty": -1.0,  # 衝突ペナルティ（1人あたり）
            "block_size": 5,  # 状態エンコーディングのブロックサイズ
            # Actor学習用パラメータ
            "alpha_h": 0.1,  # Actorの学習率
            "epsilon": 0.0,  # 探索率
        }
        self.params = (
            default_params if params is None else {**default_params, **params}
        )

        # 学習モードの検証
        valid_modes = ["critic_only", "actor_only", "both"]
        if learning_mode not in valid_modes:
            raise ValueError(
                f"learning_mode must be one of {valid_modes}, got {learning_mode}"
            )
        self.learning_mode = learning_mode

        self.map_array = map_array.astype(np.uint8)

        # SFFの読み込み（critic_onlyモードではmmap、それ以外では通常読み込み）
        if learning_mode == "critic_only":
            self.sff = np.load(sff_path, mmap_mode="r")
        else:
            sff_loaded = np.load(sff_path, mmap_mode="r")
            # SFFのinf値を0に置き換え
            self.sff = np.where(np.isinf(sff_loaded), 0.0, sff_loaded).astype(
                np.float32
            )

        self.dff = np.zeros_like(self.map_array, dtype=np.float32)
        self.N = N
        self.positions = self.initialize_agents()
        self.neighbors = self.get_neighbors()

        # Critic学習用の変数
        if pretrained_v_path and learning_mode in ["actor_only", "both"]:
            # 事前学習済みCriticを読み込み
            with open(pretrained_v_path, "rb") as f:
                pretrained_v_pickled = pickle.load(f)
            self.V = defaultdict(lambda: 0.0)
            for k, v in pretrained_v_pickled.items():
                real_key = pickle.loads(k)
                clean_key = tuple(
                    tuple(int(x) for x in sub_tuple) for sub_tuple in real_key
                )
                self.V[clean_key] = v
            self.initial_v_size = len(self.V)
            print(f"✓ 事前学習済みCriticを読み込みました: {self.initial_v_size}状態")
        else:
            self.V = defaultdict(lambda: 0.0)
            self.initial_v_size = 0
            if learning_mode == "actor_only":
                print("⚠ 警告: actor_onlyモードですが事前学習済みCriticが指定されていません")

        self.alpha_v = self.params["alpha_v"]
        self.gamma = self.params["gamma"]
        self.block_size = self.params["block_size"]

        # Actor学習用の変数（actor_only/bothモードで使用）
        if learning_mode in ["actor_only", "both"]:
            self.H = defaultdict(lambda: [])  # 状態ごとの行動価値リスト
            self.alpha_h = self.params["alpha_h"]
            self.epsilon = self.params.get("epsilon", 0.0)
        else:
            self.H = None
            self.alpha_h = None
            self.epsilon = None

    def initialize_agents(self, exit_pos=None, radius=None):
        """
        エージェントの初期配置を生成

        Args:
            exit_pos: 出口位置 (x, y)。Noneの場合は全セルから選択
            radius: 出口からの半径（L1距離）。Noneの場合は全セルから選択

        Returns:
            np.ndarray: エージェントの初期位置
        """
        if exit_pos is None or radius is None:
            # 従来通り全セルから選択
            free_cells = np.argwhere(self.map_array == 0)
            selected = free_cells[
                np.random.choice(len(free_cells), self.N, replace=False)
            ]
            return selected

        # 半径内の空きセルを取得
        exit_x, exit_y = exit_pos
        free_cells = np.argwhere(self.map_array == 0)
        radius_mask = (
            np.abs(free_cells[:, 0] - exit_x)
            + np.abs(free_cells[:, 1] - exit_y)
            <= radius
        )
        radius_cells = free_cells[radius_mask]

        # 半径内のセル数が人数より少ない場合は打ち止め
        available_count = len(radius_cells)
        actual_N = min(self.N, available_count)

        if actual_N == 0:
            # 半径内にセルがない場合は空配列を返す
            return np.empty((0, 2), dtype=np.int32)

        selected = radius_cells[
            np.random.choice(available_count, actual_N, replace=False)
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
        上下左右ランク状態エンコーディング
        - 上下左右の4方向に対して、各方向のランクを計算
        - ランク0: 移動不可（壁や人がいる）
        - ランク1: 斜め前に人がいる（左右問わない）
        - ランク2: 二歩先が埋まっている（壁や人）
        - ランク3: すべてに合致しない

        Args:
            x, y: エージェントの位置
            state_map: マップ値配列（0:歩行可能, 1:歩行者, 2:壁, 3:出口）

        Returns:
            bytes: 状態のハッシュ可能な表現
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
        block_idx = (x // self.block_size, y // self.block_size)

        # ハッシュ可能な形式に変換
        return pickle.dumps((tuple(ranks), block_idx))

    def step(self):
        move_requests = {}
        next_positions = np.copy(self.positions)

        # 学習用の変数
        states = {}  # エージェントの現在状態
        will_exit = {}  # 出口到達フラグ
        actions = {}  # エージェントが選択した行動（actor_only/bothモード用）
        action_probs = {}  # 行動選択確率（actor_only/bothモード用）
        action_choices = {}  # 各エージェントの利用可能な行動リスト（actor_only/bothモード用）
        action_valid_mask = {}  # 各エージェントの行動有効性マスク（actor_only/bothモード用）

        # マップ値配列を作成（0:歩行可能, 1:歩行者, 2:壁, 3:出口）
        state_map = self.map_array.copy()
        for pos in self.positions:
            state_map[pos[0], pos[1]] = 1  # 歩行者を1に設定

        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]

            # 現在の状態をエンコード
            state = self._encode_state(x, y, state_map)
            states[idx] = state

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
                will_exit[idx] = True
                if tuple(chosen_coord) not in move_requests:
                    move_requests[tuple(chosen_coord)] = []
                move_requests[tuple(chosen_coord)].append(idx)

                # actor_only/bothモードではアクション記録
                if self.learning_mode in ["actor_only", "both"]:
                    actions[idx] = tuple(chosen_coord)
                    action_probs[idx] = np.zeros(len(all_coords))
                    action_probs[idx][np.where(exit_mask)[0][0]] = 1.0
                    action_choices[idx] = [
                        tuple(coord) for coord in all_coords
                    ]
                    action_valid_mask[idx] = all_valid_mask
                continue

            # スコア計算（学習モードに応じて切り替え）
            if self.learning_mode == "critic_only":
                # Critic only: SFF + DFF
                sff_vals = np.array(
                    [self.sff[coord[0], coord[1]] for coord in all_coords]
                )
                dff_vals = np.array(
                    [self.dff[coord[0], coord[1]] for coord in all_coords]
                )
                score = (
                    -self.params["k_S"] * sff_vals
                    + self.params["k_D"] * dff_vals
                )

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
            else:
                # Actor only / Both: H + DFF
                # all_coordsとall_valid_maskは既に定義されている
                # DFF値を取得（全方向に対して）
                dff_vals = np.array(
                    [self.dff[coord[0], coord[1]] for coord in all_coords]
                )

                # Actorのロジット計算: kA * H(s,a) + kD * DFF
                state_key = state
                fixed_action_size = len(all_coords)
                if (
                    state_key not in self.H
                    or len(self.H[state_key]) != fixed_action_size
                ):
                    # 初回またはサイズが変わった場合、0で初期化
                    self.H[state_key] = [0.0] * fixed_action_size
                h_vals = np.array(self.H[state_key])

                # H値をSFFスケールに規格化（Hテーブル自体は変更しない）
                if hasattr(self, "sff") and len(self.H) > 0:
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

                # ε-greedy（actor_only/bothモードのみ）
                if (
                    self.learning_mode in ["actor_only", "both"]
                    and self.epsilon > 0
                    and random.random() < self.epsilon
                ):
                    # 有効な行動のみからランダム選択
                    valid_indices = np.where(all_valid_mask)[0]
                    if len(valid_indices) > 0:
                        chosen_idx = valid_indices[
                            np.random.randint(len(valid_indices))
                        ]
                        probs_record = np.zeros(len(all_coords))
                        probs_record[chosen_idx] = 1.0
                    else:
                        # フォールバック: 現在地を選択
                        chosen_idx = len(all_coords) - 1
                        probs_record = np.zeros(len(all_coords))
                        probs_record[chosen_idx] = 1.0
                else:
                    # 確率分布に従って選択
                    chosen_idx = np.random.choice(len(all_coords), p=probs)
                    probs_record = probs

                # 選択された座標を取得
                chosen = tuple(all_coords[chosen_idx])

                if chosen not in move_requests:
                    move_requests[chosen] = []
                move_requests[chosen].append(idx)

                # Actor学習用の記録（actor_only/bothモード）
                if self.learning_mode in ["actor_only", "both"]:
                    actions[idx] = chosen
                    action_probs[idx] = probs_record
                    action_choices[idx] = [
                        tuple(coord) for coord in all_coords
                    ]
                    action_valid_mask[idx] = all_valid_mask

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

        # TD更新を実行（学習モードに応じて）
        td_errors = None
        if self.learning_mode == "critic_only":
            # Criticのみ更新
            self._update_critic(
                states,
                will_exit,
                collision_counts,
                next_positions,
                state_map_next,
            )
        elif self.learning_mode == "actor_only":
            # Criticも更新（新規状態のみ）、TD誤差を取得
            self._update_critic(
                states,
                will_exit,
                collision_counts,
                next_positions,
                state_map_next,
            )
            td_errors = self._get_td_errors(
                states,
                will_exit,
                collision_counts,
                next_positions,
                state_map_next,
            )
        elif self.learning_mode == "both":
            # Critic更新とTD誤差を同時に取得
            td_errors = self._update_critic(
                states,
                will_exit,
                collision_counts,
                next_positions,
                state_map_next,
                return_td_errors=True,
            )

        # Actor更新（actor_only/bothモード）
        if (
            self.learning_mode in ["actor_only", "both"]
            and td_errors is not None
        ):
            self._update_actor(
                states,
                actions,
                action_choices,
                action_probs,
                action_valid_mask,
                td_errors,
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
        return_td_errors=False,
    ):
        """
        TD学習によるCriticの更新

        Args:
            states: エージェントの現在状態
            will_exit: 出口到達フラグ
            collision_counts: 衝突人数
            next_positions: 次の位置
            state_map_next: 次のマップ値配列
            return_td_errors: TD誤差を返すかどうか

        Returns:
            dict: エージェントごとのTD誤差（return_td_errors=Trueの場合）
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
                state_next = self._encode_state(x_next, y_next, state_map_next)
                v_next = self.V[state_next]

            # TD誤差の計算
            v_current = self.V[state]
            td_error = reward + self.gamma * v_next - v_current

            # 価値関数の更新
            self.V[state] = v_current + self.alpha_v * td_error

            if return_td_errors:
                td_errors[idx] = td_error

        return td_errors if return_td_errors else None

    def _get_td_errors(
        self,
        states,
        will_exit,
        collision_counts,
        next_positions,
        state_map_next,
    ):
        """
        TD誤差を計算（Critic更新なし）

        Args:
            states: エージェントの現在状態
            will_exit: 出口到達フラグ
            collision_counts: 衝突人数
            next_positions: 次の位置
            state_map_next: 次のマップ値配列

        Returns:
            dict: エージェントごとのTD誤差
        """
        td_errors = {}

        for idx in states:
            state = states[idx]

            # 報酬計算
            reward = self.params["step_penalty"]

            if idx in will_exit and will_exit[idx]:
                reward += self.params["exit_reward"]

            if idx in collision_counts:
                collision_penalty = (
                    collision_counts[idx] * self.params["collision_penalty"]
                )
                reward += collision_penalty

            # 次の状態の価値を計算
            if idx in will_exit and will_exit[idx]:
                v_next = 0.0
            else:
                x_next, y_next = next_positions[idx]
                state_next = self._encode_state(x_next, y_next, state_map_next)
                v_next = self.V[state_next]

            # TD誤差の計算
            v_current = self.V[state]
            td_error = reward + self.gamma * v_next - v_current
            td_errors[idx] = td_error

        return td_errors

    def _update_actor(
        self,
        states,
        actions,
        action_choices,
        action_probs,
        action_valid_mask,
        td_errors,
    ):
        """
        方策勾配法によるActorの更新

        Args:
            states: エージェントの現在状態
            actions: エージェントが選択した行動
            action_choices: 各エージェントの利用可能な行動リスト
            action_probs: 行動選択確率
            action_valid_mask: 各エージェントの行動有効性マスク
            td_errors: TD誤差
        """
        for idx in states:
            if (
                idx not in actions
                or idx not in td_errors
                or idx not in action_choices
                or idx not in action_valid_mask
            ):
                continue

            state = states[idx]
            action = actions[idx]
            td_error = td_errors[idx]
            choices = action_choices[idx]
            valid_mask = action_valid_mask[idx]

            # 選択された行動のインデックスを見つける
            try:
                chosen_idx = choices.index(action)
            except ValueError:
                continue

            # 状態をkeyとして行動価値リストを更新
            state_key = state
            fixed_action_size = len(choices)
            if (
                state_key not in self.H
                or len(self.H[state_key]) != fixed_action_size
            ):
                self.H[state_key] = [0.0] * fixed_action_size

            # 選択された行動のみ更新
            if chosen_idx < len(self.H[state_key]) and valid_mask[chosen_idx]:
                self.H[state_key][chosen_idx] += self.alpha_h * td_error

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

    def reset(self, exit_pos=None, radius=None):
        """
        エピソードをリセット（学習用）
        位置とDFFをリセットするが、V/Hテーブルは保持

        Args:
            exit_pos: 出口位置 (x, y)。Noneの場合は全セルから選択
            radius: 出口からの半径（L1距離）。Noneの場合は全セルから選択
        """
        self.positions = self.initialize_agents(
            exit_pos=exit_pos, radius=radius
        )
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
        self.V = defaultdict(lambda: 0.0, v_table)

    def get_v_table_size(self):
        """
        Vテーブルのサイズを取得

        Returns:
            int: 状態数（critic_only/bothモード）
            tuple: (初期サイズ, 現在サイズ, 増加数)（actor_onlyモード）
        """
        if self.learning_mode == "actor_only":
            current_size = len(self.V)
            new_states = current_size - self.initial_v_size
            return (self.initial_v_size, current_size, new_states)
        else:
            return len(self.V)

    def get_h_table(self):
        """
        現在のHテーブルを取得（actor_only/bothモードのみ）

        Returns:
            dict: 状態ごとの行動価値リスト
        """
        if self.learning_mode in ["actor_only", "both"]:
            return dict(self.H)
        else:
            return None

    def set_epsilon(self, epsilon):
        """
        ε-greedyのεを設定（actor_only/bothモードのみ）

        Args:
            epsilon (float): 0.0〜1.0の探索率
        """
        if self.learning_mode in ["actor_only", "both"]:
            self.epsilon = float(np.clip(epsilon, 0.0, 1.0))

    def get_h_table_size(self):
        """
        Hテーブルのサイズを取得（actor_only/bothモードのみ）

        Returns:
            tuple: (状態数, 総行動数)
        """
        if self.learning_mode in ["actor_only", "both"]:
            total_actions = sum(len(actions) for actions in self.H.values())
            return (len(self.H), total_actions)
        else:
            return None

    def run(
        self,
        save_prefix=None,
        save_interval=100,
        max_steps=None,
        return_trajectory=False,
    ):
        """
        シミュレーションを実行

        Args:
            save_prefix: 保存用プレフィックス
            save_interval: 保存間隔
            max_steps: 最大ステップ数（Noneの場合は無制限）
            return_trajectory: 軌跡を返すかどうか

        Returns:
            int: 実行ステップ数（return_trajectory=Falseの場合）
            tuple: (実行ステップ数, 軌跡)（return_trajectory=Trueの場合）
        """
        buffer = []
        trajectory = [] if return_trajectory else None
        step = 0

        while self.positions.shape[0] > 0:
            if max_steps is not None and step >= max_steps:
                break

            self.step()
            positions_copy = np.copy(self.positions)
            buffer.append(positions_copy)
            if return_trajectory:
                trajectory.append(positions_copy)
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

        if return_trajectory:
            return step, np.array(trajectory, dtype=object)
        return step
