import random

import numpy as np


class FloorFieldModel:
    def __init__(self, map_array, sff_path, N, params=None):
        default_params = {
            "k_S": 3,
            "k_D": 1,
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "moore"
        }
        self.params = default_params if params is None else {**default_params, **params}
        self.map_array = map_array.astype(np.uint8)
        self.sff = np.load(sff_path, mmap_mode='r')
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)
        self.N = N
        self.positions = self.initialize_agents()
        self.neighbors = self.get_neighbors()

        # 学習用
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.9

    def initialize_agents(self):
        free_cells = np.argwhere(self.map_array == 0)
        selected = free_cells[np.random.choice(len(free_cells), self.N, replace=False)]
        return selected

    def get_neighbors(self):
        if self.params["neighborhood"] == "neumann":
            return [(-1,0), (1,0), (0,-1), (0,1)]
        else:
            return [(-1,-1), (-1,0), (-1,1),
                    (0,-1),          (0,1),
                    (1,-1),  (1,0),  (1,1)]

    def reset(self):
        self.positions = self.initialize_agents()
        self.dff = np.zeros_like(self.dff)

    def step(self, beta):
        move_requests = {}
        next_positions = np.copy(self.positions)

        occupied = self.positions.view([('', self.positions.dtype)] * 2).reshape(-1)
        decisions = {}

        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]
            current_pos = np.array([x, y])

            offsets = np.array(self.neighbors)
            neighbor_coords = self.positions[idx] + offsets

            mask = (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 0) | \
                   (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3)
            neighbor_coords = neighbor_coords[mask]

            neighbor_struct = neighbor_coords.view([('', neighbor_coords.dtype)] * 2).reshape(-1)
            mask = ~np.isin(neighbor_struct, occupied)
            neighbor_coords = neighbor_coords[mask]

            # 動かない選択肢を追加
            if neighbor_coords.shape[0] > 0:
                neighbor_coords = np.vstack([neighbor_coords, current_pos])

                exit_mask = self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3
                if np.any(exit_mask):
                    chosen_coord = neighbor_coords[exit_mask][0]

                    # 出口の場合も適当なQ更新用のダミーを登録
                    state = tuple(map(tuple, (neighbor_coords - self.positions[idx])))
                    if state not in self.Q:
                        self.Q[state] = np.zeros(len(neighbor_coords))

                    chosen_idx = np.where(exit_mask)[0][0]  # 最初の出口のインデックス
                    decisions[idx] = (chosen_idx, state, neighbor_coords)

                    if tuple(chosen_coord) not in move_requests:
                        move_requests[tuple(chosen_coord)] = []
                    move_requests[tuple(chosen_coord)].append(idx)
                    continue

                # 状態をtupleに変換
                state = tuple(map(tuple, (neighbor_coords - self.positions[idx])))
                if state not in self.Q:
                    self.Q[state] = np.zeros(len(neighbor_coords))

                sff_vals = self.sff[neighbor_coords[:, 0], neighbor_coords[:, 1]]
                dff_vals = self.dff[neighbor_coords[:, 0], neighbor_coords[:, 1]]
                q_vals = self.Q[state]

                val = beta * (-self.params["k_S"] * sff_vals + self.params["k_D"] * dff_vals) + (1 - beta) * q_vals
                score_max = np.max(val)
                probs = np.exp(val - score_max)
                probs_sum = probs.sum()
                if probs_sum == 0 or not np.isfinite(probs_sum):
                    continue
                probs /= probs_sum

                chosen_idx = np.random.choice(len(neighbor_coords), p=probs)
                chosen = tuple(neighbor_coords[chosen_idx])

                if chosen not in move_requests:
                    move_requests[chosen] = []
                move_requests[chosen].append(idx)

                decisions[idx] = (chosen_idx, state, neighbor_coords)

        # 衝突解決 + Q更新
        for target, agents in move_requests.items():
            if len(agents) == 1:
                idx = agents[0]
                chosen_idx, state, neighbor_coords = decisions[idx]

                prev_sff = self.sff[self.positions[idx][0], self.positions[idx][1]]
                next_positions[idx] = target
                next_sff = self.sff[target[0], target[1]]

                reward = beta * (prev_sff - next_sff)
                if self.map_array[target[0], target[1]] == 3:
                    reward += 10

                next_state = state
                self.Q[state][chosen_idx] += self.alpha * (
                    reward + self.gamma * np.max(self.Q.get(next_state, np.zeros_like(self.Q[state])))
                    - self.Q[state][chosen_idx]
                )
                self.dff[self.positions[idx][0], self.positions[idx][1]] += 1

            else:
                if np.random.rand() < 0.5:
                    chosen = random.choice(agents)
                    idx = chosen
                    chosen_idx, state, neighbor_coords = decisions[idx]

                    prev_sff = self.sff[self.positions[idx][0], self.positions[idx][1]]
                    next_positions[idx] = target
                    next_sff = self.sff[target[0], target[1]]

                    reward = beta * (prev_sff - next_sff)
                    if self.map_array[target[0], target[1]] == 3:
                        reward += 10

                    next_state = state
                    self.Q[state][chosen_idx] += self.alpha * (
                        reward + self.gamma * np.max(self.Q.get(next_state, np.zeros_like(self.Q[state])))
                        - self.Q[state][chosen_idx]
                    )
                    self.dff[self.positions[idx][0], self.positions[idx][1]] += 1

                # 衝突ペナルティ
                for idx in agents:
                    chosen_idx, state, neighbor_coords = decisions[idx]
                    next_state = state
                    self.Q[state][chosen_idx] += self.alpha * (
                        -1 + self.gamma * np.max(self.Q.get(next_state, np.zeros_like(self.Q[state])))
                        - self.Q[state][chosen_idx]
                    )

        keep_mask = self.map_array[next_positions[:, 0], next_positions[:, 1]] != 3
        self.positions = next_positions[keep_mask]
        self.update_dff()

    def update_dff(self):
        diffuse = self.params["diffuse"]
        decay = self.params["decay"]
        new_dff = (1 - decay) * (1 - diffuse) * self.dff

        padded = np.pad(new_dff, 1, mode='constant')
        for dx, dy in self.neighbors:
            new_dff += decay * (1 - diffuse) / len(self.neighbors) * padded[1+dx:new_dff.shape[0]+1+dx, 1+dy:new_dff.shape[1]+1+dy]

        self.dff = new_dff
        threshold = 1e-4
        self.dff[self.dff < threshold] = 0
