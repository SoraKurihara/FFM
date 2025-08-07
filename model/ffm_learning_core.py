import pickle
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
        self.prev_direction = np.zeros((self.N, 2), dtype=int)

        # 学習
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.paths = [[] for _ in range(self.N)]
        self.action_size = len(self.neighbors) + 1  # 停止も含む

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
        self.prev_direction = np.zeros((self.N, 2), dtype=int)
        self.paths = [[] for _ in range(self.N)]

    def get_block_index(self, x, y, block_size=3):
        return (x // block_size, y // block_size)

    def extract_state(self, x, y, idx):
        H, W = self.map_array.shape
        local_map = np.zeros((5, 5), dtype=int)  # デフォルトは 0（何もない）
        local_occupancy = np.zeros((5, 5), dtype=int)

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                xi = x + dx
                yi = y + dy
                if 0 <= xi < H and 0 <= yi < W:
                    local_map[dx + 2, dy + 2] = self.map_array[xi, yi]

        for px, py in self.positions:
            dx = px - x
            dy = py - y
            if -2 <= dx <= 2 and -2 <= dy <= 2:
                local_occupancy[dx + 2, dy + 2] = 1

        local_occupancy[2, 2] = 0  # 自分は除外

        # map + occupancy を結合
        combined = (local_map + (local_occupancy)).flatten()

        block_index = self.get_block_index(x, y)

        return (
            tuple(combined),
            block_index
        )

    def step(self, beta):
        move_requests = {}
        next_positions = np.copy(self.positions)
        occupied = self.positions.view([('', self.positions.dtype)] * 2).reshape(-1)
        decisions = {}
        arrived_indices = []

        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]
            state = self.extract_state(x, y, idx)

            offsets = np.array(self.neighbors)
            neighbor_coords = self.positions[idx] + offsets
            mask = (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 0) | \
                   (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3)
            neighbor_coords = neighbor_coords[mask]

            neighbor_struct = neighbor_coords.view([('', neighbor_coords.dtype)] * 2).reshape(-1)
            mask = ~np.isin(neighbor_struct, occupied)
            neighbor_coords = neighbor_coords[mask]

            current_pos = np.array([x, y])
            neighbor_coords = np.vstack([neighbor_coords, current_pos])  # 停止含む

            if state not in self.Q:
                self.Q[state] = np.zeros(self.action_size)

            action_to_index = {tuple(current_pos + np.array(offset)): i for i, offset in enumerate(self.neighbors)}
            action_to_index[tuple(current_pos)] = self.action_size - 1

            exit_mask = self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3
            if np.any(exit_mask):
                chosen_idx = np.where(exit_mask)[0][0]
                chosen_coord = tuple(neighbor_coords[chosen_idx])
                action = action_to_index.get(tuple(neighbor_coords[chosen_idx]), self.action_size - 1)
                self.paths[idx].append((state, action, 0))
                decisions[idx] = (chosen_coord, idx)
                move_requests.setdefault(chosen_coord, []).append(idx)
                continue

            sff_vals = self.sff[neighbor_coords[:, 0], neighbor_coords[:, 1]]
            dff_vals = self.dff[neighbor_coords[:, 0], neighbor_coords[:, 1]]
            q_vals = np.array([self.Q[state][action_to_index.get(tuple(pos), self.action_size - 1)]
                               for pos in neighbor_coords])

            val = beta * (-self.params["k_S"] * sff_vals + self.params["k_D"] * dff_vals) + (1 - beta) * q_vals
            score_max = np.max(val)
            probs = np.exp(val - score_max)
            probs_sum = probs.sum()
            if probs_sum == 0 or not np.isfinite(probs_sum):
                continue
            probs /= probs_sum

            chosen_idx = np.random.choice(len(neighbor_coords), p=probs)
            chosen_coord = tuple(neighbor_coords[chosen_idx])
            action = action_to_index.get(chosen_coord, self.action_size - 1)

            reward = -0.1 if np.array_equal(neighbor_coords[chosen_idx], current_pos) else 0.0
            self.paths[idx].append((state, action, reward))
            decisions[idx] = (chosen_coord, idx)
            move_requests.setdefault(chosen_coord, []).append(idx)

        for target, agents in move_requests.items():
            if len(agents) == 1:
                idx = agents[0]
                next_positions[idx] = target
                self.prev_direction[idx] = target - self.positions[idx]
                self.dff[self.positions[idx][0], self.positions[idx][1]] += 1
                if self.map_array[target[0], target[1]] == 3:
                    arrived_indices.append(idx)
            else:
                for idx in agents:
                    if self.paths[idx]:
                        state, action, _ = self.paths[idx][-1]
                        self.paths[idx][-1] = (state, action, -1)
                if np.random.rand() < 0.5:
                    chosen = random.choice(agents)
                    next_positions[chosen] = target
                    self.prev_direction[chosen] = target - self.positions[chosen]
                    self.dff[self.positions[chosen][0], self.positions[chosen][1]] += 1
                    if self.map_array[target[0], target[1]] == 3:
                        arrived_indices.append(chosen)

        self.positions = next_positions

        for idx in sorted(arrived_indices, reverse=True):
            path = self.paths[idx]
            if path:
                state, action, _ = path[-1]
                path[-1] = (state, action, 10)

            G = 0
            for state, action, reward in reversed(path):
                G = reward + self.gamma * G
                if state not in self.Q:
                    self.Q[state] = np.zeros(self.action_size)
                self.Q[state][action] += self.alpha * (G - self.Q[state][action])

            self.positions = np.delete(self.positions, idx, axis=0)
            self.prev_direction = np.delete(self.prev_direction, idx, axis=0)
            del self.paths[idx]

        self.update_dff()

    def update_dff(self):
        diffuse = self.params["diffuse"]
        decay = self.params["decay"]
        new_dff = (1 - decay) * (1 - diffuse) * self.dff
        padded = np.pad(new_dff, 1, mode='constant')
        for dx, dy in self.neighbors:
            new_dff += decay * (1 - diffuse) / len(self.neighbors) * padded[1+dx:new_dff.shape[0]+1+dx, 1+dy:new_dff.shape[1]+1+dy]
        self.dff = new_dff
        self.dff[self.dff < 1e-4] = 0

    def save_Q(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.Q, f)
