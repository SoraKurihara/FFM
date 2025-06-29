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

    def step(self):
        move_requests = {}
        next_positions = np.copy(self.positions)

        for idx in range(self.positions.shape[0]):
            x, y = self.positions[idx]
            current_pos = np.array([x, y])

            offsets = np.array(self.neighbors)
            neighbor_coords = self.positions[idx] + offsets

            # occupied set の numpy版（自分を除外）
            other_positions = np.delete(self.positions, idx, axis=0)
            occupied = other_positions.view([('', other_positions.dtype)] * 2).reshape(-1)

            # 移動可能セル（壁・出口）
            mask = (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 0) | \
                (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3)
            neighbor_coords = neighbor_coords[mask]

            # occupied 判定
            if neighbor_coords.shape[0] > 0:
                neighbor_struct = neighbor_coords.view([('', neighbor_coords.dtype)] * 2).reshape(-1)
                mask = ~np.isin(neighbor_struct, occupied)
                neighbor_coords = neighbor_coords[mask]

            # ★現在地を追加して「動かない」という選択肢を入れる
            if neighbor_coords.shape[0] > 0:
                neighbor_coords = np.vstack([neighbor_coords, current_pos])

                exit_mask = self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3
                if np.any(exit_mask):
                    chosen_coord = neighbor_coords[exit_mask][0]
                    if tuple(chosen_coord) not in move_requests:
                        move_requests[tuple(chosen_coord)] = []
                    move_requests[tuple(chosen_coord)].append(idx)
                    continue

                sff_vals = self.sff[neighbor_coords[:, 0], neighbor_coords[:, 1]]
                dff_vals = self.dff[neighbor_coords[:, 0], neighbor_coords[:, 1]]

                score = -self.params["k_S"] * sff_vals + self.params["k_D"] * dff_vals
                score_max = np.max(score)

                probs = np.exp(score - score_max)
                probs_sum = probs.sum()
                if np.isfinite(probs_sum) and probs_sum != 0:
                    probs /= probs_sum
                    chosen_idx = np.random.choice(len(neighbor_coords), p=probs)
                    chosen = tuple(neighbor_coords[chosen_idx])
                    if chosen not in move_requests:
                        move_requests[chosen] = []
                    move_requests[chosen].append(idx)

        for target, agents in move_requests.items():
            if len(agents) == 1:
                next_positions[agents[0]] = target
                self.dff[self.positions[agents[0]][0], self.positions[agents[0]][1]] += 1
            else:
                if np.random.rand() < 0.5:
                    chosen = random.choice(agents)
                    next_positions[chosen] = target
                    self.dff[self.positions[chosen][0], self.positions[chosen][1]] += 1

        # 出口に到達した歩行者を除外
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

    def run(self, save_prefix=None, save_interval=100):
        buffer = []
        step = 0

        while self.positions.shape[0] > 0:
            self.step()
            buffer.append(np.copy(self.positions))
            step += 1

            if save_prefix and (step % save_interval == 0):
                np.savez_compressed(f"{save_prefix}_{step}.npz", positions=np.array(buffer, dtype=np.int32))
                buffer = []

        if save_prefix and buffer:
            np.savez_compressed(f"{save_prefix}_final.npz", positions=np.array(buffer, dtype=np.int32))
