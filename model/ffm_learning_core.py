import pickle
import random
import struct
from typing import Dict, List, Tuple

import numpy as np


class FloorFieldModel:
    """
    - Selection: softmax over pre-softmax mix val = beta*FFM + (1-beta)*k_Q*Q
    - SFF/DFF kept; DFF evolves each step
    - MC (reverse) update at exit/timeout
    - Collision penalty only to agents who failed to move
    - Fast state encoding: bytes (3x3 local map + 3x3 occupancy, uint8) + packed block index
    - 3x3 extraction via padding (no nested loops)
    """

    def __init__(
        self,
        map_array: np.ndarray,
        sff_path: str,
        N: int,
        params: Dict = None,
        state_encoding: str = "bytes",   # "bytes" (fast, default) or "tuple"
        block_size: int = 3,
        state_radius: int = 1,           # ← 3x3 (=2*1+1)
    ):
        # ---------------- Params ----------------
        default_params = {
            "k_S": 3.0,
            "k_D": 1.0,
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "moore",      # 実行側のconfigで "neumann" に上書きされます
            # rewards & episode control
            "success_reward": 100.0,
            "timeout_penalty": -20.0,
            "step_penalty": -0.01,
            "stop_penalty": -0.2,
            "collision_penalty": -1.0,
            "direction_keep_bonus": 0.05,
            "backtrack_penalty": 0.1,     # subtracted when applied
            "max_steps": 500,
            # Q scaling
            "k_Q": 1.0,                   # 先輩はここを 5 に設定
        }
        self.params = default_params if params is None else {**default_params, **params}

        # ---------------- Fields ----------------
        self.map_array = map_array.astype(np.uint8, copy=False)
        self.sff = np.load(sff_path, mmap_mode="r")   # FFMに使用
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)

        self.N = int(N)
        self.block_size = int(block_size)
        self.state_encoding = state_encoding
        self.state_radius = int(state_radius)         # ← 3x3 用
        self.win = 2 * self.state_radius + 1

        # neighbors & action mapping
        self.neighbors = self._get_neighbors(self.params["neighborhood"])
        self.neighbor_offsets = np.array(self.neighbors, dtype=np.int8)  # (K,2)
        self.action_size = len(self.neighbors) + 1  # + stop
        self.stop_action = self.action_size - 1
        self.delta_to_action = {tuple(delta): i for i, delta in enumerate(self.neighbors)}
        self.delta_to_action[(0, 0)] = self.stop_action

        # agent state
        self.positions = self._initialize_agents()
        self.prev_direction = np.zeros((self.N, 2), dtype=np.int8)

        # RL
        self.Q: Dict[bytes, np.ndarray] = {}
        self.alpha = 0.1
        self.gamma = 0.99                 # ← デフォルトを 0.99 に
        self.paths: List[List[Tuple[bytes, int, float]]] = [[] for _ in range(self.N)]

        # episode
        self.step_count = 0

        # static padded map for 3x3 slicing
        self._map_padded = np.pad(self.map_array, self.state_radius, mode="constant")

    # ---------------- Init helpers ----------------
    def _get_neighbors(self, neighborhood: str):
        if neighborhood == "neumann":
            return [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # moore
        return [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),            (0, 1),
                (1, -1),  (1, 0),   (1, 1)]

    def _initialize_agents(self) -> np.ndarray:
        free_cells = np.argwhere(self.map_array == 0)
        selected = free_cells[np.random.choice(len(free_cells), self.N, replace=False)]
        return selected.astype(np.int32, copy=False)

    # ---------------- Public API ----------------
    def reset(self):
        self.positions = self._initialize_agents()
        self.dff.fill(0.0)
        self.prev_direction.fill(0)
        self.paths = [[] for _ in range(self.N)]
        self.step_count = 0

    def save_Q(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.Q, f)

    # ---------------- State encoding ----------------
    def _get_block_index(self, x: int, y: int) -> Tuple[int, int]:
        return (x // self.block_size, y // self.block_size)

    def _encode_state_tuple(self, x: int, y: int, occ_pad: np.ndarray):
        r = self.state_radius
        mp = self._map_padded[x: x + 2*r + 1, y: y + 2*r + 1].astype(np.uint8, copy=False)
        oc = occ_pad[x: x + 2*r + 1, y: y + 2*r + 1].astype(np.uint8, copy=True)
        oc[r, r] = 0  # 自分は除外
        combined = (mp + oc).reshape(-1)
        return (tuple(combined.tolist()), self._get_block_index(x, y))

    # ---------------- Main step ----------------
    def step(self, beta: float):
        """
        beta is provided from the outer schedule.
        Selection: softmax over pre-softmax mix val = beta*FFM + (1-beta)*k_Q*Q
        """
        self.step_count += 1
        H, W = self.map_array.shape
        r = self.state_radius
        win = self.win

        # occupancy (True: occupied now)
        occupancy = np.zeros((H, W), dtype=bool)
        occupancy[self.positions[:, 0], self.positions[:, 1]] = True
        occupancy_padded = np.pad(occupancy, r, mode="constant")

        move_requests: Dict[Tuple[int, int], List[int]] = {}
        next_positions = self.positions.copy()
        arrived_indices: List[int] = []

        k_S = float(self.params["k_S"])
        k_D = float(self.params["k_D"])
        k_Q = float(self.params.get("k_Q", 1.0))

        for idx in range(self.positions.shape[0]):
            x, y = int(self.positions[idx, 0]), int(self.positions[idx, 1])

            # ---- state (3x3 map + 3x3 occupancy, center=0) ----
            mp = self._map_padded[x: x + win, y: y + win].astype(np.uint8, copy=False)
            oc = occupancy_padded[x: x + win, y: y + win].astype(np.uint8, copy=True)
            oc[r, r] = 0
            combined = mp + oc
            bx, by = self._get_block_index(x, y)
            if self.state_encoding == "bytes":
                state = combined.tobytes() + struct.pack("HH", bx, by)
            else:
                state = (tuple(combined.reshape(-1).tolist()), (bx, by))

            # init Q[state]
            if state not in self.Q:
                self.Q[state] = np.zeros(self.action_size, dtype=np.float32)

            # ---- candidates (legal per map; exclude currently occupied), + stop ----
            neighbor_coords = self.positions[idx] + self.neighbor_offsets  # (K,2)
            legal = (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 0) | \
                    (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3)
            neighbor_coords = neighbor_coords[legal]

            if neighbor_coords.size > 0:
                occ_mask = ~occupancy[neighbor_coords[:, 0], neighbor_coords[:, 1]]
                neighbor_coords = neighbor_coords[occ_mask]

            current_pos = np.array([x, y], dtype=np.int32)
            if neighbor_coords.size == 0:
                neighbor_coords = current_pos.reshape(1, 2)
            else:
                neighbor_coords = np.vstack([neighbor_coords, current_pos])

            deltas = neighbor_coords - current_pos
            action_indices = np.fromiter(
                (self.delta_to_action.get((int(dx), int(dy)), self.stop_action) for dx, dy in deltas),
                dtype=np.int32, count=deltas.shape[0]
            )

            # ---- exit immediate (deterministic) ----
            exit_mask = (self.map_array[neighbor_coords[:, 0], neighbor_coords[:, 1]] == 3)
            if np.any(exit_mask):
                chosen_idx = int(np.where(exit_mask)[0][0])
                chosen_coord = tuple(neighbor_coords[chosen_idx])
                action = int(action_indices[chosen_idx])

                reward = float(self.params["step_penalty"])
                if chosen_coord == (x, y):
                    reward += (self.params["stop_penalty"] - self.params["step_penalty"])

                self.paths[idx].append((state, action, reward))
                move_requests.setdefault(chosen_coord, []).append(idx)
                continue

            # ---- values: pre-softmax mix ----
            sff_vals = self.sff[neighbor_coords[:, 0], neighbor_coords[:, 1]]
            dff_vals = self.dff[neighbor_coords[:, 0], neighbor_coords[:, 1]]
            q_vals = self.Q[state][action_indices]

            ffm_vals = (-k_S * sff_vals + k_D * dff_vals).astype(np.float32, copy=False)
            val = beta * ffm_vals + (1.0 - beta) * (k_Q * q_vals)

            # softmax (stable)
            logits = val - np.max(val)
            probs = np.exp(logits.astype(np.float64))
            sum_probs = probs.sum()
            if sum_probs == 0.0 or not np.isfinite(sum_probs):
                probs = np.ones_like(probs, dtype=np.float64) / probs.size
            else:
                probs /= sum_probs

            chosen_idx = int(np.random.choice(len(neighbor_coords), p=probs))
            chosen_coord = tuple(neighbor_coords[chosen_idx])
            action = int(action_indices[chosen_idx])

            reward = float(self.params["step_penalty"])
            if chosen_coord == (x, y):
                reward += (self.params["stop_penalty"] - self.params["step_penalty"])

            self.paths[idx].append((state, action, reward))
            move_requests.setdefault(chosen_coord, []).append(idx)

        # ---------------- resolve moves ----------------
        arrived_indices.clear()
        for target, agents in move_requests.items():
            if len(agents) == 1:
                idx = agents[0]
                prev_pos = tuple(self.positions[idx])
                next_positions[idx] = target

                move_vec = (np.array(target) - self.positions[idx]).astype(np.int8, copy=False)
                if (move_vec != 0).any():
                    if np.array_equal(move_vec, self.prev_direction[idx]):
                        s, a, r = self.paths[idx][-1]
                        self.paths[idx][-1] = (s, a, r + self.params["direction_keep_bonus"])
                    elif np.array_equal(move_vec, -self.prev_direction[idx]):
                        s, a, r = self.paths[idx][-1]
                        self.paths[idx][-1] = (s, a, r - self.params["backtrack_penalty"])

                self.prev_direction[idx] = move_vec
                if (move_vec != 0).any():
                    self.dff[prev_pos[0], prev_pos[1]] += 1.0

                if self.map_array[target[0], target[1]] == 3:
                    arrived_indices.append(idx)

            else:
                # collision group: maybe one winner passes
                allow_one = (np.random.rand() < 0.5)
                winner = None
                if allow_one:
                    winner = random.choice(agents)
                    prev_pos = tuple(self.positions[winner])
                    next_positions[winner] = target
                    move_vec = (np.array(target) - self.positions[winner]).astype(np.int8, copy=False)

                    if (move_vec != 0).any():
                        if np.array_equal(move_vec, self.prev_direction[winner]):
                            s, a, r = self.paths[winner][-1]
                            self.paths[winner][-1] = (s, a, r + self.params["direction_keep_bonus"])
                        elif np.array_equal(move_vec, -self.prev_direction[winner]):
                            s, a, r = self.paths[winner][-1]
                            self.paths[winner][-1] = (s, a, r - self.params["backtrack_penalty"])

                    self.prev_direction[winner] = move_vec
                    if (move_vec != 0).any():
                        self.dff[prev_pos[0], prev_pos[1]] += 1.0

                    if self.map_array[target[0], target[1]] == 3:
                        arrived_indices.append(winner)

                # losers only: collision penalty
                for idx in agents:
                    if (winner is not None) and (idx == winner):
                        continue
                    if self.paths[idx]:
                        s, a, _ = self.paths[idx][-1]
                        self.paths[idx][-1] = (s, a, float(self.params["collision_penalty"]))
                    # prev_direction unchanged; no DFF increment

        # commit new positions
        self.positions = next_positions

        # ---------------- arrivals: MC reverse update ----------------
        for idx in sorted(arrived_indices, reverse=True):
            path = self.paths[idx]
            if path:
                s, a, r = path[-1]
                path[-1] = (s, a, r + float(self.params["success_reward"]))
            self._mc_update_and_remove(idx)

        # ---------------- timeout handling ----------------
        if self.step_count >= int(self.params["max_steps"]) and self.positions.size > 0:
            for local_idx in reversed(range(self.positions.shape[0])):
                path = self.paths[local_idx]
                if path:
                    s, a, r = path[-1]
                    path[-1] = (s, a, r + float(self.params["timeout_penalty"]))
                self._mc_update_and_remove(local_idx)
            self.step_count = 0

        self.update_dff()

    # ---------------- Learning update ----------------
    def _mc_update_and_remove(self, idx: int):
        path = self.paths[idx]
        G = 0.0
        for state, action, reward in reversed(path):
            G = reward + self.gamma * G
            if state not in self.Q:
                self.Q[state] = np.zeros(self.action_size, dtype=np.float32)
            self.Q[state][action] += self.alpha * (G - self.Q[state][action])

        self.positions = np.delete(self.positions, idx, axis=0)
        self.prev_direction = np.delete(self.prev_direction, idx, axis=0)
        del self.paths[idx]

    # ---------------- DFF evolution ----------------
    def update_dff(self):
        diffuse = float(self.params["diffuse"])
        decay = float(self.params["decay"])

        new_dff = (1.0 - decay) * (1.0 - diffuse) * self.dff
        padded = np.pad(new_dff, 1, mode="constant")

        denom = len(self.neighbors)
        coef = decay * (1.0 - diffuse) / denom
        for dx, dy in self.neighbors:
            new_dff += coef * padded[1 + dx:new_dff.shape[0] + 1 + dx,
                                     1 + dy:new_dff.shape[1] + 1 + dy]
        self.dff = new_dff
        self.dff[self.dff < 1e-4] = 0.0
