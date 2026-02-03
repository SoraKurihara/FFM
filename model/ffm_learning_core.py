import pickle
import random
from typing import Dict, List, Tuple

import numpy as np


class FloorFieldModel:
    """
    Floor-Field Model with target-centric Q-learning (Monte Carlo, reverse).

    State key  : (combined3x3.tobytes(), (block_x, block_y))
      - combined3x3 = map3x3 + occ3x3
        map codes: free=0, wall/OOB=2, exit=3
        occ3x3   : other agents only (0/1). Ensure occ3x3[map3x3!=0]=0
      - block_x, block_y = x//3, y//3 (coarse location)

    Action set : [FROM_UP, FROM_DOWN, FROM_LEFT, FROM_RIGHT, FROM_SELF]
                 indices: 0,1,2,3,4 (STOP is FROM_SELF)

    Selection  : For each candidate target T, build S_target(T) and read Q[S_target][from_dir].
                 Logit(T) = beta * (-k_S * SFF[T]) + k_D * DFF[T] + k_Q * Q_value
                 STOP uses: Logit(STOP) = k_Q * Q_value   (no SFF/DFF for STOP)
                 Missing Q -> fallback 0 (do NOT create entries on read path).

    Learning   : Reverse Monte Carlo with per-step immediate rewards already logged.
                 On agent exit, add exit_reward to the last step and back up G.
                 On conflict loss, overwrite last reward with -collision_penalty.

    DFF update : Increment only when an agent actually moved (not on STOP).
    """

    # Action indices (Neumann + STOP)
    FROM_UP = 0
    FROM_DOWN = 1
    FROM_LEFT = 2
    FROM_RIGHT = 3
    FROM_SELF = 4  # STOP

    def __init__(self,
                 map_array: np.ndarray,
                 sff_path: str,
                 N: int,
                 params: Dict = None):
        default_params = {
            "k_S": 3.0,
            "k_D": 1.0,
            "k_Q": 1.0,
            "diffuse": 0.2,
            "decay": 0.2,
            "neighborhood": "neumann",
            # rewards (costs are positive here; applied as negative rewards)
            "step_penalty": 0.00,#0.01,
            "stop_penalty": 0.00,#0.30,
            "collision_penalty": 0.00,#0.70,
            "exit_reward": 100.0,
            "timeout_penalty": 50.0,
            "max_steps": 500,
        }
        self.params = default_params if params is None else {**default_params, **params}

        # map / fields
        self.map_array = map_array.astype(np.uint8)
        self.sff = np.load(sff_path, mmap_mode='r')  # static floor field (distance-like)
        self.dff = np.zeros_like(self.map_array, dtype=np.float32)  # dynamic (footprints)

        # agents
        self.N = int(N)
        self.positions = self._initialize_agents()
        self.prev_direction = np.zeros((self.N, 2), dtype=np.int16)

        # neighborhood (force Neumann for action mapping)
        self.neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

        # learning (Monte Carlo)
        self.Q: Dict[Tuple[bytes, Tuple[int, int]], np.ndarray] = {}
        self.alpha = 0.1
        self.gamma = 0.99
        self.paths: List[List[Tuple[Tuple[bytes, Tuple[int, int]], int, float]]] = [
            [] for _ in range(self.N)
        ]
        self.action_size = 5  # FROM_UP/DOWN/LEFT/RIGHT/SELF

        # workspace buffers
        self._H, self._W = self.map_array.shape
        self._passable_mask = (self.map_array == 0) | (self.map_array == 3)

        # episode step counter & limits
        self.max_steps = int(self.params["max_steps"])
        self._step_count = 0

    # -----------------------------
    # Initialization & Reset
    # -----------------------------
    def _initialize_agents(self) -> np.ndarray:
        """Place N agents uniformly at random on free cells (map==0)."""
        free_cells = np.argwhere(self.map_array == 0)
        idx = np.random.choice(len(free_cells), self.N, replace=False)
        return free_cells[idx].astype(np.int16)

    def reset(self) -> None:
        self.positions = self._initialize_agents()
        self.prev_direction = np.zeros((self.N, 2), dtype=np.int16)
        self.dff.fill(0.0)
        self.paths = [[] for _ in range(self.N)]
        self._step_count = 0

    # -----------------------------
    # State construction (target-centric)
    # -----------------------------
    @staticmethod
    def _block_index(x: int, y: int, block_size: int = 3) -> Tuple[int, int]:
        return (x // block_size, y // block_size)

    def _combined3x3_at_target(self, tx: int, ty: int, occ_grid: np.ndarray, self_xy: Tuple[int, int]) -> np.ndarray:
        """Build combined3x3 = map3x3 + occ3x3 around target (tx,ty).
        - map codes: free=0, wall/OOB=2, exit=3
        - occ3x3: others only (0/1), and enforce occ3x3[map!=0]=0
        - self agent must be excluded even if inside the 3x3 window.
        """
        H, W = self._H, self._W
        c = np.full((3, 3), 2, dtype=np.uint8)  # OOB=2
        x0, y0 = tx - 1, ty - 1
        # fill map window
        ix0, ix1 = max(0, x0), min(H, tx + 2)
        iy0, iy1 = max(0, y0), min(W, ty + 2)
        c[(ix0 - x0):(ix1 - x0), (iy0 - y0):(iy1 - y0)] = self.map_array[ix0:ix1, iy0:iy1]

        # occupancy window (copy from occ_grid)
        occ = np.zeros((3, 3), dtype=np.uint8)
        occ[(ix0 - x0):(ix1 - x0), (iy0 - y0):(iy1 - y0)] = occ_grid[ix0:ix1, iy0:iy1].astype(np.uint8)

        # # exclude self if inside the 3x3
        # sx, sy = self_xy
        # if abs(sx - tx) <= 1 and abs(sy - ty) <= 1:
        #     occ[sx - tx + 1, sy - ty + 1] = 0

        # enforce occ only on free cells
        occ[c != 0] = 0
        return (c + occ).astype(np.uint8)

    # -----------------------------
    # One simulation step
    # -----------------------------
    def step(self, beta: float) -> None:
        H, W = self._H, self._W
        k_S = float(self.params["k_S"])  # SFF weight
        k_D = float(self.params["k_D"])  # DFF weight
        k_Q = float(self.params["k_Q"])  # Q   weight

        step_pen = float(self.params["step_penalty"])     # cost (will be negative reward)
        stop_pen = float(self.params["stop_penalty"])     # cost
        coll_pen = float(self.params["collision_penalty"])  # cost

        # step counter
        self._step_count += 1

        # occupancy grid (bool) at current frame
        occ_grid = np.zeros((H, W), dtype=np.bool_)
        occ_grid[self.positions[:, 0], self.positions[:, 1]] = True

        move_requests: Dict[Tuple[int, int], List[int]] = {}
        decisions: Dict[int, Tuple[Tuple[int, int], int]] = {}  # idx -> (target, action_idx)
        next_positions = self.positions.copy()
        arrived_indices: List[int] = []

        # fast mask for current occupancy (structured view trick is not needed here)
        occupied_struct = self.positions.view([('', self.positions.dtype)] * 2).reshape(-1)

        for idx in range(self.positions.shape[0]):
            x, y = int(self.positions[idx, 0]), int(self.positions[idx, 1])

            # candidate targets: 4-neighbors that are passable & not currently occupied, plus STOP
            cand = []  # list of (tx,ty, action_idx)
            for a_idx, (dx, dy) in enumerate(self.neighbors):
                tx, ty = x + dx, y + dy
                if 0 <= tx < H and 0 <= ty < W and self._passable_mask[tx, ty]:
                    # cannot target a cell currently occupied at time t
                    key_struct = np.array([(tx, ty)], dtype=self.positions.dtype).view([('', self.positions.dtype)] * 2).reshape(-1)[0]
                    if key_struct not in occupied_struct:
                        cand.append((tx, ty, self._dir_to_from(x, y, tx, ty)))
            # STOP
            cand.append((x, y, FloorFieldModel.FROM_SELF))

            # build logits for softmax
            logits: List[float] = []
            states_for_log: List[Tuple[Tuple[bytes, Tuple[int, int]], int]] = []  # (state_key, action_idx)
            for (tx, ty, a_idx) in cand:
                # target-centric state
                combined = self._combined3x3_at_target(tx, ty, occ_grid, (x, y))
                state_key = (combined.tobytes(), self._block_index(tx, ty))
                q_vec = self.Q.get(state_key, None)
                q_val = 0.0 if (q_vec is None) else float(q_vec[a_idx])

                logit = beta * (-k_S * float(self.sff[tx, ty])) + k_D * float(self.dff[tx, ty]) + (1-beta) * k_Q * q_val
                logits.append(logit)
                states_for_log.append((state_key, a_idx))

            # softmax over logits
            logits_arr = np.asarray(logits, dtype=np.float64)
            mx = np.max(logits_arr)
            probs = np.exp(logits_arr - mx)
            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                # skip (no decision) – extremely unlikely; treat as STOP locally
                chosen = len(cand) - 1  # STOP entry
            else:
                probs /= s
                chosen = int(np.random.choice(len(cand), p=probs))

            tx, ty, a_idx = cand[chosen]
            state_key, a_for_log = states_for_log[chosen]

            # immediate reward (preliminary). Will be overwritten if conflict occurs or exit happens
            if a_idx == FloorFieldModel.FROM_SELF:
                reward = -stop_pen
            else:
                reward = -step_pen

            # log (target-centric)
            self._ensure_qvec(state_key)
            self.paths[idx].append((state_key, a_for_log, reward))

            # register move request
            decisions[idx] = ((tx, ty), a_idx)
            move_requests.setdefault((tx, ty), []).append(idx)

        # resolve moves & apply DFF and arrivals
        for (tx, ty), agents in move_requests.items():
            if len(agents) == 1:
                i = agents[0]
                src = (int(self.positions[i, 0]), int(self.positions[i, 1]))
                if (tx, ty) != src:
                    # move succeeds
                    self.dff[src[0], src[1]] += 1.0
                    next_positions[i, 0], next_positions[i, 1] = tx, ty
                    self.prev_direction[i] = np.array([tx - src[0], ty - src[1]], dtype=np.int16)
                # arrival?
                if self.map_array[tx, ty] == 3:
                    arrived_indices.append(i)
            else:
                # conflict: one randomly wins, others lose
                winner = random.choice(agents)
                for i in agents:
                    src = (int(self.positions[i, 0]), int(self.positions[i, 1]))
                    if i == winner:
                        if (tx, ty) != src:
                            self.dff[src[0], src[1]] += 1.0
                            next_positions[i, 0], next_positions[i, 1] = tx, ty
                            self.prev_direction[i] = np.array([tx - src[0], ty - src[1]], dtype=np.int16)
                        if self.map_array[tx, ty] == 3:
                            arrived_indices.append(i)
                    else:
                        # overwrite the last reward with collision penalty (loss)
                        if self.paths[i]:
                            sk, ac, _ = self.paths[i][-1]
                            self.paths[i][-1] = (sk, ac, -coll_pen)

        # commit positions
        self.positions = next_positions

        # process arrivals: add exit reward to last step, then MC backup + remove agent
        for idx in sorted(arrived_indices, reverse=True):
            if self.paths[idx]:
                sk, ac, _ = self.paths[idx][-1]
                self.paths[idx][-1] = (sk, ac, float(self.params["exit_reward"]))

            # reverse MC backup
            G = 0.0
            for sk, ac, r in reversed(self.paths[idx]):
                G = r + self.gamma * G
                self._ensure_qvec(sk)
                self.Q[sk][ac] += self.alpha * (G - self.Q[sk][ac])

            # remove agent data
            self.positions = np.delete(self.positions, idx, axis=0)
            self.prev_direction = np.delete(self.prev_direction, idx, axis=0)
            del self.paths[idx]

        # evolve DFF field
        self._update_dff()

        # finalize remaining agents if max_steps reached
        if self._step_count >= int(self.params["max_steps"]) and self.positions.shape[0] > 0:
            self.finalize_timeouts()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _ensure_qvec(self, state_key: Tuple[bytes, Tuple[int, int]]) -> None:
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(self.action_size, dtype=np.float32)

    @staticmethod
    def _dir_to_from(x: int, y: int, tx: int, ty: int) -> int:
        dx, dy = tx - x, ty - y
        if dx == -1 and dy == 0:
            return FloorFieldModel.FROM_DOWN   # coming from below to go up
        if dx == 1 and dy == 0:
            return FloorFieldModel.FROM_UP     # coming from above to go down
        if dx == 0 and dy == -1:
            return FloorFieldModel.FROM_RIGHT  # coming from right to go left
        if dx == 0 and dy == 1:
            return FloorFieldModel.FROM_LEFT   # coming from left to go right
        return FloorFieldModel.FROM_SELF

    def _update_dff(self) -> None:
        diffuse = float(self.params["diffuse"])
        decay = float(self.params["decay"])
        H, W = self._H, self._W

        base = (1.0 - decay) * (1.0 - diffuse) * self.dff
        padded = np.pad(base, 1, mode='constant')
        acc = np.zeros_like(base)
        # 8-neighborhood diffusion (Moore) for smoother footprint spread
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in neigh:
            acc += padded[1 + dx:H + 1 + dx, 1 + dy:W + 1 + dy]
        acc *= decay * (1.0 - diffuse) / len(neigh)
        self.dff = base + acc
        self.dff[self.dff < 1e-4] = 0.0

    # -----------------------------
    # Timeouts finalization
    # -----------------------------
    def finalize_timeouts(self) -> None:
        """Force-finish the episode when step cap is reached.
        - Append a timeout penalty to each remaining agent's path (STOP @ current cell)
        - Monte Carlo backup for each, then remove all.
        """
        if self.positions.shape[0] == 0:
            return
        H, W = self._H, self._W
        timeout_pen = float(self.params["timeout_penalty"])  # cost

        # build current occupancy grid
        occ_grid = np.zeros((H, W), dtype=np.bool_)
        occ_grid[self.positions[:, 0], self.positions[:, 1]] = True

        # finalize each remaining agent
        for idx in range(self.positions.shape[0]):
            x, y = int(self.positions[idx, 0]), int(self.positions[idx, 1])
            combined = self._combined3x3_at_target(x, y, occ_grid, (x, y))
            state_key = (combined.tobytes(), self._block_index(x, y))
            self._ensure_qvec(state_key)

            # push final timeout step (STOP at current cell)
            self.paths[idx].append((state_key, FloorFieldModel.FROM_SELF, -timeout_pen))

            # reverse MC backup
            G = 0.0
            for sk, ac, r in reversed(self.paths[idx]):
                G = r + self.gamma * G
                self._ensure_qvec(sk)
                self.Q[sk][ac] += self.alpha * (G - self.Q[sk][ac])

        # clear all remaining agents
        self.positions = np.empty((0, 2), dtype=self.positions.dtype)
        self.prev_direction = np.empty((0, 2), dtype=self.prev_direction.dtype)
        self.paths = []

    # -----------------------------
    # Persistence
    # -----------------------------
    def save_Q(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.Q, f)
