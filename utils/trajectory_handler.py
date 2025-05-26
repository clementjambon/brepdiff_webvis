from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from utils.trajectory import Trajectory

import deepdish as dd


class TrajectoryHandler:
    """
    The Trajectory Handler stores all trajectories and allows to save/replay them + export them
    """

    trajectories: Dict[int, Trajectory]

    # Maps i_traj to traj_idx (mostly for gui)
    traj_idx_map: List[int]

    def __init__(self):
        self.counter: int = -1
        self.trajectories = {}

        self.current_idx: int = -1

        self._update_traj_idx_map()

    def add(self, trajectory: Trajectory, replace_current: bool = False):
        if replace_current:
            self.trajectories[self.current_idx] = trajectory
        else:
            self.counter += 1
            self.trajectories[self.counter] = trajectory
            self.current_idx = self.counter
        self._update_traj_idx_map()

    def _update_traj_idx_map(self):
        self.traj_idx_map = sorted(list(self.trajectories.keys()))

    def deserialize(self, data: Dict[str, Any]):
        for traj_data in data["trajectories"].values():
            trajectory = Trajectory.deserialize(traj_data)
            self.add(trajectory)

    @staticmethod
    def load(path: str) -> TrajectoryHandler:
        data = dd.io.load(path)
        handler = TrajectoryHandler()
        handler.deserialize(data)
        return handler
