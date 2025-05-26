from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np

from utils.uvgrid import UvGrid, uncat_uvgrids
from utils.denoiser_config import DenoiserConfig


@dataclass
class Trajectory:
    """
    A trajectory stores all the Tokens/UvGrid/Config for both Forward and Backward trajectories.
    This allows to store all parameters and potentially export trajectories for reproducibility.
    """

    # Frozen Primitives
    frozen_prims: List[bool]

    # Indexed by t (as done for sampling)
    traj_uv_grids: Dict[int, UvGrid]

    # Stores current `i_t` = different from i_t because i_t directly index into the set of
    # possible  whereas i_step is used to keep track of denoising steps! NB: this means
    # that the "renoiser" will be in charge of manually incrementing i_t to display the new
    # state but this should be much safer!
    i_t: int
    i_step: int

    # Stores current slider `i_batch`
    i_batch: int

    # Keymap mapping i_t -> t (it's actually a list)
    keymap: List[int]
    inv_keymap: Dict[int, int]

    # (Optional) Reconstructed Brep
    brep_mesh_vertices: List[Optional[np.ndarray]]
    brep_mesh_faces: List[Optional[np.ndarray]]
    brep_solid: List[Optional[Any]]  # Not serialized!

    @property
    def batch_size(self):
        return next(iter(self.traj_uv_grids.values())).coord.shape[0]

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Trajectory:
        keymap = data["keymap"]
        frozen_prims = data["frozen_prims"]
        i_step = data["i_step"]
        # Worse case we show the first step
        i_t = min(max(data["i_t"], 0), len(keymap) - 1) if "i_t" in data else 0
        denoiser_config = DenoiserConfig.from_yaml(data["denoiser_config"])
        uvgrids = uncat_uvgrids(
            UvGrid.deserialize(data["traj_uv_grids"]),
            cat_dim=0,
            stride=denoiser_config.batch_size,
        )
        uvgrids = {k: v for k, v in zip(keymap, uvgrids)}

        trajectory = Trajectory(
            t=keymap[i_t],
            uv_grids=uvgrids,
            frozen_prims=frozen_prims,
            i_step=i_step,
            i_t=i_t,
        )

        # if "brep_mesh_vertices" in data and "brep_mesh_faces" in data:

        #     if isinstance(data["brep_mesh_vertices"], list) and isinstance(
        #         data["brep_mesh_faces"], list
        #     ):
        #         trajectory.brep_mesh_vertices = data["brep_mesh_vertices"]
        #         trajectory.brep_mesh_faces = data["brep_mesh_faces"]
        #     # DEPRECATED but implemented for backward compatibility
        #     elif isinstance(data["brep_mesh_vertices"], np.ndarray) and isinstance(
        #         data["brep_mesh_faces"], np.ndarray
        #     ):
        #         trajectory.brep_mesh_vertices = [data["brep_mesh_vertices"]] + [
        #             None
        #         ] * (tmp_uv_grids.coord.shape[0] - 1)
        #         trajectory.brep_mesh_faces = [data["brep_mesh_faces"]] + [None] * (
        #             tmp_uv_grids.coord.shape[0] - 1
        #         )

        return trajectory

    def _update_key_map(self):
        self.keymap = sorted(list(self.traj_uv_grids.keys()))
        self.inv_keymap = {t: i_t for i_t, t in enumerate(self.keymap)}

    def __init__(
        self,
        t: int,
        uv_grids: UvGrid,
        frozen_prims: List[bool],
        i_step: int,
        i_t: int = 0,
    ) -> None:
        # Set Frozen Primitives
        self.frozen_prims = frozen_prims

        # Set i_step
        self.i_step = i_step

        # Set i_t
        self.i_t = i_t

        # Set i_batch to 0!
        self.i_batch = 0

        # Store values
        self.traj_uv_grids = uv_grids if isinstance(uv_grids, dict) else {t: uv_grids}

        # Debug: this is for sanity
        self.prev_t = t

        # Brep null initialization
        batch_size = next(iter(self.traj_uv_grids.values())).coord.shape[0]
        self.brep_mesh_vertices = [None] * batch_size
        self.brep_mesh_faces = [None] * batch_size
        self.brep_solid = [None] * batch_size

        self._update_key_map()

    def has(self, t: int):
        test = t in self.traj_uv_grids
        if not test:
            print(f"WARNING: {t} if not in the current trajectory!")
        return test

    @property
    def size(self):
        return len(self.traj_uv_grids)
