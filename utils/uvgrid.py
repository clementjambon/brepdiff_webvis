from __future__ import annotations
from typing import List, Dict, Tuple, Union, Optional, Any

import numpy as np


class UvGrid:
    """
    A class for storing uv grids associcated with each face (3D)
    """

    eps = 1e-2  # to set coordinates near zero be masked

    def __init__(
        self,
        coord: Union[np.ndarray],
        grid_mask: Union[np.ndarray, None],
        empty_mask: Union[np.ndarray, None],
        normal: Union[np.ndarray, None] = None,
        prim_type: Union[np.ndarray, None] = None,
        face_adj: Union[Any, List[Any]] = None,
        coord_logits: Union[np.ndarray] = None,
    ):
        """
        :param coord: float array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            coordinates of each uv grid
        :param grid_mask: bool array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            whether each uv grid lies on surface. True indicates that grid lies on surface
        :param empty_mask: bool array of n_prims (or B x n_prims)
            whether each face is empty or not. True indicates face is empty
        :param normal: float array of n_prims x n_grid x n_grid x 3 (or B x n_prims x n_grid x n_grid x 3)
            normal of each uv grid
        :param prim_type: long array of n_prims (or B x n_prims)
            primitive type of each face

        """
        self.coord = coord
        self.grid_mask = grid_mask  # True indicates that coord lies on surface
        self.empty_mask = empty_mask  # True indicates empty!
        self.normal = normal
        self.prim_type = prim_type
        self.face_adj = face_adj
        self.coord_logits = coord_logits

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> UvGrid:
        # Easy: just instanciate with everything in data!
        return UvGrid(**data)


def uncat_uvgrids(uvgrids: UvGrid, cat_dim: int = 0, stride: int = 1) -> List[UvGrid]:
    """
    Given a single batch of uvgrids, uncat the parameters and create a list of batched uvgrids
    :param uvgrids:
    :param cat_dim: dimension over which to concat the uv grids (can be either batch_wise or prim_wise)
    :return:
    """
    if len(uvgrids.coord.shape) != 5:
        raise ValueError(
            "Un-Concatenation is only supported for batched uv grids for now!"
        )

    if cat_dim != 0:
        raise ValueError(
            "Un-Concatenation is only supported along batch or n_prim dimension!"
        )

    return [
        UvGrid(
            coord=uvgrids.coord[i : i + stride] if uvgrids.coord is not None else None,
            grid_mask=(
                uvgrids.grid_mask[i : i + stride]
                if uvgrids.grid_mask is not None
                else None
            ),
            empty_mask=(
                uvgrids.empty_mask[i : i + stride]
                if uvgrids.empty_mask is not None
                else None
            ),
            normal=(
                uvgrids.normal[i : i + stride] if uvgrids.normal is not None else None
            ),
            prim_type=(
                uvgrids.prim_type[i : i + stride]
                if uvgrids.prim_type is not None
                else None
            ),
            face_adj=(
                uvgrids.face_adj[i : i + stride]
                if uvgrids.face_adj is not None
                else None
            ),
        )
        for i in range(0, len(uvgrids.coord), stride)
    ]
