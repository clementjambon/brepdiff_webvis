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

    def meshify(
        self, face_idx: int, use_grid_mask: bool = True, batch_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates uv grid mesh for a face

        Args:
            face_idx: Index of face/primitive to meshify
            use_grid_mask: Whether to use grid mask to filter vertices
            batch_idx: Which batch item to meshify. If None, assumes unbatched data
        Returns:
            - vertices: array of V x 3
            - faces: array of idx of F x 3
        """
        # Handle batched case by selecting correct batch
        is_batched = len(self.coord.shape) == 5
        if is_batched:
            if batch_idx is None:
                raise ValueError("batch_idx must be provided for batched UvGrid")
            coord = self.coord[batch_idx]
            grid_mask = (
                self.grid_mask[batch_idx] if self.grid_mask is not None else None
            )
        else:
            coord = self.coord
            grid_mask = self.grid_mask

        if use_grid_mask and (grid_mask is not None):
            grid_mask = grid_mask[face_idx]
        else:
            grid_mask = np.ones(coord[face_idx].shape[:2], dtype=bool)

        coord = coord[face_idx]

        # create valid vertices from grid_mask and mapping from uv to vertex idx
        vertices, faces = [], []
        uv2vertex_idx = -1 * np.ones(coord.shape[:2], dtype=np.int32)
        for i in range(coord.shape[0]):
            for j in range(coord.shape[1]):
                if not grid_mask[i, j]:
                    continue
                uv2vertex_idx[i, j] = len(vertices)
                vertices.append(coord[i, j])

        # create faces
        for i in range(coord.shape[0] - 1):
            for j in range(coord.shape[1] - 1):
                if not grid_mask[i, j]:
                    continue
                if not grid_mask[i + 1, j + 1]:
                    continue
                if grid_mask[i + 1, j]:
                    faces.append(
                        (
                            uv2vertex_idx[i, j],
                            uv2vertex_idx[i + 1, j],
                            uv2vertex_idx[i + 1, j + 1],
                        )
                    )
                if grid_mask[i, j + 1]:
                    faces.append(
                        (
                            uv2vertex_idx[i, j],
                            uv2vertex_idx[i + 1, j + 1],
                            uv2vertex_idx[i, j + 1],
                        )
                    )

        # TODO: this is a diy fix when there are no faces
        if len(faces) == 0:
            vertices = np.zeros((3, 3))
            faces = np.array([[0, 1, 2]]).astype(np.int64)
        else:
            vertices = np.stack(vertices, axis=0)
            faces = np.stack(faces, axis=0)

        return vertices, faces

    def meshify_all(
        self, use_grid_mask: bool = True, batch_idx: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_vertices = []
        all_faces = []
        all_indices = []
        sum_vertices = 0
        for face_idx in np.argwhere(~self.empty_mask[batch_idx]):
            vertices, faces = self.meshify(
                face_idx.item(), use_grid_mask=use_grid_mask, batch_idx=batch_idx
            )
            all_vertices.append(vertices)
            all_faces.append(faces + sum_vertices)
            all_indices.append(np.ones(len(all_vertices), dtype=np.int32))
            sum_vertices += len(vertices)
        return np.concat(all_vertices), np.concat(all_faces), np.concat(all_indices)


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
