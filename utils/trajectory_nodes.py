import time
from typing import List, Optional

import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.infra
import viser.transforms as tf

from utils.uvgrid import UvGrid
from utils.trajectory import Trajectory
from utils.vis_utils import get_cmap, DEFAULT_COLORMAP


def create_nodes_from_uvgrids(
    uvgrids: List[UvGrid],
    server: viser.ViserServer,
    framerate: int = 5,
    serializer: Optional[viser.infra.StateSerializer] = None,
    inverted_order: bool = True,
):

    point_nodes = []
    for t, uvgrid in tqdm(enumerate(uvgrids[:: 1 if inverted_order else -1])):

        # ==========
        # POINTS
        # ==========

        # Place the point cloud in the frame.
        uvgrid_coord = uvgrid.coord[~uvgrid.empty_mask]
        uvgrid_grid_mask = uvgrid.grid_mask[~uvgrid.empty_mask]
        valid_uvgrid_coord = uvgrid_coord[uvgrid_grid_mask]
        valid_uvgrid_colors = (
            get_cmap(uvgrid_coord.shape[0])[:, None, None, :]
            .repeat(uvgrid_coord.shape[1], axis=1)
            .repeat(uvgrid_coord.shape[2], axis=2)
        )
        valid_uvgrid_colors = valid_uvgrid_colors[uvgrid_grid_mask]
        points = valid_uvgrid_coord.reshape(-1, 3)
        point_nodes.append(
            server.scene.add_point_cloud(
                name=(
                    "point_cloud"
                    if serializer is not None
                    else f"/traj/{t}/point_cloud"
                ),
                points=points,
                colors=valid_uvgrid_colors,
                point_size=0.01,
                point_shape="circle",
            )
        )

        if serializer is not None:
            serializer.insert_sleep(1.0 / framerate)

    return [[node] for node in point_nodes]


def create_nodes(
    trajectory: Trajectory,
    server: viser.ViserServer,
    show_mesh: bool = False,
    queried_i_batch: Optional[int] = None,
    queried_t: Optional[int] = None,
    framerate: int = 5,
    serializer: Optional[viser.infra.StateSerializer] = None,
    inverted_order: bool = True,
):

    point_nodes: List[List[viser.PointCloudHandle]] = []
    mesh_nodes: List[List[viser.MeshHandle]] = []
    for t, uvgrid in tqdm(
        list(trajectory.traj_uv_grids.items())[:: -1 if inverted_order else 1]
    ):
        if queried_t is not None and t != queried_t:
            continue

        t_point_nodes = []
        t_mesh_nodes = []
        for i_batch in range(uvgrid.coord.shape[0]):

            if inverted_order:
                i_batch = uvgrid.coord.shape[0] - 1 - i_batch

            if queried_i_batch is not None and i_batch != queried_i_batch:
                continue

            # ==========
            # POINTS
            # ==========

            # Place the point cloud in the frame.
            uvgrid_coord = uvgrid.coord[i_batch][~uvgrid.empty_mask[i_batch]]
            uvgrid_grid_mask = uvgrid.grid_mask[i_batch][~uvgrid.empty_mask[i_batch]]
            valid_uvgrid_coord = uvgrid_coord[uvgrid_grid_mask]
            valid_uvgrid_colors = (
                get_cmap(uvgrid_coord.shape[0])[:, None, None, :]
                .repeat(uvgrid_coord.shape[1], axis=1)
                .repeat(uvgrid_coord.shape[2], axis=2)
            )
            valid_uvgrid_colors = valid_uvgrid_colors[uvgrid_grid_mask]
            points = valid_uvgrid_coord.reshape(-1, 3)
            t_point_nodes.append(
                server.scene.add_point_cloud(
                    name=(
                        "point_cloud"
                        if serializer is not None
                        else f"/traj/{i_batch}/{t}/point_cloud"
                    ),
                    points=points,
                    colors=valid_uvgrid_colors,
                    point_size=0.01,
                    point_shape="circle",
                )
            )

            # ==========
            # MESHES
            # ==========
            if False:
                vertices, faces, indices = uvgrid.meshify_all(
                    use_grid_mask=True, batch_idx=i_batch
                )
                # vertex_colors = DEFAULT_COLORMAP[indices]
                # vertices, faces = uvgrid.meshify(0, use_grid_mask=True, batch_idx=i_batch)
                t_mesh_nodes.append(
                    server.scene.add_mesh_simple(
                        name=(
                            "mesh"
                            if serializer is not None
                            else f"/traj/{i_batch}/{t}/mesh"
                        ),
                        vertices=vertices,
                        faces=faces,
                        # wireframe=True,
                        opacity=0.3,
                    )
                )

            if serializer is not None:
                serializer.insert_sleep(1.0 / framerate)

        point_nodes.append(t_point_nodes)
        mesh_nodes.append(t_mesh_nodes)

    return point_nodes, mesh_nodes
