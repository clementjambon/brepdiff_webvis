import time
from typing import List

import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf

from utils.trajectory_handler import TrajectoryHandler
from utils.vis_utils import get_cmap


def main(
    data_path: str = "data/interp_0001.traj",
    share: bool = False,
    i_traj: int = 0,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading trajectories!")
    handler = TrajectoryHandler.load(data_path)
    trajectory = handler.trajectories[i_traj]
    num_frames = trajectory.size
    # num_frames = min(max_frames, loader.num_frames())

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.02,
            step=1e-3,
            initial_value=0.01,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=trajectory.size - 1,
            step=1,
            initial_value=0,
        )
        gui_batch = server.gui.add_slider(
            "Batch",
            min=0,
            max=trajectory.batch_size - 1,
            step=1,
            initial_value=0,
        )

        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=30
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    prev_timestep = gui_timestep.value
    prev_batch = gui_batch.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        current_batch = gui_batch.value
        with server.atomic():
            # Toggle point visibility.
            point_nodes[current_timestep][current_batch].visible = True
            point_nodes[prev_timestep][current_batch].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Toggle frame visibility when the batch slider changes.
    @gui_batch.on_update
    def _(_) -> None:
        nonlocal prev_batch
        current_timestep = gui_timestep.value
        current_batch = gui_batch.value
        with server.atomic():
            # Toggle point visibility.
            point_nodes[current_timestep][current_batch].visible = True
            point_nodes[current_timestep][prev_batch].visible = False
        prev_batch = current_batch
        server.flush()  # Optional!

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    point_nodes: List[List[viser.PointCloudHandle]] = []
    for t, uvgrid in tqdm(trajectory.traj_uv_grids.items()):
        t_point_nodes = []
        for i_batch in range(uvgrid.coord.shape[0]):

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
                    name=f"/traj_{i_traj}/{i_batch}/{t}/point_cloud",
                    points=points,
                    colors=valid_uvgrid_colors,
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                )
            )
        point_nodes.append(t_point_nodes)

    # Hide all but the current pc.
    for i_t, t_point_nodes in enumerate(point_nodes):
        for i_batch, point_node in enumerate(t_point_nodes):
            point_node.visible = (i_t == gui_timestep.value) and (
                i_batch == gui_batch.value
            )

    # Playback update loop.
    prev_timestep = gui_timestep.value
    prev_batch = gui_batch.value
    while True:

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value][
            gui_batch.value
        ].point_size = gui_point_size.value


if __name__ == "__main__":
    tyro.cli(main)
