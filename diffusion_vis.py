import time
from typing import List

import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.extras
import viser.transforms as tf

from utils.trajectory_handler import TrajectoryHandler
from utils.vis_utils import get_cmap, DEFAULT_COLORMAP
from utils.trajectory_nodes import create_nodes


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
            # Toggle mesh visibility.
            mesh_nodes[current_timestep][current_batch].visible = True
            mesh_nodes[prev_timestep][current_batch].visible = False
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
            # Toggle mesh visibility.
            mesh_nodes[current_timestep][current_batch].visible = True
            mesh_nodes[current_timestep][prev_batch].visible = False

        prev_batch = current_batch
        server.flush()  # Optional!

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    point_nodes, mesh_nodes = create_nodes(trajectory, server, show_mesh=True)

    # Hide all but the current pc.
    for i_t, (t_point_nodes, t_mesh_nodes) in enumerate(zip(point_nodes, mesh_nodes)):
        for i_batch, (point_node, mesh_node) in enumerate(
            zip(t_point_nodes, t_mesh_nodes)
        ):
            point_node.visible = (i_t == gui_timestep.value) and (
                i_batch == gui_batch.value
            )
            # point_node.visible = False
            mesh_node.visible = (i_t == gui_timestep.value) and (
                i_batch == gui_batch.value
            )

    # Playback update loop.
    prev_timestep = gui_timestep.value
    prev_batch = gui_batch.value
    while True:

        # Update point size of this timestep
        point_nodes[gui_timestep.value][
            gui_batch.value
        ].point_size = gui_point_size.value


if __name__ == "__main__":
    tyro.cli(main)
