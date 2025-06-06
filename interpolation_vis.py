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
    exclude: List[int] = [],
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading trajectories!")
    handler = TrajectoryHandler.load(data_path)
    trajectory = handler.trajectories[i_traj]

    # Add playback UI.
    with server.gui.add_folder("Playback"):
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

    prev_batch = gui_batch.value

    # Toggle frame visibility when the batch slider changes.
    @gui_batch.on_update
    def _(_) -> None:
        nonlocal prev_batch
        current_batch = gui_batch.value
        with server.atomic():
            # Toggle mesh visibility.
            if mesh_nodes[current_batch] is not None:
                mesh_nodes[current_batch].visible = True

            if mesh_nodes[prev_batch] is not None:
                mesh_nodes[prev_batch].visible = False

        prev_batch = current_batch
        server.flush()  # Optional!

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    # Create nodes
    mesh_nodes = []
    uvgrid = trajectory.traj_uv_grids[0]
    for i_batch in range(uvgrid.coord.shape[0]):
        if (
            trajectory.brep_mesh_vertices[i_batch] is not None
            and trajectory.brep_mesh_faces[i_batch] is not None
            and not i_batch in exclude
        ):
            mesh_nodes.append(
                server.scene.add_mesh_simple(
                    name=(f"/traj/{i_batch}/mesh"),
                    vertices=trajectory.brep_mesh_vertices[i_batch],
                    faces=trajectory.brep_mesh_faces[i_batch],
                    flat_shading=True,
                )
            )
        else:
            mesh_nodes.append(None)

    # Hide all but the current pc.
    for i_batch, mesh_node in enumerate(mesh_nodes):
        if mesh_node is not None:
            mesh_node.visible = i_batch == gui_batch.value

    # Playback update loop.
    prev_batch = gui_batch.value
    while True:

        pass


if __name__ == "__main__":
    tyro.cli(main)
