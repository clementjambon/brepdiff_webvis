import time
from typing import List
import os
import glob

import numpy as np
import tyro
from tqdm.auto import tqdm
from pathlib import Path

import viser
import viser.extras
import viser.transforms as tf

from utils.uvgrid import UvGrid
from utils.trajectory_handler import TrajectoryHandler
from utils.vis_utils import get_cmap
from utils.trajectory_nodes import create_nodes, create_nodes_from_uvgrids


def main(
    data_path: str = "data/interp_0001.traj",
    recording_path: str = "recordings/recording.viser",
    share: bool = False,
    i_traj: int = 0,
    i_batch: int = 0,
    i_t: int = -1,
    framerate: int = 20,
    hold: float = 1.5,
    reps: int = 1,
    batch_stride: int = 1,
    inverted_order: bool = True,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading trajectories!")
    if os.path.isdir(data_path):
        all_uv_grids = sorted(glob.glob(os.path.join(data_path, "*.npz")))
        all_uv_grids = [UvGrid.deserialize(np.load(uvgrid)) for uvgrid in all_uv_grids]
    elif os.path.splitext(data_path)[1] == ".traj":
        print("Loading trajectories!")
        handler = TrajectoryHandler.load(data_path)
        trajectory = handler.trajectories[i_traj]
    else:
        raise ValueError()

    # Create serializer.
    serializer = server.get_scene_serializer()

    for i_rep in range(reps):

        if os.path.isdir(data_path):
            create_nodes_from_uvgrids(
                all_uv_grids, server, framerate=framerate, serializer=serializer
            )
        elif os.path.splitext(data_path)[1] == ".traj":
            create_nodes(
                trajectory,
                server,
                queried_i_batch=None if i_t >= 0 else i_batch,
                queried_t=i_t if i_t >= 0 else None,
                framerate=framerate,
                serializer=serializer,
                inverted_order=inverted_order,
                batch_stride=batch_stride,
            )
        serializer.insert_sleep(hold)

        inverted_order = not inverted_order

    # Save the complete animation.
    data = serializer.serialize()  # Returns bytes
    os.makedirs(os.path.dirname(recording_path), exist_ok=True)
    Path(recording_path).write_bytes(data)


if __name__ == "__main__":
    tyro.cli(main)
