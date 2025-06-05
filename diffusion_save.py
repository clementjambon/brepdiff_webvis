import time
from typing import List

import numpy as np
import tyro
from tqdm.auto import tqdm
from pathlib import Path

import viser
import viser.extras
import viser.transforms as tf

from utils.trajectory_handler import TrajectoryHandler
from utils.vis_utils import get_cmap
from utils.trajectory_nodes import create_nodes


def main(
    data_path: str = "data/interp_0001.traj",
    share: bool = False,
    i_traj: int = 0,
    i_batch: int = 0,
    framerate: int = 5,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading trajectories!")
    handler = TrajectoryHandler.load(data_path)
    trajectory = handler.trajectories[i_traj]

    # Create serializer.
    serializer = server.get_scene_serializer()

    create_nodes(
        trajectory,
        server,
        queried_i_batch=i_batch,
        framerate=framerate,
        serializer=serializer,
    )

    # Save the complete animation.
    data = serializer.serialize()  # Returns bytes
    Path("recordings/recording.viser").write_bytes(data)


if __name__ == "__main__":
    tyro.cli(main)
