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
from diffusion_save import main as main_single


def main(
    data_path: str,
    recording_path: str = "recordings/",
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
    subfolders = [
        os.path.join(data_path, name)
        for name in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, name))
    ]
    subnames = [os.path.basename(folder) for folder in subfolders]

    for subfolder, subname in zip(subfolders, subnames):
        main_single(
            data_path=subfolder,
            recording_path=os.path.join(recording_path, subname + ".viser"),
            share=share,
            i_traj=i_traj,
            i_batch=i_batch,
            i_t=i_t,
            framerate=framerate,
            hold=hold,
            reps=reps,
            batch_stride=batch_stride,
            inverted_order=inverted_order,
        )


if __name__ == "__main__":
    tyro.cli(main)
