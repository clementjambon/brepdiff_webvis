import numpy as np

# Create color map for different faces
DEFAULT_COLORMAP = np.array(
    [
        [0.12, 0.47, 0.71],  # blue
        [0.20, 0.63, 0.17],  # green
        [0.89, 0.10, 0.11],  # red
        [0.84, 0.49, 0.00],  # orange
        [0.58, 0.40, 0.74],  # purple
        [0.55, 0.34, 0.29],  # brown
        [0.75, 0.75, 0.00],  # yellow
        [0.70, 0.70, 0.70],  # gray
    ]
)


def get_cmap(n, cmap: np.ndarray = DEFAULT_COLORMAP):
    N = cmap.shape[0]
    # create an index array of length M that wraps around 0…N–1
    idx = np.arange(n) % N
    return cmap[idx]
