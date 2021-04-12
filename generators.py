import numpy as np
from dataclasses import dataclass, field
from typing import List
from numpy.typing import ArrayLike
import random
from scipy.spatial.transform import Rotation
from scipy.spatial import KDTree

@dataclass()
class Cube:
    position: ArrayLike
    orientation: ArrayLike
    color: ArrayLike
    name: str = "Cube"

@dataclass()
class Environment:
    cubes: List[Cube]
    control_points: ArrayLike
    kdtree: KDTree = field(init=False)

    def __post_init__(self):
        self.kdtree = KDTree(self.control_points)



def sample_positions(num_samples, *, max_samples=1000, min_distance=0.1):
    """Rejection sample a set of cube positions

    The positions are all at least ``min_distance`` apart. With the default
    setting this means .05x.05x.05 cubes can't touch each other.
    """

    if num_samples == 0:
        return np.array([])

    center = np.array((0.4, 0, 0.025))
    half_extent = np.array((0.25, 0.5, 0))

    accepted_samples = np.empty((num_samples, 3))
    accepted_samples[0] = center + np.random.rand(3) * 2 * half_extent - half_extent
    num_accepted = 1

    samples_per_step = 3
    samples_generated = 0
    while num_accepted < num_samples and samples_generated < max_samples:
        samples = (
            center[:, None]
            + np.random.rand(3, samples_per_step) * 2 * half_extent[:, None]
            - half_extent[:, None]
        )
        samples_generated += samples_per_step

        while samples.shape[1] > 0:
            actual_samples = accepted_samples[:num_accepted]
            # remove samples closer than 0.1 to any accepted pos
            distance = np.linalg.norm(
                samples[None, ...] - actual_samples[..., None], axis=1
            )
            distance_ok = np.all(distance >= min_distance, axis=0)
            samples = samples[:, distance_ok]

            # add the first stample if any are left
            if samples.shape[1] > 0:
                accepted_samples[num_accepted] = samples[:, 0]
                num_accepted += 1

                if num_accepted >= num_samples:
                    break

    return accepted_samples


def generate_environment(num_cubes=5, num_control=1000) -> Environment:
    """Generate a legibility environment

    Parameters
    ----------
    num_cubes:
        The number of cubes in the world
    num_control:
        The number of control points for trajectory generation
    """

    cube_positions = sample_positions(num_cubes)
    angles_deg = np.random.rand(num_cubes) * np.pi / 2 - np.pi / 4
    angles = Rotation.from_euler("z", angles_deg).as_quat()[:, [3, 0, 1, 2]]

    colors = [
        ".6 .75 .3 1",
        ".25 .56 .99 1",
        ".43 .84 .85 1",
        ".98 .43 .27 1",
        ".94 .82 .79 1",
    ]

    cubes = list()
    for idx in range(num_cubes):
        cubes.append(
            Cube(
                position=cube_positions[idx],
                orientation=angles[idx],
                color=colors[idx % len(colors)],
            )
        )

    # define control points for trajectory sampling
    center = np.array((0.4, 0, 0.25))
    half_extent = np.array((0.25, 0.5, 0.15))
    control_points = center + np.random.rand(1000, 3) * 2 * half_extent - half_extent

    return Environment(cubes=cubes, control_points=control_points)


def sample_trajectory(start, end, env, *, num_control=0):
    """Generate a random trajectory in the environment"""

    if num_control == 0:
        return np.stack((start, end), axis=0)

    indices = np.random.choice(len(env.control_points), size=(num_control), replace=False)
    remaining_samples = env.control_points[indices]

    ordered_points = [start]
    current_point = start
    for _ in range(num_control):
        idx = np.argmin(np.linalg.norm(remaining_samples - current_point,axis=-1), axis=0)
        current_point = remaining_samples[idx]
        ordered_points.append(current_point)

        others = np.ones(remaining_samples.shape[0], dtype=bool)
        others[idx] = False
        remaining_samples = remaining_samples[others]
    ordered_points.append(end)

    return np.stack(ordered_points, axis=0)