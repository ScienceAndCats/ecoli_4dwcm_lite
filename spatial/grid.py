from dataclasses import dataclass

import cupy as cp
import numpy as np

from geometry.cell import build_spherocylinder_mask
from spatial.config import SpatialConfig


@dataclass
class SpatialGrid:
    shape: tuple
    mask: np.ndarray

    active_flat: np.ndarray
    flat_to_active: np.ndarray

    neighbors_cpu: np.ndarray
    neighbors_gpu: cp.ndarray

    voxel_size_um: float
    voxel_volume_um3: float

    @classmethod
    def from_config(
        cls,
        config: SpatialConfig,
    ):

        mask, _ = build_spherocylinder_mask(
            voxel_size_um=config.voxel_size_um,
            cell_length_um=config.cell_length_um,
            cell_diameter_um=config.cell_diameter_um,
        )

        shape = mask.shape

        mask_flat = mask.reshape(-1)

        active_flat = np.flatnonzero(
            mask_flat
        ).astype(np.int32)

        cell_voxels = active_flat.size

        flat_to_active = np.full(
            mask_flat.size,
            -1,
            dtype=np.int32,
        )

        flat_to_active[
            active_flat
        ] = np.arange(
            cell_voxels,
            dtype=np.int32,
        )

        neighbors_cpu = cls._build_neighbors(
            shape=shape,
            active_flat=active_flat,
            flat_to_active=flat_to_active,
        )

        neighbors_gpu = cp.asarray(
            neighbors_cpu
        )

        return cls(
            shape=shape,
            mask=mask,

            active_flat=active_flat,
            flat_to_active=flat_to_active,

            neighbors_cpu=neighbors_cpu,
            neighbors_gpu=neighbors_gpu,

            voxel_size_um=config.voxel_size_um,
            voxel_volume_um3=config.voxel_volume_um3,
        )


    @staticmethod
    def _build_neighbors(
        shape,
        active_flat,
        flat_to_active,
    ):

        nx, ny, nz = shape

        plane = ny * nz

        neighbors = np.empty(
            (
                len(active_flat),
                6,
            ),
            dtype=np.int32,
        )

        for active_id, flat_id in enumerate(
            active_flat
        ):

            xi = flat_id // plane

            remainder = (
                flat_id
                - xi * plane
            )

            yi = remainder // nz
            zi = remainder - yi * nz

            candidates = [
                flat_id + plane if xi + 1 < nx else flat_id,
                flat_id - plane if xi > 0 else flat_id,

                flat_id + nz if yi + 1 < ny else flat_id,
                flat_id - nz if yi > 0 else flat_id,

                flat_id + 1 if zi + 1 < nz else flat_id,
                flat_id - 1 if zi > 0 else flat_id,
            ]

            for direction, candidate in enumerate(
                candidates
            ):

                destination = (
                    flat_to_active[candidate]
                )

                if destination >= 0:

                    neighbors[
                        active_id,
                        direction
                    ] = destination

                else:

                    # Reflecting boundary.
                    neighbors[
                        active_id,
                        direction
                    ] = active_id

        return neighbors


    @property
    def cell_voxels(self):
        return len(
            self.active_flat
        )


    @property
    def bounding_voxels(self):
        return int(
            np.prod(
                self.shape
            )
        )


    @property
    def cell_volume_um3(self):
        return (
            self.cell_voxels
            * self.voxel_volume_um3
        )
