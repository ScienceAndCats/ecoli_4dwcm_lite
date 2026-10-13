from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from spatial.config import (
    SpatialConfig,
)
from spatial.crowding import (
    CrowdingState,
)
from spatial.voxel_counts import (
    VoxelCountState,
)


class VoxelCountDiffusion:

    THREADS = 256


    def __init__(
        self,
        voxel_count_state: VoxelCountState,
        spatial_config: SpatialConfig,
        timestep_s: float,
        seed: int,
        crowding_state: CrowdingState | None = None,
        effective_volumes=None,
    ):

        self.state = (
            voxel_count_state
        )

        self.spatial = (
            spatial_config
        )

        self.timestep_s = float(
            timestep_s
        )

        self.seed = int(
            seed
        )

        self.step_count = 0


        if (
            self.timestep_s
            <= 0
        ):

            raise ValueError(
                "Voxel-count diffusion timestep "
                "must be > 0."
            )


        self.crowding_state = (
            crowding_state
        )


        if (
            crowding_state is not None
            and crowding_state.cell_voxels
            != self.state.cell_voxels
        ):

            raise ValueError(
                "Crowding and voxel-count states "
                "use different spatial grids."
            )


        self.effective_volumes = (
            effective_volumes
        )


        # ====================================================
        # Diffusion probabilities
        # ====================================================

        self.direction_probability_cpu = (
            self._build_direction_probabilities()
        )


        self.direction_probability = cp.asarray(
            self.direction_probability_cpu
        )


        # ====================================================
        # Effective molecular volumes
        # ====================================================

        if (
            self.effective_volumes
            is not None
        ):

            volume_lut = np.asarray(
                self.effective_volumes
                .voxel_count_volume_lut_nm3,

                dtype=np.float32,
            )


            if (
                volume_lut.shape
                != (
                    self.state.species_count,
                )
            ):

                raise ValueError(
                    "Voxel-count effective-volume lookup "
                    "does not match voxel-count species."
                )


            self.species_volume_nm3 = cp.asarray(
                volume_lut
            )


        else:

            self.species_volume_nm3 = cp.zeros(
                self.state.species_count,
                dtype=cp.float32,
            )


        # ====================================================
        # Diagnostics
        # ====================================================

        self.attempted_hops = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        self.actual_hops = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        self.crowding_rejections = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        self.capacity_rejections = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        # ====================================================
        # Per-step physical-volume reservations
        # ====================================================

        self.reserved_incoming_nm3 = cp.zeros(
            self.state.cell_voxels,
            dtype=cp.float32,
        )


        # ====================================================
        # Dummy arrays used when crowding/capacity are disabled
        # ====================================================

        self._dummy_accessibility = cp.ones(
            1,
            dtype=cp.float32,
        )


        self._dummy_occupied_volume = cp.zeros(
            1,
            dtype=cp.float32,
        )


        # ====================================================
        # CUDA kernel
        # ====================================================

        kernel_path = (
            Path(__file__).parent
            / "kernels"
            / "voxel_count_diffusion.cu"
        )


        module = cp.RawModule(
            code=kernel_path.read_text(),

            options=(
                "--std=c++11",
            ),
        )


        self._diffuse = (
            module.get_function(
                "diffuse_voxel_counts"
            )
        )


    # ========================================================
    # Diffusion probabilities
    # ========================================================

    def _build_direction_probabilities(
        self,
    ):

        probabilities = np.zeros(
            self.state.species_count,
            dtype=np.float32,
        )


        dx_squared = (
            self.spatial.voxel_size_um
            ** 2
        )


        for species in (
            self.state.registry
            .voxel_count_species
        ):

            q = (
                species.diffusion_um2_s
                * self.timestep_s
                / dx_squared
            )


            if (
                6.0 * q
                >
                self.spatial
                .max_total_hop_probability
                + 1e-6
            ):

                raise ValueError(
                    f"Voxel-count species "
                    f"{species.name} exceeds "
                    "maximum hop probability."
                )


            probabilities[
                species.voxel_count_id
            ] = q


        return probabilities


    # ========================================================
    # Launch helpers
    # ========================================================

    @classmethod
    def _blocks(
        cls,
        total,
    ):

        return (
            int(
                (
                    total
                    + cls.THREADS
                    - 1
                )
                // cls.THREADS
            ),
        )


    # ========================================================
    # One diffusion step
    # ========================================================

    def step(
        self,
    ):

        state = (
            self.state
        )


        total_entries = (
            state.species_count
            * state.cell_voxels
        )


        if (
            total_entries == 0
        ):

            self.step_count += 1

            return


        state.clear_next()


        self.reserved_incoming_nm3.fill(
            0
        )


        # ====================================================
        # Crowding inputs
        # ====================================================

        if (
            self.crowding_state
            is not None
        ):

            accessibility = (
                self.crowding_state
                .refresh_movement_accessibility()
            )


            occupied_volume = (
                self.crowding_state
                .total_volume_nm3
            )


            voxel_volume_nm3 = float(
                self.crowding_state
                .voxel_volume_nm3
            )


            use_crowding = 1


        else:

            accessibility = (
                self._dummy_accessibility
            )


            occupied_volume = (
                self._dummy_occupied_volume
            )


            voxel_volume_nm3 = 0.0

            use_crowding = 0


        # ====================================================
        # Hard capacity is active only when BOTH:
        #
        #     crowding state exists
        #     effective molecular volumes exist
        # ====================================================

        use_capacity = int(
            self.crowding_state
            is not None
            and self.effective_volumes
            is not None
        )


        self._diffuse(
            self._blocks(
                total_entries
            ),

            (
                self.THREADS,
            ),

            (
                state.counts,
                state.counts_next,

                state.grid
                .neighbors_gpu
                .reshape(
                    -1
                ),

                self.direction_probability,

                accessibility,

                self.species_volume_nm3,

                occupied_volume,

                self.reserved_incoming_nm3,

                self.attempted_hops,
                self.actual_hops,
                self.crowding_rejections,
                self.capacity_rejections,

                np.int64(
                    state.cell_voxels
                ),

                np.int64(
                    state.species_count
                ),

                np.int64(
                    total_entries
                ),

                np.float32(
                    voxel_volume_nm3
                ),

                np.int32(
                    use_crowding
                ),

                np.int32(
                    use_capacity
                ),

                np.uint64(
                    self.seed
                ),

                np.uint64(
                    self.step_count
                ),
            ),
        )


        state.swap_buffers()

        self.step_count += 1


    # ========================================================
    # Multiple steps
    # ========================================================

    def run_steps(
        self,
        steps: int,
    ):

        if steps < 0:

            raise ValueError(
                "steps must be >= 0"
            )


        for _ in range(
            steps
        ):

            self.step()


    # ========================================================
    # Diagnostics
    # ========================================================

    @property
    def attempted_hop_count(self):

        return int(
            self.attempted_hops[
                0
            ].item()
        )


    @property
    def actual_hop_count(self):

        return int(
            self.actual_hops[
                0
            ].item()
        )


    @property
    def crowding_rejection_count(self):

        return int(
            self.crowding_rejections[
                0
            ].item()
        )


    @property
    def capacity_rejection_count(self):

        return int(
            self.capacity_rejections[
                0
            ].item()
        )


    def validate(self):

        self.state.validate(
            require_initial_totals=False
        )


        if (
            self.crowding_state
            is not None
        ):

            self.crowding_state.validate()


        return True


    def describe(self):

        print()
        print("VOXEL-COUNT DIFFUSION")
        print("--------------------------------------------")


        print(
            f"Completed diffusion steps:  "
            f"{self.step_count:,}"
        )


        print(
            f"Diffusion timestep:         "
            f"{self.timestep_s * 1000:.4f} ms"
        )


        print(
            f"Attempted hops:             "
            f"{self.attempted_hop_count:,}"
        )


        print(
            f"Actual hops:                "
            f"{self.actual_hop_count:,}"
        )


        print(
            f"Crowding-rejected hops:     "
            f"{self.crowding_rejection_count:,}"
        )


        print(
            f"Physical-capacity rejects:  "
            f"{self.capacity_rejection_count:,}"
        )