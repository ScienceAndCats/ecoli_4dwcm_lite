from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from spatial.config import SpatialConfig
from spatial.crowding import CrowdingState
from spatial.particles import ParticleState


class ParticleDiffusion:

    THREADS = 256


    def __init__(
        self,
        particle_state: ParticleState,
        spatial_config: SpatialConfig,
        base_dt_s: float,
        seed: int,
        crowding_state: CrowdingState | None = None,
        effective_volumes=None,
    ):

        self.state = particle_state

        self.spatial = spatial_config

        self.base_dt_s = float(
            base_dt_s
        )

        self.seed = int(
            seed
        )

        self.step_count = 0


        self.crowding_state = (
            crowding_state
        )


        if (
            crowding_state is not None
            and crowding_state.cell_voxels
            != particle_state.cell_voxels
        ):

            raise ValueError(
                "Crowding and particle states "
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
        # Particle effective-volume lookup
        #
        # ID 0 is the empty slot and therefore volume zero.
        # ====================================================

        if (
            effective_volumes
            is not None
        ):

            volume_lut = np.asarray(
                effective_volumes
                .particle_volume_lut_nm3,

                dtype=np.float32,
            )


            expected = (
                self.state.registry
                .particle_count
                + 1
            )


            if (
                volume_lut.shape
                != (
                    expected,
                )
            ):

                raise ValueError(
                    "Particle effective-volume lookup "
                    "does not match particle registry."
                )


            self.species_volume_nm3 = cp.asarray(
                volume_lut
            )


        else:

            self.species_volume_nm3 = cp.zeros(
                self.state.registry
                .particle_count
                + 1,

                dtype=cp.float32,
            )


        # ====================================================
        # Diagnostics
        # ====================================================

        self.spill_writes = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        self.slot_rejected_moves = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        self.hard_overflow = cp.zeros(
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
        # Physical-volume reservation buffer
        # ====================================================

        self.reserved_incoming_nm3 = cp.zeros(
            self.state.cell_voxels,
            dtype=cp.float32,
        )


        self._dummy_accessibility = cp.ones(
            1,
            dtype=cp.float32,
        )


        self._dummy_occupied_volume = cp.zeros(
            1,
            dtype=cp.float32,
        )


        # ====================================================
        # CUDA
        # ====================================================

        kernel_path = (
            Path(__file__).parent
            / "kernels"
            / "particle_diffusion.cu"
        )


        module = cp.RawModule(
            code=kernel_path.read_text(),

            options=(
                "--std=c++11",
            ),
        )


        self._diffuse_primary = (
            module.get_function(
                "diffuse_primary"
            )
        )


        self._diffuse_spill = (
            module.get_function(
                "diffuse_spill"
            )
        )


    # ========================================================
    # Diffusion probabilities
    # ========================================================

    def _build_direction_probabilities(
        self,
    ):

        registry = (
            self.state.registry
        )


        probabilities = np.zeros(
            registry.particle_count + 1,
            dtype=np.float32,
        )


        dx_squared = (
            self.spatial.voxel_size_um
            ** 2
        )


        for species in (
            registry.particle_species
        ):

            q = (
                species.diffusion_um2_s
                * self.base_dt_s
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
                    f"Particle species "
                    f"{species.name} exceeds "
                    "maximum hop probability."
                )


            probabilities[
                species.particle_id
            ] = q


        return probabilities


    # ========================================================
    # Launch helper
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


        state.clear_next()


        self.reserved_incoming_nm3.fill(
            0
        )


        # ====================================================
        # Crowding data
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


        use_capacity = int(
            self.crowding_state
            is not None
            and self.effective_volumes
            is not None
        )


        primary_total = (
            state.cell_voxels
            * state.primary_slots
        )


        spill_total = (
            state.cell_voxels
            * state.spill_slots
        )


        # ====================================================
        # Primary
        # ====================================================

        self._diffuse_primary(
            self._blocks(
                primary_total
            ),

            (
                self.THREADS,
            ),

            (
                state.primary,
                state.primary_occupancy,

                state.primary_next,
                state.primary_occupancy_next,

                state.spill_next,
                state.spill_occupancy_next,

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

                self.spill_writes,
                self.slot_rejected_moves,
                self.hard_overflow,
                self.crowding_rejections,
                self.capacity_rejections,

                np.int32(
                    state.primary_slots
                ),

                np.int32(
                    state.spill_slots
                ),

                np.int64(
                    state.cell_voxels
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


        # ====================================================
        # Spill
        # ====================================================

        if (
            state.spill_slots > 0
        ):

            self._diffuse_spill(
                self._blocks(
                    spill_total
                ),

                (
                    self.THREADS,
                ),

                (
                    state.spill,
                    state.spill_occupancy,

                    state.primary_next,
                    state.primary_occupancy_next,

                    state.spill_next,
                    state.spill_occupancy_next,

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

                    self.spill_writes,
                    self.slot_rejected_moves,
                    self.hard_overflow,
                    self.crowding_rejections,
                    self.capacity_rejections,

                    np.int32(
                        state.primary_slots
                    ),

                    np.int32(
                        state.spill_slots
                    ),

                    np.int64(
                        state.cell_voxels
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
    def spill_write_count(self):

        return int(
            self.spill_writes[
                0
            ].item()
        )


    @property
    def rejected_move_count(self):

        return int(
            self.slot_rejected_moves[
                0
            ].item()
        )


    @property
    def hard_overflow_count(self):

        return int(
            self.hard_overflow[
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


    # ========================================================
    # Validation
    # ========================================================

    def validate(self):

        self.state.validate()


        if (
            self.hard_overflow_count
            != 0
        ):

            raise RuntimeError(
                "Particle diffusion experienced "
                f"{self.hard_overflow_count:,} "
                "hard overflow events."
            )


        if (
            self.crowding_state
            is not None
        ):

            self.crowding_state.validate()


        return True


    # ========================================================
    # Summary
    # ========================================================

    def describe(self):

        print()
        print("PARTICLE DIFFUSION")
        print("--------------------------------------------")


        print(
            f"Completed diffusion steps:  "
            f"{self.step_count:,}"
        )


        print(
            f"Diffusion timestep:         "
            f"{self.base_dt_s * 1e6:.2f} us"
        )


        print(
            f"Spill writes:               "
            f"{self.spill_write_count:,}"
        )


        print(
            f"Slot-capacity rejects:      "
            f"{self.rejected_move_count:,}"
        )


        print(
            f"Crowding-rejected moves:    "
            f"{self.crowding_rejection_count:,}"
        )


        print(
            f"Physical-capacity rejects:  "
            f"{self.capacity_rejection_count:,}"
        )


        print(
            f"Hard overflow events:       "
            f"{self.hard_overflow_count:,}"
        )