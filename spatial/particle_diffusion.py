from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from spatial.config import SpatialConfig
from spatial.particles import ParticleState


class ParticleDiffusion:
    """
    GPU particle-diffusion controller.

    Each particle species has its own diffusion coefficient.

    The global base timestep is supplied by SimulationEngine
    and is constrained by the fastest species in the loaded
    biological model.
    """

    THREADS = 256


    def __init__(
        self,
        particle_state: ParticleState,
        spatial_config: SpatialConfig,
        base_dt_s: float,
        seed: int,
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


        # ====================================================
        # Build per-species hop probabilities
        # ====================================================

        self.direction_probability_cpu = (
            self._build_direction_probabilities()
        )

        self.direction_probability = cp.asarray(
            self.direction_probability_cpu
        )


        # ====================================================
        # Runtime counters
        # ====================================================

        self.spill_writes = cp.zeros(
            1,
            dtype=cp.uint64,
        )

        self.rejected_moves = cp.zeros(
            1,
            dtype=cp.uint64,
        )

        self.hard_overflow = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        # ====================================================
        # Compile/load CUDA kernels
        # ====================================================

        kernel_path = (
            Path(__file__).parent
            / "kernels"
            / "particle_diffusion.cu"
        )

        source = kernel_path.read_text()

        module = cp.RawModule(
            code=source,
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
    # Per-species probabilities
    # ========================================================

    def _build_direction_probabilities(
        self,
    ):

        registry = self.state.registry

        probabilities = np.zeros(
            registry.particle_count + 1,
            dtype=np.float32,
        )


        dx_um = (
            self.spatial.voxel_size_um
        )

        dx_squared = (
            dx_um ** 2
        )


        for species in registry.particle_species:

            q = (
                species.diffusion_um2_s
                * self.base_dt_s
                / dx_squared
            )


            total_probability = (
                6.0 * q
            )


            if (
                total_probability
                >
                self.spatial.max_total_hop_probability
                + 1e-6
            ):

                raise ValueError(
                    f"Particle species {species.name} "
                    f"has total hop probability "
                    f"{total_probability:.6f}, exceeding "
                    f"configured maximum "
                    f"{self.spatial.max_total_hop_probability:.6f}."
                )


            probabilities[
                species.particle_id
            ] = q


        return probabilities


    # ========================================================
    # Kernel launch helpers
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
    # One diffusion timestep
    # ========================================================

    def step(self):

        state = self.state


        # Destination arrays must start empty.

        state.clear_next()


        primary_total = (
            state.cell_voxels
            * state.primary_slots
        )

        spill_total = (
            state.cell_voxels
            * state.spill_slots
        )


        # ====================================================
        # Diffuse primary particles
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

                state.grid.neighbors_gpu.reshape(-1),

                self.direction_probability,

                self.spill_writes,
                self.rejected_moves,
                self.hard_overflow,

                np.int32(
                    state.primary_slots
                ),

                np.int32(
                    state.spill_slots
                ),

                np.int64(
                    state.cell_voxels
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
        # Diffuse spill particles
        # ====================================================

        if state.spill_slots > 0:

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

                    state.grid.neighbors_gpu.reshape(-1),

                    self.direction_probability,

                    self.spill_writes,
                    self.rejected_moves,
                    self.hard_overflow,

                    np.int32(
                        state.primary_slots
                    ),

                    np.int32(
                        state.spill_slots
                    ),

                    np.int64(
                        state.cell_voxels
                    ),

                    np.uint64(
                        self.seed
                    ),

                    np.uint64(
                        self.step_count
                    ),
                ),
            )


        # New state becomes current state.

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
            self.rejected_moves[
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

        return True


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
            f"Rejected full-voxel moves:  "
            f"{self.rejected_move_count:,}"
        )

        print(
            f"Hard overflow events:       "
            f"{self.hard_overflow_count:,}"
        )
