from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from spatial.config import SpatialConfig
from spatial.fields import FieldState


class FieldDiffusion:
    """
    GPU diffusion controller for concentration fields.

    Each field species has its own diffusion coefficient.
    """

    THREADS = 256


    def __init__(
        self,
        field_state: FieldState,
        spatial_config: SpatialConfig,
        timestep_s: float,
    ):

        self.state = field_state
        self.spatial = spatial_config

        self.timestep_s = float(
            timestep_s
        )

        self.step_count = 0


        # ====================================================
        # Build per-field diffusion alpha
        #
        # alpha = D * dt / dx^2
        # ====================================================

        self.alpha_cpu = (
            self._build_alpha()
        )

        self.alpha = cp.asarray(
            self.alpha_cpu
        )


        # ====================================================
        # Compile CUDA kernel
        # ====================================================

        kernel_path = (
            Path(__file__).parent
            / "kernels"
            / "field_diffusion.cu"
        )

        source = kernel_path.read_text()

        module = cp.RawModule(
            code=source,
            options=(
                "--std=c++11",
            ),
        )

        self._diffuse_fields = (
            module.get_function(
                "diffuse_fields"
            )
        )


    # ========================================================
    # Diffusion coefficients
    # ========================================================

    def _build_alpha(self):

        registry = self.state.registry

        alpha = np.zeros(
            registry.field_count,
            dtype=np.float32,
        )

        dx_squared = (
            self.spatial.voxel_size_um
            ** 2
        )


        for species in registry.field_species:

            value = (
                species.diffusion_um2_s
                * self.timestep_s
                / dx_squared
            )


            total_fraction = (
                6.0 * value
            )


            if (
                total_fraction
                >
                self.spatial.max_total_hop_probability
                + 1e-6
            ):

                raise ValueError(
                    f"Field species {species.name} "
                    f"has diffusion fraction "
                    f"{total_fraction:.6f}, exceeding "
                    f"configured maximum "
                    f"{self.spatial.max_total_hop_probability:.6f}."
                )


            alpha[
                species.field_id
            ] = value


        return alpha


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
    # One diffusion step
    # ========================================================

    def step(self):

        state = self.state

        total = (
            state.field_species_count
            * state.cell_voxels
        )


        if total == 0:
            self.step_count += 1
            return


        self._diffuse_fields(
            self._blocks(
                total
            ),

            (
                self.THREADS,
            ),

            (
                state.values,
                state.values_next,

                state.grid.neighbors_gpu.reshape(
                    -1
                ),

                self.alpha,

                np.int64(
                    state.cell_voxels
                ),

                np.int64(
                    state.field_species_count
                ),

                np.int64(
                    total
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
    # Validation
    # ========================================================

    def validate(self):

        self.state.validate()

        return True


    # ========================================================
    # Diagnostics
    # ========================================================

    def describe(self):

        print()
        print("FIELD DIFFUSION")
        print("--------------------------------------------")

        print(
            f"Completed diffusion steps:  "
            f"{self.step_count:,}"
        )

        print(
            f"Diffusion timestep:         "
            f"{self.timestep_s * 1e6:.2f} us"
        )

        print(
            f"Field species:              "
            f"{self.state.field_species_count:,}"
        )
