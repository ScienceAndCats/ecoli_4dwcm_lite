from __future__ import annotations

import cupy as cp
import numpy as np

from spatial.config import FieldConfig
from spatial.grid import SpatialGrid
from spatial.species import SpeciesRegistry


class FieldState:
    """
    Spatial concentration storage for abundant species.

    Array layout:

        [field_species, active_voxel]

    Each value currently represents concentration in mM.

    Two buffers are maintained because diffusion later
    ping-pongs between:

        current state
        next state
    """

    def __init__(
        self,
        grid: SpatialGrid,
        registry: SpeciesRegistry,
        config: FieldConfig,
    ):

        self.grid = grid
        self.registry = registry
        self.config = config

        self.cell_voxels = (
            grid.cell_voxels
        )

        self.field_species_count = (
            registry.field_count
        )

        self.dtype = self._resolve_dtype(
            config.dtype
        )


        # ====================================================
        # Initial concentrations
        # ====================================================

        initial_cpu = np.zeros(
            (
                self.field_species_count,
                self.cell_voxels,
            ),
            dtype=np.float32,
        )


        for species in registry.field_species:

            concentration = float(
                species.initial_concentration_mM
                or 0.0
            )

            initial_cpu[
                species.field_id,
                :
            ] = concentration


        # ====================================================
        # Current GPU state
        # ====================================================

        self.values = cp.asarray(
            initial_cpu,
            dtype=self.dtype,
        )


        # ====================================================
        # Next-state diffusion buffer
        # ====================================================

        self.values_next = cp.zeros_like(
            self.values
        )


        # ====================================================
        # Per-field diffusion coefficients
        # ====================================================

        diffusion_cpu = np.zeros(
            self.field_species_count,
            dtype=np.float32,
        )

        for species in registry.field_species:

            diffusion_cpu[
                species.field_id
            ] = species.diffusion_um2_s


        self.diffusion_um2_s = cp.asarray(
            diffusion_cpu
        )


    # ========================================================
    # Data type
    # ========================================================

    @staticmethod
    def _resolve_dtype(dtype_name):

        if dtype_name == "float32":
            return cp.float32

        if dtype_name == "float64":
            return cp.float64

        raise ValueError(
            "Unsupported field dtype: "
            f"{dtype_name}. "
            "Use float32 or float64."
        )


    # ========================================================
    # Buffer management
    # ========================================================

    def clear_next(self):

        self.values_next.fill(0)


    def swap_buffers(self):

        (
            self.values,
            self.values_next,
        ) = (
            self.values_next,
            self.values,
        )


    # ========================================================
    # Species access
    # ========================================================

    def field_id(self, species_name):

        species = self.registry.get(
            species_name
        )

        if species.field_id is None:

            raise ValueError(
                f"{species_name} is not a field species."
            )

        return species.field_id


    def get_field(
        self,
        species_name,
    ):

        field_id = self.field_id(
            species_name
        )

        return self.values[
            field_id
        ]


    # ========================================================
    # Summaries
    # ========================================================

    @property
    def state_bytes(self):

        return (
            self.values.nbytes
            + self.values_next.nbytes
            + self.diffusion_um2_s.nbytes
        )


    @property
    def state_mib(self):

        return (
            self.state_bytes
            / 1024**2
        )


    def mean_concentration_mM(
        self,
        species_name,
    ):

        field = self.get_field(
            species_name
        )

        return float(
            cp.mean(
                field
            ).item()
        )


    # ========================================================
    # Validation
    # ========================================================

    def validate(self):

        expected_shape = (
            self.field_species_count,
            self.cell_voxels,
        )

        if (
            self.values.shape
            != expected_shape
        ):

            raise RuntimeError(
                "Field-state shape mismatch: "
                f"expected {expected_shape}, "
                f"found {self.values.shape}"
            )


        if bool(
            cp.any(
                self.values < 0
            ).item()
        ):

            raise RuntimeError(
                "Negative field concentration detected."
            )


        if bool(
            cp.any(
                ~cp.isfinite(
                    self.values
                )
            ).item()
        ):

            raise RuntimeError(
                "Non-finite field concentration detected."
            )


        return True


    # ========================================================
    # Human-readable summary
    # ========================================================

    def describe(self):

        print()
        print("CONCENTRATION FIELD STATE")
        print("--------------------------------------------")

        print(
            f"Field species:              "
            f"{self.field_species_count:,}"
        )

        print(
            f"Active voxels / field:      "
            f"{self.cell_voxels:,}"
        )

        print(
            f"Data type:                  "
            f"{self.values.dtype}"
        )

        print(
            f"Field buffer memory:        "
            f"{self.state_mib:.2f} MiB"
        )


        if self.field_species_count > 0:

            print()
            print("INITIAL FIELD CONCENTRATIONS")
            print("--------------------------------------------")

            for species in self.registry.field_species:

                print(
                    f"{species.field_id:5d}  "
                    f"{species.name:24s} "
                    f"{self.mean_concentration_mM(species.name):.3f} mM"
                )