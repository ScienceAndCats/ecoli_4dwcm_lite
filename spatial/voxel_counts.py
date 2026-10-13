from __future__ import annotations

import cupy as cp
import numpy as np

from spatial.grid import SpatialGrid
from spatial.species import (
    Representation,
    SpeciesRegistry,
)


class VoxelCountState:
    """
    Dense integer spatial storage for abundant but discrete
    molecular species.

    Array layout:

        [voxel_count_species, active_voxel]

    Example:

        counts[ribosome_id, voxel_id] = 7

    means:

        seven free ribosomes currently occupy that 50-nm
        voxel.

    Individual molecules do NOT have persistent identities.

    A second buffer is maintained because stochastic diffusion
    will ping-pong between current and next state.
    """

    def __init__(
        self,
        grid: SpatialGrid,
        registry: SpeciesRegistry,
        seed: int,
    ):

        self.grid = grid

        self.registry = registry

        self.seed = int(
            seed
        )


        self.cell_voxels = (
            grid.cell_voxels
        )


        self.species_count = (
            registry.voxel_count_count
        )


        self.dtype = cp.uint32


        # ====================================================
        # Build initial integer distribution on CPU.
        #
        # For now initialization is spatially uniform.
        #
        # This is only the INITIAL condition. Once diffusion,
        # microenvironment effects, mRNA capture, and crowding
        # are active, the distribution can become strongly
        # nonuniform.
        # ====================================================

        initial_cpu = np.zeros(
            (
                self.species_count,
                self.cell_voxels,
            ),

            dtype=np.uint32,
        )


        rng = np.random.default_rng(
            self.seed
        )


        if self.cell_voxels <= 0:

            raise ValueError(
                "Voxel-count state requires at least "
                "one active cell voxel."
            )


        probabilities = np.full(
            self.cell_voxels,

            1.0
            / self.cell_voxels,

            dtype=np.float64,
        )


        for species in (
            registry.voxel_count_species
        ):

            count = int(
                species.initial_count
                or 0
            )


            if count == 0:

                continue


            distribution = (
                rng.multinomial(
                    count,
                    probabilities,
                )
            )


            maximum = int(
                distribution.max(
                    initial=0
                )
            )


            if (
                maximum
                > np.iinfo(
                    np.uint32
                ).max
            ):

                raise OverflowError(
                    f"{species.name}: voxel count exceeds "
                    "uint32 capacity."
                )


            initial_cpu[
                species.voxel_count_id,
                :
            ] = distribution.astype(
                np.uint32
            )


        # ====================================================
        # GPU state
        # ====================================================

        self.counts = cp.asarray(
            initial_cpu
        )


        # ====================================================
        # Next-state buffer
        #
        # Used by integer stochastic diffusion in the next
        # checklist step.
        # ====================================================

        self.counts_next = cp.zeros_like(
            self.counts
        )


        # ====================================================
        # Per-species diffusion coefficients
        # ====================================================

        diffusion_cpu = np.zeros(
            self.species_count,
            dtype=np.float32,
        )


        for species in (
            registry.voxel_count_species
        ):

            diffusion_cpu[
                species.voxel_count_id
            ] = (
                species.diffusion_um2_s
            )


        self.diffusion_um2_s = cp.asarray(
            diffusion_cpu
        )


        # ====================================================
        # Store exact initial totals for conservation checks.
        # ====================================================

        self.initial_counts = np.asarray(
            [
                int(
                    species.initial_count
                    or 0
                )

                for species
                in registry.voxel_count_species
            ],

            dtype=np.int64,
        )


    # ========================================================
    # Buffer management
    # ========================================================

    def clear_next(self):

        self.counts_next.fill(
            0
        )


    def swap_buffers(self):

        (
            self.counts,
            self.counts_next,
        ) = (
            self.counts_next,
            self.counts,
        )


    # ========================================================
    # Species access
    # ========================================================

    def voxel_count_id(
        self,
        species_name,
    ):

        species = (
            self.registry.get(
                species_name
            )
        )


        if (
            species.representation
            is not Representation.VOXEL_COUNT
        ):

            raise ValueError(
                f"{species_name} is not a "
                "voxel_count species."
            )


        if species.voxel_count_id is None:

            raise RuntimeError(
                f"{species_name} has no voxel-count ID."
            )


        return (
            species.voxel_count_id
        )


    def get_counts(
        self,
        species_name,
    ):

        species_id = (
            self.voxel_count_id(
                species_name
            )
        )


        return self.counts[
            species_id
        ]


    # ========================================================
    # Summaries
    # ========================================================

    def total_count(
        self,
        species_name,
    ):

        values = self.get_counts(
            species_name
        )


        return int(
            cp.sum(
                values,
                dtype=cp.uint64,
            ).item()
        )


    def mean_count_per_voxel(
        self,
        species_name,
    ):

        values = self.get_counts(
            species_name
        )


        return float(
            cp.mean(
                values,
                dtype=cp.float64,
            ).item()
        )


    @property
    def state_bytes(self):

        return (
            self.counts.nbytes
            + self.counts_next.nbytes
            + self.diffusion_um2_s.nbytes
        )


    @property
    def state_kib(self):

        return (
            self.state_bytes
            / 1024
        )


    # ========================================================
    # Validation
    # ========================================================

    def validate(
        self,
        require_initial_totals=True,
    ):

        expected_shape = (
            self.species_count,
            self.cell_voxels,
        )


        if (
            self.counts.shape
            != expected_shape
        ):

            raise RuntimeError(
                "Voxel-count state shape mismatch: "
                f"expected {expected_shape}, "
                f"found {self.counts.shape}"
            )


        if (
            self.counts.dtype
            != cp.uint32
        ):

            raise RuntimeError(
                "Voxel-count state must use uint32."
            )


        if require_initial_totals:

            for species in (
                self.registry.voxel_count_species
            ):

                observed = (
                    self.total_count(
                        species.name
                    )
                )


                expected = int(
                    species.initial_count
                    or 0
                )


                if observed != expected:

                    raise RuntimeError(
                        f"{species.name} count mismatch: "
                        f"expected {expected:,}, "
                        f"found {observed:,}."
                    )


        return True


    # ========================================================
    # Human-readable summary
    # ========================================================

    def describe(self):

        print()
        print("VOXEL-COUNT STATE")
        print("--------------------------------------------")


        print(
            f"Voxel-count species:        "
            f"{self.species_count:,}"
        )


        print(
            f"Active voxels / species:    "
            f"{self.cell_voxels:,}"
        )


        print(
            f"Data type:                  "
            f"{self.counts.dtype}"
        )


        print(
            f"State memory:               "
            f"{self.state_kib:.2f} KiB"
        )


        if self.species_count > 0:

            print()
            print("INITIAL VOXEL COUNTS")
            print("--------------------------------------------")


            for species in (
                self.registry.voxel_count_species
            ):

                print(
                    f"{species.voxel_count_id:5d}  "
                    f"{species.name:24s} "
                    f"total="
                    f"{self.total_count(species.name):<8,d} "
                    f"mean/voxel="
                    f"{self.mean_count_per_voxel(species.name):.3f}"
                )
