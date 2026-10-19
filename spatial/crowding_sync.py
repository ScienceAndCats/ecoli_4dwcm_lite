from __future__ import annotations

import cupy as cp
import numpy as np


class DynamicCrowdingSynchronizer:
    """
    Keep dynamic molecular occupancy synchronized with the
    actual molecular state of the simulation.

    Currently supported directly:

        ParticleState
        VoxelCountState

    Future states such as:

        mRNA
        transcription complexes
        translation complexes

    can contribute through registered volume providers.


    Conceptually:

        molecular state
            ↓
        effective molecular volumes
            ↓
        volume per voxel
            ↓
        crowding.dynamic_volume_nm3
    """

    def __init__(
        self,
        crowding_state,
        effective_volumes,
        particle_state=None,
        voxel_count_state=None,
    ):

        self.crowding_state = (
            crowding_state
        )

        self.effective_volumes = (
            effective_volumes
        )

        self.particle_state = (
            particle_state
        )

        self.voxel_count_state = (
            voxel_count_state
        )


        self.cell_voxels = (
            crowding_state.cell_voxels
        )


        # ====================================================
        # Validate grids
        # ====================================================

        if (
            particle_state is not None
            and particle_state.cell_voxels
            != self.cell_voxels
        ):

            raise ValueError(
                "Particle state and crowding state "
                "use different spatial grids."
            )


        if (
            voxel_count_state is not None
            and voxel_count_state.cell_voxels
            != self.cell_voxels
        ):

            raise ValueError(
                "Voxel-count state and crowding state "
                "use different spatial grids."
            )


        # ====================================================
        # Validate species registry identity.
        #
        # Numeric IDs in the effective-volume lookup tables
        # must refer to the exact same registry used by the
        # molecular states.
        # ====================================================

        volume_registry = (
            effective_volumes
            .species_registry
        )


        if (
            particle_state is not None
            and particle_state.registry
            is not volume_registry
        ):

            raise ValueError(
                "Particle state and effective-volume "
                "registry do not share the same "
                "SpeciesRegistry."
            )


        if (
            voxel_count_state is not None
            and voxel_count_state.registry
            is not volume_registry
        ):

            raise ValueError(
                "Voxel-count state and effective-volume "
                "registry do not share the same "
                "SpeciesRegistry."
            )


        # ====================================================
        # Require explicit effective volumes for every
        # discrete species represented by the supplied states.
        #
        # This avoids silently treating a molecule as having
        # zero volume because its configuration was forgotten.
        # ====================================================

        if particle_state is not None:

            for species in (
                particle_state.registry
                .particle_species
            ):

                effective_volumes.species_volume_nm3(
                    species.name
                )


        if voxel_count_state is not None:

            for species in (
                voxel_count_state.registry
                .voxel_count_species
            ):

                effective_volumes.species_volume_nm3(
                    species.name
                )


        # ====================================================
        # GPU lookup tables
        # ====================================================

        self.particle_volume_lut_nm3 = cp.asarray(
            np.asarray(
                effective_volumes
                .particle_volume_lut_nm3,

                dtype=np.float32,
            )
        )


        self.voxel_count_volume_lut_nm3 = cp.asarray(
            np.asarray(
                effective_volumes
                .voxel_count_volume_lut_nm3,

                dtype=np.float32,
            )
        )


        # ====================================================
        # Scratch array used when rebuilding occupancy.
        # ====================================================

        self._scratch_nm3 = cp.zeros(
            self.cell_voxels,
            dtype=cp.float32,
        )


        # ====================================================
        # Extra dynamic-volume providers.
        #
        # Each provider is a callable returning:
        #
        #     float32[cell_voxels]
        #
        # Later an mRNA state or translation-complex state
        # can provide its own occupancy this way.
        # ====================================================

        self._extra_providers = {}


        self.rebuild_count = 0


        # Build the correct initial crowding map immediately.

        self.rebuild_dynamic()


    # ========================================================
    # Extra provider registration
    # ========================================================

    def register_volume_provider(
        self,
        name,
        provider,
    ):

        name = str(
            name
        )


        if name in self._extra_providers:

            raise ValueError(
                f"Duplicate crowding-volume provider: "
                f"{name}"
            )


        if not callable(
            provider
        ):

            raise TypeError(
                "Crowding-volume provider must be callable."
            )


        self._extra_providers[
            name
        ] = provider


        # Keep the map immediately synchronized.

        self.rebuild_dynamic()


    def remove_volume_provider(
        self,
        name,
    ):

        del self._extra_providers[
            name
        ]


        self.rebuild_dynamic()


    # ========================================================
    # Particle contribution
    # ========================================================

    def _add_particles(
        self,
        target,
    ):

        state = (
            self.particle_state
        )


        if state is None:

            return


        # Empty particle slots contain species ID 0.
        #
        # LUT[0] is zero volume, so every slot can safely be
        # included without constructing occupancy masks.

        primary_volumes = (
            self.particle_volume_lut_nm3[
                state.primary
            ]
        )


        target += cp.sum(
            primary_volumes,
            axis=1,
            dtype=cp.float32,
        )


        if state.spill_slots > 0:

            spill_volumes = (
                self.particle_volume_lut_nm3[
                    state.spill
                ]
            )


            target += cp.sum(
                spill_volumes,
                axis=1,
                dtype=cp.float32,
            )


    # ========================================================
    # Voxel-count contribution
    # ========================================================

    def _add_voxel_counts(
        self,
        target,
    ):

        state = (
            self.voxel_count_state
        )


        if (
            state is None
            or state.species_count == 0
        ):

            return


        # counts:
        #
        #     [species, voxel]
        #
        # volumes:
        #
        #     [species]
        #
        # Therefore:
        #
        # sum(
        #     molecule_count
        #     * volume_per_molecule
        # )
        #
        # gives occupied volume in each voxel.

        counts_float = (
            state.counts.astype(
                cp.float32
            )
        )


        contribution = cp.sum(
            counts_float
            * self.voxel_count_volume_lut_nm3[
                :,
                None,
            ],

            axis=0,

            dtype=cp.float32,
        )


        target += contribution


    # ========================================================
    # Additional future contributors
    # ========================================================

    def _add_extra_providers(
        self,
        target,
    ):

        for (
            name,
            provider,
        ) in self._extra_providers.items():

            values = cp.asarray(
                provider(),
                dtype=cp.float32,
            )


            if (
                values.shape
                != (
                    self.cell_voxels,
                )
            ):

                raise ValueError(
                    f"Crowding-volume provider "
                    f"{name} returned shape "
                    f"{values.shape}; expected "
                    f"({self.cell_voxels},)."
                )


            if bool(
                cp.any(
                    ~cp.isfinite(
                        values
                    )
                ).item()
            ):

                raise ValueError(
                    f"Crowding-volume provider "
                    f"{name} returned non-finite values."
                )


            if bool(
                cp.any(
                    values < 0
                ).item()
            ):

                raise ValueError(
                    f"Crowding-volume provider "
                    f"{name} returned negative volume."
                )


            target += values


    # ========================================================
    # Calculate occupancy from current molecular states
    # ========================================================

    def _calculate_dynamic_volume(
        self,
        target,
    ):

        target.fill(
            0
        )


        self._add_particles(
            target
        )


        self._add_voxel_counts(
            target
        )


        self._add_extra_providers(
            target
        )


        return target


    # ========================================================
    # Synchronize
    # ========================================================

    def rebuild_dynamic(
        self,
    ):

        self._calculate_dynamic_volume(
            self._scratch_nm3
        )


        self.crowding_state.dynamic_volume_nm3[
            :
        ] = self._scratch_nm3


        self.crowding_state.refresh_movement_accessibility()


        self.rebuild_count += 1


        return (
            self.crowding_state
            .dynamic_volume_nm3
        )


    # ========================================================
    # Diagnostics
    # ========================================================

    @property
    def dynamic_volume_nm3(
        self,
    ):

        return float(
            cp.sum(
                self.crowding_state
                .dynamic_volume_nm3,

                dtype=cp.float64,
            ).item()
        )


    @property
    def provider_count(
        self,
    ):

        return len(
            self._extra_providers
        )


    # ========================================================
    # Validation
    # ========================================================

    def validate(
        self,
    ):

        expected = cp.zeros(
            self.cell_voxels,
            dtype=cp.float32,
        )


        self._calculate_dynamic_volume(
            expected
        )


        observed = (
            self.crowding_state
            .dynamic_volume_nm3
        )


        if not bool(
            cp.allclose(
                expected,
                observed,
                rtol=1e-6,
                atol=1e-3,
            ).item()
        ):

            maximum_error = float(
                cp.max(
                    cp.abs(
                        expected
                        - observed
                    )
                ).item()
            )


            raise RuntimeError(
                "Dynamic crowding state is not "
                "synchronized with molecular state. "
                f"Maximum voxel error: "
                f"{maximum_error:.6f} nm^3"
            )


        self.crowding_state.validate()


        return True


    # ========================================================
    # Summary
    # ========================================================

    def describe(
        self,
    ):

        print()
        print("DYNAMIC CROWDING SYNCHRONIZER")
        print("--------------------------------------------")


        print(
            f"Rebuilds completed:          "
            f"{self.rebuild_count:,}"
        )


        print(
            f"Particle state attached:    "
            f"{self.particle_state is not None}"
        )


        print(
            f"Voxel-count state attached: "
            f"{self.voxel_count_state is not None}"
        )


        print(
            f"Extra providers:            "
            f"{self.provider_count:,}"
        )


        print(
            f"Dynamic molecular volume:   "
            f"{self.dynamic_volume_nm3:,.1f} nm^3"
        )
