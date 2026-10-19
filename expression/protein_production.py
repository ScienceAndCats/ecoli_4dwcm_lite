from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np

from expression.translation import (
    PRODUCT_PARTICLE,
    PRODUCT_VOXEL_COUNT,
    TranslationEngine,
)
from spatial.particles import (
    ParticleState,
)
from spatial.voxel_counts import (
    VoxelCountState,
)


class ProteinProductionEngine:
    """
    Convert completed translation events into spatial protein
    molecules.

    A completed translation event contains:

        transcription-template index
        completion voxel

    The registry then determines whether the protein product
    should become:

        an explicit particle

    or:

        an integer voxel-count molecule


    Products are created in the same voxel in which their
    ribosome completed translation.
    """

    THREADS = 256


    def __init__(
        self,
        translation_engine: TranslationEngine,
        particle_state: ParticleState | None = None,
        voxel_count_state: VoxelCountState | None = None,
        crowding_synchronizer=None,
    ):

        self.translation_engine = (
            translation_engine
        )


        self.registry = (
            translation_engine
            .registry
        )


        self.particle_state = (
            particle_state
        )


        self.voxel_count_state = (
            voxel_count_state
        )


        self.crowding_synchronizer = (
            crowding_synchronizer
        )


        species_registry = (
            self.registry
            .species_registry
        )


        # ====================================================
        # Validate required storage backends
        # ====================================================

        needs_particles = bool(
            np.any(
                self.registry
                .product_representation_codes
                == PRODUCT_PARTICLE
            )
        )


        needs_voxel_counts = bool(
            np.any(
                self.registry
                .product_representation_codes
                == PRODUCT_VOXEL_COUNT
            )
        )


        if (
            needs_particles
            and particle_state is None
        ):

            raise ValueError(
                "Translation registry contains particle "
                "protein products but no ParticleState "
                "was supplied."
            )


        if (
            needs_voxel_counts
            and voxel_count_state is None
        ):

            raise ValueError(
                "Translation registry contains voxel-count "
                "protein products but no VoxelCountState "
                "was supplied."
            )


        if (
            particle_state is not None
            and particle_state.registry
            is not species_registry
        ):

            raise ValueError(
                "Protein production ParticleState does not "
                "share the translation SpeciesRegistry."
            )


        if (
            voxel_count_state is not None
            and voxel_count_state.registry
            is not species_registry
        ):

            raise ValueError(
                "Protein production VoxelCountState does not "
                "share the translation SpeciesRegistry."
            )


        if (
            particle_state is not None
            and particle_state.cell_voxels
            != self.registry.cell_voxels
        ):

            raise ValueError(
                "ParticleState and translation registry "
                "use different spatial grids."
            )


        if (
            voxel_count_state is not None
            and voxel_count_state.cell_voxels
            != self.registry.cell_voxels
        ):

            raise ValueError(
                "VoxelCountState and translation registry "
                "use different spatial grids."
            )


        # ====================================================
        # Product lookup arrays
        # ====================================================

        self.product_representation_codes = cp.asarray(
            self.registry
            .product_representation_codes
        )


        self.product_numeric_ids = cp.asarray(
            self.registry
            .product_numeric_ids
        )


        # ====================================================
        # Particle backend or safe dummy arrays
        # ====================================================

        if particle_state is not None:

            self.primary = (
                particle_state.primary
            )


            self.primary_occupancy = (
                particle_state.primary_occupancy
            )


            self.spill = (
                particle_state.spill
            )


            self.spill_occupancy = (
                particle_state.spill_occupancy
            )


            self.primary_slots = int(
                particle_state.primary_slots
            )


            self.spill_slots = int(
                particle_state.spill_slots
            )


        else:

            self.primary = cp.zeros(
                1,
                dtype=cp.uint16,
            )


            self.primary_occupancy = cp.zeros(
                1,
                dtype=cp.uint32,
            )


            self.spill = cp.zeros(
                1,
                dtype=cp.uint16,
            )


            self.spill_occupancy = cp.zeros(
                1,
                dtype=cp.uint32,
            )


            self.primary_slots = 0
            self.spill_slots = 0


        # ====================================================
        # Voxel-count backend or dummy array
        # ====================================================

        if voxel_count_state is not None:

            self.voxel_counts = (
                voxel_count_state.counts
            )


            self.voxel_count_species = int(
                voxel_count_state.species_count
            )


        else:

            self.voxel_counts = cp.zeros(
                1,
                dtype=cp.uint32,
            )


            self.voxel_count_species = 0


        # ====================================================
        # Diagnostics
        # ====================================================

        self.produced_by_template = cp.zeros(
            self.registry
            .transcription_template_count,

            dtype=cp.uint64,
        )


        # 0 successfully produced
        # 1 particle storage temporarily full
        # 2 voxel-count integer overflow blocked

        self.event_counts = cp.zeros(
            3,
            dtype=cp.uint64,
        )


        self.step_count = 0


        # ====================================================
        # CUDA
        # ====================================================

        kernel_path = (
            Path(__file__).parents[1]
            / "spatial"
            / "kernels"
            / "protein_production.cu"
        )


        module = cp.RawModule(
            code=kernel_path.read_text(),

            options=(
                "--std=c++11",
            ),
        )


        self._produce = (
            module.get_function(
                "produce_proteins"
            )
        )


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
    # Production
    # ========================================================

    def step(self):

        maximum = (
            self.registry
            .max_active_complexes
        )


        self._produce(
            self._blocks(
                maximum
            ),

            (
                self.THREADS,
            ),

            (
                self.translation_engine
                .product_event_active,

                self.translation_engine
                .product_event_template_index,

                self.translation_engine
                .product_event_voxel_id,

                self.product_representation_codes,
                self.product_numeric_ids,

                self.primary,
                self.primary_occupancy,

                self.spill,
                self.spill_occupancy,

                self.voxel_counts,

                self.produced_by_template,
                self.event_counts,

                np.int32(
                    maximum
                ),

                np.int32(
                    self.registry
                    .cell_voxels
                ),

                np.int32(
                    self.primary_slots
                ),

                np.int32(
                    self.spill_slots
                ),

                np.int32(
                    self.voxel_count_species
                ),
            ),
        )


        if (
            self.crowding_synchronizer
            is not None
        ):

            self.crowding_synchronizer.rebuild_dynamic()


        self.step_count += 1


    # ========================================================
    # Statistics
    # ========================================================

    @property
    def total_produced(self):

        return int(
            self.event_counts[
                0
            ].item()
        )


    @property
    def particle_storage_block_events(self):

        return int(
            self.event_counts[
                1
            ].item()
        )


    @property
    def voxel_count_overflow_events(self):

        return int(
            self.event_counts[
                2
            ].item()
        )


    @property
    def pending_product_count(self):

        return (
            self.translation_engine
            .pending_product_count
        )


    def produced_for_template(
        self,
        template_index,
    ):

        return int(
            self.produced_by_template[
                int(
                    template_index
                )
            ].item()
        )


    # ========================================================
    # Validation
    # ========================================================

    def validate(self):

        self.translation_engine.validate()


        if (
            self.particle_state
            is not None
        ):

            self.particle_state.validate()


        if (
            self.voxel_count_state
            is not None
        ):

            self.voxel_count_state.validate(
                require_initial_totals=False
            )


        if (
            self.crowding_synchronizer
            is not None
        ):

            self.crowding_synchronizer.validate()


        return True


    # ========================================================
    # Description
    # ========================================================

    def describe(self):

        print()
        print("PROTEIN PRODUCTION")
        print("--------------------------------------------")


        print(
            f"Production steps:           "
            f"{self.step_count:,}"
        )


        print(
            f"Proteins produced:          "
            f"{self.total_produced:,}"
        )


        print(
            f"Pending products:           "
            f"{self.pending_product_count:,}"
        )


        print(
            f"Particle-storage blocks:    "
            f"{self.particle_storage_block_events:,}"
        )


        print(
            f"Voxel-count overflow:       "
            f"{self.voxel_count_overflow_events:,}"
        )
