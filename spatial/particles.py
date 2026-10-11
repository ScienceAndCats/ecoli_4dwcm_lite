from __future__ import annotations

import cupy as cp
import numpy as np

from spatial.config import ParticleConfig
from spatial.grid import SpatialGrid
from spatial.species import SpeciesRegistry


MAX_PARTICLE_SPECIES_ID = 65534


class ParticleState:

    def __init__(
        self,
        grid: SpatialGrid,
        registry: SpeciesRegistry,
        config: ParticleConfig,
        seed: int,
    ):

        self.grid = grid
        self.registry = registry
        self.config = config
        self.seed = seed

        self.primary_slots = (
            config.primary_slots_per_voxel
        )

        self.spill_slots = (
            config.spill_slots_per_voxel
        )

        self.total_slots = (
            config.total_slots_per_voxel
        )

        self.cell_voxels = (
            grid.cell_voxels
        )


        if (
            registry.particle_count
            > MAX_PARTICLE_SPECIES_ID
        ):

            raise ValueError(
                "Too many particle species for uint16 "
                f"storage: {registry.particle_count:,}. "
                f"Current maximum is "
                f"{MAX_PARTICLE_SPECIES_ID:,}."
            )


        (
            primary_cpu,
            spill_cpu,
            primary_occ_cpu,
            spill_occ_cpu,
        ) = self._build_initial_state()


        self.primary = cp.asarray(
            primary_cpu
        )

        self.spill = cp.asarray(
            spill_cpu
        )

        self.primary_occupancy = cp.asarray(
            primary_occ_cpu
        )

        self.spill_occupancy = cp.asarray(
            spill_occ_cpu
        )


        self.primary_next = cp.zeros_like(
            self.primary
        )

        self.spill_next = cp.zeros_like(
            self.spill
        )

        self.primary_occupancy_next = (
            cp.zeros_like(
                self.primary_occupancy
            )
        )

        self.spill_occupancy_next = (
            cp.zeros_like(
                self.spill_occupancy
            )
        )


        self.initial_particle_count = int(
            sum(
                species.initial_count or 0
                for species
                in registry.particle_species
            )
        )


    def _build_initial_state(self):

        particle_ids = []
        total_particles = 0


        for species in self.registry.particle_species:

            count = int(
                species.initial_count or 0
            )

            if count == 0:
                continue

            total_particles += count

            particle_ids.append(
                np.full(
                    count,
                    species.particle_id,
                    dtype=np.uint16,
                )
            )


        if total_particles == 0:

            species_ids = np.empty(
                0,
                dtype=np.uint16,
            )

        else:

            species_ids = np.concatenate(
                particle_ids
            )


        rng = np.random.default_rng(
            self.seed
        )

        voxel_ids = rng.integers(
            0,
            self.cell_voxels,
            size=total_particles,
            dtype=np.int32,
        )

        rng.shuffle(
            species_ids
        )


        order = np.argsort(
            voxel_ids,
            kind="stable",
        )

        voxel_ids = voxel_ids[
            order
        ]

        species_ids = species_ids[
            order
        ]


        total_occupancy = np.bincount(
            voxel_ids,
            minlength=self.cell_voxels,
        ).astype(
            np.uint32
        )


        max_occupancy = int(
            total_occupancy.max(
                initial=0
            )
        )


        if max_occupancy > self.total_slots:

            raise RuntimeError(
                "Initial particle placement exceeds voxel "
                "capacity.\n"
                f"Maximum occupancy: {max_occupancy}\n"
                f"Capacity / voxel: {self.total_slots}"
            )


        primary = np.zeros(
            (
                self.cell_voxels,
                self.primary_slots,
            ),
            dtype=np.uint16,
        )

        spill = np.zeros(
            (
                self.cell_voxels,
                self.spill_slots,
            ),
            dtype=np.uint16,
        )

        primary_occupancy = np.zeros(
            self.cell_voxels,
            dtype=np.uint32,
        )

        spill_occupancy = np.zeros(
            self.cell_voxels,
            dtype=np.uint32,
        )


        if total_particles == 0:

            return (
                primary,
                spill,
                primary_occupancy,
                spill_occupancy,
            )


        starts = (
            np.cumsum(
                total_occupancy,
                dtype=np.int64,
            )
            - total_occupancy
        )


        for voxel in range(
            self.cell_voxels
        ):

            count = int(
                total_occupancy[
                    voxel
                ]
            )

            if count == 0:
                continue


            start = int(
                starts[
                    voxel
                ]
            )

            end = (
                start
                + count
            )

            species_here = species_ids[
                start:end
            ]


            primary_count = min(
                count,
                self.primary_slots,
            )

            spill_count = (
                count
                - primary_count
            )


            primary[
                voxel,
                :primary_count
            ] = species_here[
                :primary_count
            ]

            primary_occupancy[
                voxel
            ] = primary_count


            if spill_count > 0:

                spill[
                    voxel,
                    :spill_count
                ] = species_here[
                    primary_count:
                ]

                spill_occupancy[
                    voxel
                ] = spill_count


        return (
            primary,
            spill,
            primary_occupancy,
            spill_occupancy,
        )


    def clear_next(self):

        self.primary_next.fill(0)
        self.spill_next.fill(0)

        self.primary_occupancy_next.fill(0)
        self.spill_occupancy_next.fill(0)


    def swap_buffers(self):

        (
            self.primary,
            self.primary_next,
        ) = (
            self.primary_next,
            self.primary,
        )

        (
            self.spill,
            self.spill_next,
        ) = (
            self.spill_next,
            self.spill,
        )

        (
            self.primary_occupancy,
            self.primary_occupancy_next,
        ) = (
            self.primary_occupancy_next,
            self.primary_occupancy,
        )

        (
            self.spill_occupancy,
            self.spill_occupancy_next,
        ) = (
            self.spill_occupancy_next,
            self.spill_occupancy,
        )


    @property
    def primary_particle_count(self):

        return int(
            cp.sum(
                self.primary_occupancy
            ).item()
        )


    @property
    def spill_particle_count(self):

        return int(
            cp.sum(
                self.spill_occupancy
            ).item()
        )


    @property
    def total_particle_count(self):

        return (
            self.primary_particle_count
            + self.spill_particle_count
        )


    @property
    def max_primary_occupancy(self):

        return int(
            cp.max(
                self.primary_occupancy
            ).item()
        )


    @property
    def max_spill_occupancy(self):

        return int(
            cp.max(
                self.spill_occupancy
            ).item()
        )


    @property
    def state_bytes(self):

        arrays = (
            self.primary,
            self.spill,
            self.primary_occupancy,
            self.spill_occupancy,
            self.primary_next,
            self.spill_next,
            self.primary_occupancy_next,
            self.spill_occupancy_next,
        )

        return sum(
            array.nbytes
            for array in arrays
        )


    @property
    def state_mib(self):

        return (
            self.state_bytes
            / 1024**2
        )


    def validate(
        self,
        expected_count=None,
    ):

        errors = []


        if (
            self.max_primary_occupancy
            > self.primary_slots
        ):

            errors.append(
                "Primary occupancy exceeds "
                "primary capacity."
            )


        if (
            self.max_spill_occupancy
            > self.spill_slots
        ):

            errors.append(
                "Spill occupancy exceeds "
                "spill capacity."
            )


        if expected_count is not None:

            if (
                self.total_particle_count
                != expected_count
            ):

                errors.append(
                    "Particle count mismatch: "
                    f"expected {expected_count:,}, "
                    f"found "
                    f"{self.total_particle_count:,}."
                )


        if errors:

            raise RuntimeError(
                "\n".join(
                    errors
                )
            )


        return True


    def describe(self):

        print()
        print("PARTICLE STATE")
        print("--------------------------------------------")

        print(
            f"Initial particles:          "
            f"{self.initial_particle_count:,}"
        )

        print(
            f"Current particles:          "
            f"{self.total_particle_count:,}"
        )

        print(
            f"Primary particles:          "
            f"{self.primary_particle_count:,}"
        )

        print(
            f"Spill particles:            "
            f"{self.spill_particle_count:,}"
        )

        print(
            f"Primary slots / voxel:      "
            f"{self.primary_slots}"
        )

        print(
            f"Spill slots / voxel:        "
            f"{self.spill_slots}"
        )

        print(
            f"Maximum primary occupancy:  "
            f"{self.max_primary_occupancy}"
        )

        print(
            f"Maximum spill occupancy:    "
            f"{self.max_spill_occupancy}"
        )

        print(
            f"Particle buffer memory:     "
            f"{self.state_mib:.2f} MiB"
        )