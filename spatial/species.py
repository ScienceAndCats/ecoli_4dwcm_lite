from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class Representation(str, Enum):
    PARTICLE = "particle"
    FIELD = "field"
    VOXEL_COUNT = "voxel_count"


@dataclass(frozen=True)
class Species:
    name: str

    representation: Representation

    diffusion_um2_s: float

    # Used by:
    #
    #   particle
    #   voxel_count
    #
    # but not continuous concentration fields.
    initial_count: int | None = None

    # Continuous concentration-field species.
    initial_concentration_mM: float | None = None


    # ========================================================
    # Numeric IDs
    #
    # Each representation has its own ID namespace.
    # ========================================================

    # Particle ID 0 is reserved for an empty particle slot.
    #
    # Real particle species therefore start at 1.
    particle_id: int | None = None

    # Field IDs start at 0.
    field_id: int | None = None

    # Voxel-count IDs also start at 0.
    voxel_count_id: int | None = None


class SpeciesRegistry:

    def __init__(self):

        self._by_name = {}

        self._particles = []
        self._fields = []
        self._voxel_counts = []


    @classmethod
    def from_yaml(
        cls,
        path,
    ):

        path = Path(
            path
        )


        with path.open(
            "r"
        ) as handle:

            raw = yaml.safe_load(
                handle
            )


        registry = cls()


        species_data = raw.get(
            "species",
            {}
        )


        for name, definition in (
            species_data.items()
        ):

            registry.add_from_definition(
                name=name,
                definition=definition,
            )


        return registry


    def add_from_definition(
        self,
        name,
        definition,
    ):

        if name in self._by_name:

            raise ValueError(
                f"Duplicate species name: {name}"
            )


        representation = Representation(
            definition[
                "representation"
            ]
        )


        diffusion = float(
            definition[
                "diffusion_um2_s"
            ]
        )


        if diffusion < 0:

            raise ValueError(
                f"{name}: diffusion_um2_s "
                "cannot be negative"
            )


        # ====================================================
        # Explicit particle species
        # ====================================================

        if (
            representation
            is Representation.PARTICLE
        ):

            if (
                "initial_concentration_mM"
                in definition
            ):

                raise ValueError(
                    f"{name}: particle species cannot define "
                    "initial_concentration_mM"
                )


            initial_count = int(
                definition.get(
                    "initial_count",
                    0,
                )
            )


            if initial_count < 0:

                raise ValueError(
                    f"{name}: initial_count "
                    "cannot be negative"
                )


            particle_id = (
                len(
                    self._particles
                )
                + 1
            )


            species = Species(
                name=name,

                representation=(
                    representation
                ),

                diffusion_um2_s=(
                    diffusion
                ),

                initial_count=(
                    initial_count
                ),

                particle_id=(
                    particle_id
                ),
            )


            self._particles.append(
                species
            )


        # ====================================================
        # Continuous concentration-field species
        # ====================================================

        elif (
            representation
            is Representation.FIELD
        ):

            if (
                "initial_count"
                in definition
            ):

                raise ValueError(
                    f"{name}: field species cannot define "
                    "initial_count"
                )


            initial_concentration = float(
                definition.get(
                    "initial_concentration_mM",
                    0.0,
                )
            )


            if (
                initial_concentration
                < 0
            ):

                raise ValueError(
                    f"{name}: concentration "
                    "cannot be negative"
                )


            field_id = len(
                self._fields
            )


            species = Species(
                name=name,

                representation=(
                    representation
                ),

                diffusion_um2_s=(
                    diffusion
                ),

                initial_concentration_mM=(
                    initial_concentration
                ),

                field_id=(
                    field_id
                ),
            )


            self._fields.append(
                species
            )


        # ====================================================
        # Integer voxel-count species
        #
        # Example:
        #
        # voxel 812:
        #     free ribosomes = 7
        #
        # Individual molecules have no persistent identity.
        # ====================================================

        elif (
            representation
            is Representation.VOXEL_COUNT
        ):

            if (
                "initial_concentration_mM"
                in definition
            ):

                raise ValueError(
                    f"{name}: voxel_count species cannot "
                    "define initial_concentration_mM"
                )


            initial_count = int(
                definition.get(
                    "initial_count",
                    0,
                )
            )


            if initial_count < 0:

                raise ValueError(
                    f"{name}: initial_count "
                    "cannot be negative"
                )


            voxel_count_id = len(
                self._voxel_counts
            )


            species = Species(
                name=name,

                representation=(
                    representation
                ),

                diffusion_um2_s=(
                    diffusion
                ),

                initial_count=(
                    initial_count
                ),

                voxel_count_id=(
                    voxel_count_id
                ),
            )


            self._voxel_counts.append(
                species
            )


        else:

            raise ValueError(
                f"Unsupported representation: "
                f"{representation}"
            )


        self._by_name[
            name
        ] = species


    def get(
        self,
        name,
    ):

        try:

            return self._by_name[
                name
            ]


        except KeyError:

            raise KeyError(
                f"Unknown species: {name}"
            )


    @property
    def particle_species(self):

        return tuple(
            self._particles
        )


    @property
    def field_species(self):

        return tuple(
            self._fields
        )


    @property
    def voxel_count_species(self):

        return tuple(
            self._voxel_counts
        )


    @property
    def particle_count(self):

        return len(
            self._particles
        )


    @property
    def field_count(self):

        return len(
            self._fields
        )


    @property
    def voxel_count_count(self):

        return len(
            self._voxel_counts
        )


    @property
    def total_species(self):

        return len(
            self._by_name
        )


    @property
    def fastest_diffusion_um2_s(self):

        if not self._by_name:

            return 0.0


        return max(
            species.diffusion_um2_s

            for species
            in self._by_name.values()
        )


    def describe(self):

        print()
        print("SPECIES REGISTRY")
        print("--------------------------------------------")


        print(
            f"Total species:              "
            f"{self.total_species:,}"
        )


        print(
            f"Particle species:           "
            f"{self.particle_count:,}"
        )


        print(
            f"Field species:              "
            f"{self.field_count:,}"
        )


        print(
            f"Voxel-count species:        "
            f"{self.voxel_count_count:,}"
        )


        print(
            f"Fastest diffusion D:        "
            f"{self.fastest_diffusion_um2_s:.3f} um^2/s"
        )


        print()
        print("PARTICLE SPECIES")
        print("--------------------------------------------")


        for species in (
            self._particles
        ):

            print(
                f"{species.particle_id:5d}  "
                f"{species.name:24s} "
                f"count={species.initial_count:<8d} "
                f"D={species.diffusion_um2_s:.3f}"
            )


        print()
        print("FIELD SPECIES")
        print("--------------------------------------------")


        for species in (
            self._fields
        ):

            print(
                f"{species.field_id:5d}  "
                f"{species.name:24s} "
                f"initial="
                f"{species.initial_concentration_mM:<8.3f} mM "
                f"D={species.diffusion_um2_s:.3f}"
            )


        print()
        print("VOXEL-COUNT SPECIES")
        print("--------------------------------------------")


        for species in (
            self._voxel_counts
        ):

            print(
                f"{species.voxel_count_id:5d}  "
                f"{species.name:24s} "
                f"count={species.initial_count:<8d} "
                f"D={species.diffusion_um2_s:.3f}"
            )