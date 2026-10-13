from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from spatial.species import (
    Representation,
    SpeciesRegistry,
)


class EffectiveVolumeRegistry:
    """
    Effective excluded/crowding volumes for molecular objects.

    Volumes are stored in nm^3 per molecule/object.

    This registry is deliberately separate from SpeciesRegistry.

    SpeciesRegistry answers:

        What is this molecule?
        How does it diffuse?
        How is it represented?

    EffectiveVolumeRegistry answers:

        How much physical/crowding volume should one copy
        contribute to its voxel?


    Two categories are supported:

        species

            Named SpeciesRegistry entries such as:

                example_TF
                RNAP
                Ribosome


        complexes

            Objects that are not ordinary free species, such as:

                transcription_complex
                translation_complex
                TF_promoter_complex


    Continuous concentration fields are NOT assigned explicit
    per-molecule volumes here because they are not represented
    as discrete molecule counts.
    """

    def __init__(
        self,
        species_registry: SpeciesRegistry,
        species_volumes,
        complex_volumes,
    ):

        self.species_registry = (
            species_registry
        )


        self._species_volumes = {
            str(
                name
            ): float(
                volume
            )

            for name, volume
            in species_volumes.items()
        }


        self._complex_volumes = {
            str(
                name
            ): float(
                volume
            )

            for name, volume
            in complex_volumes.items()
        }


        self._validate()

        self._compile_lookup_tables()


    # ========================================================
    # Loading
    # ========================================================

    @classmethod
    def from_yaml(
        cls,
        path,
        species_registry: SpeciesRegistry,
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


        if raw is None:

            raw = {}


        section = raw.get(
            "effective_volumes",
            {}
        )


        species_raw = section.get(
            "species",
            {}
        )


        complex_raw = section.get(
            "complexes",
            {}
        )


        species_volumes = {}


        for name, definition in (
            species_raw.items()
        ):

            species_volumes[
                name
            ] = cls._read_volume(
                name=name,
                definition=definition,
                category="species",
            )


        complex_volumes = {}


        for name, definition in (
            complex_raw.items()
        ):

            complex_volumes[
                name
            ] = cls._read_volume(
                name=name,
                definition=definition,
                category="complex",
            )


        return cls(
            species_registry=species_registry,
            species_volumes=species_volumes,
            complex_volumes=complex_volumes,
        )


    @staticmethod
    def _read_volume(
        name,
        definition,
        category,
    ):

        # Allow either:
        #
        #   Ribosome: 8000
        #
        # or:
        #
        #   Ribosome:
        #       effective_volume_nm3: 8000

        if isinstance(
            definition,
            dict,
        ):

            if (
                "effective_volume_nm3"
                not in definition
            ):

                raise ValueError(
                    f"{category} {name} must define "
                    "effective_volume_nm3."
                )


            value = definition[
                "effective_volume_nm3"
            ]


        else:

            value = definition


        return float(
            value
        )


    # ========================================================
    # Validation
    # ========================================================

    def _validate(
        self,
    ):

        # ====================================================
        # Species names must actually exist.
        # ====================================================

        for name, volume in (
            self._species_volumes.items()
        ):

            try:

                species = (
                    self.species_registry.get(
                        name
                    )
                )


            except KeyError as error:

                raise ValueError(
                    f"Effective-volume definition refers "
                    f"to unknown species: {name}"
                ) from error


            # Continuous concentration fields do not have an
            # integer molecular count that can simply be
            # multiplied by a per-molecule excluded volume.

            if (
                species.representation
                is Representation.FIELD
            ):

                raise ValueError(
                    f"{name}: continuous field species "
                    "cannot currently define a discrete "
                    "effective_volume_nm3."
                )


            if not np.isfinite(
                volume
            ):

                raise ValueError(
                    f"{name}: effective volume "
                    "must be finite."
                )


            if volume < 0:

                raise ValueError(
                    f"{name}: effective volume "
                    "cannot be negative."
                )


        # ====================================================
        # Complex volumes
        # ====================================================

        for name, volume in (
            self._complex_volumes.items()
        ):

            if not np.isfinite(
                volume
            ):

                raise ValueError(
                    f"{name}: complex effective volume "
                    "must be finite."
                )


            if volume < 0:

                raise ValueError(
                    f"{name}: complex effective volume "
                    "cannot be negative."
                )


    # ========================================================
    # Compile GPU-friendly lookup tables
    # ========================================================

    def _compile_lookup_tables(
        self,
    ):

        # ====================================================
        # Particle IDs
        #
        # Particle ID 0 means empty slot, so its volume
        # must always be zero.
        # ====================================================

        self.particle_volume_lut_nm3 = np.zeros(
            self.species_registry
            .particle_count
            + 1,

            dtype=np.float32,
        )


        for species in (
            self.species_registry
            .particle_species
        ):

            volume = (
                self._species_volumes.get(
                    species.name,
                    0.0,
                )
            )


            self.particle_volume_lut_nm3[
                species.particle_id
            ] = volume


        # ====================================================
        # Voxel-count IDs start at zero.
        # ====================================================

        self.voxel_count_volume_lut_nm3 = np.zeros(
            self.species_registry
            .voxel_count_count,

            dtype=np.float32,
        )


        for species in (
            self.species_registry
            .voxel_count_species
        ):

            volume = (
                self._species_volumes.get(
                    species.name,
                    0.0,
                )
            )


            self.voxel_count_volume_lut_nm3[
                species.voxel_count_id
            ] = volume


    # ========================================================
    # Lookup
    # ========================================================

    def species_volume_nm3(
        self,
        species_name,
    ):

        # Confirm that the species itself exists.

        species = (
            self.species_registry.get(
                species_name
            )
        )


        if (
            species.representation
            is Representation.FIELD
        ):

            raise ValueError(
                f"{species_name} is a continuous "
                "concentration field."
            )


        try:

            return self._species_volumes[
                species_name
            ]


        except KeyError:

            raise KeyError(
                f"No effective volume configured "
                f"for species {species_name}."
            )


    def complex_volume_nm3(
        self,
        complex_name,
    ):

        try:

            return self._complex_volumes[
                complex_name
            ]


        except KeyError:

            raise KeyError(
                f"No effective volume configured "
                f"for complex {complex_name}."
            )


    def volume_for_species_count(
        self,
        species_name,
        count,
    ):

        count = int(
            count
        )


        if count < 0:

            raise ValueError(
                "Molecule count cannot be negative."
            )


        return (
            self.species_volume_nm3(
                species_name
            )
            * count
        )


    # ========================================================
    # Completeness
    # ========================================================

    def missing_discrete_species(
        self,
    ):

        missing = []


        for species in (
            self.species_registry
            .particle_species
        ):

            if (
                species.name
                not in self._species_volumes
            ):

                missing.append(
                    species.name
                )


        for species in (
            self.species_registry
            .voxel_count_species
        ):

            if (
                species.name
                not in self._species_volumes
            ):

                missing.append(
                    species.name
                )


        return tuple(
            missing
        )


    def validate_complete_discrete(
        self,
    ):

        missing = (
            self.missing_discrete_species()
        )


        if missing:

            raise RuntimeError(
                "Missing effective volumes for "
                "discrete species: "
                + ", ".join(
                    missing
                )
            )


        return True


    # ========================================================
    # Summary
    # ========================================================

    @property
    def species_volume_count(
        self,
    ):

        return len(
            self._species_volumes
        )


    @property
    def complex_volume_count(
        self,
    ):

        return len(
            self._complex_volumes
        )


    def describe(
        self,
    ):

        print()
        print("EFFECTIVE VOLUME REGISTRY")
        print("--------------------------------------------")


        print(
            f"Species volumes:            "
            f"{self.species_volume_count:,}"
        )


        print(
            f"Complex volumes:            "
            f"{self.complex_volume_count:,}"
        )


        print()
        print("SPECIES")
        print("--------------------------------------------")


        for name, volume in sorted(
            self._species_volumes.items()
        ):

            print(
                f"{name:28s} "
                f"{volume:12.2f} nm^3"
            )


        print()
        print("COMPLEXES")
        print("--------------------------------------------")


        for name, volume in sorted(
            self._complex_volumes.items()
        ):

            print(
                f"{name:28s} "
                f"{volume:12.2f} nm^3"
            )
