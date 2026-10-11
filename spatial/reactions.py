from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cupy as cp
import numpy as np
import yaml

from spatial.fields import FieldState
from spatial.particles import ParticleState
from spatial.species import SpeciesRegistry


AVOGADRO = 6.02214076e23


class ReactionType(IntEnum):
    DECAY = 1
    CONVERSION = 2
    CATALYTIC = 3
    BINDING = 4
    UNBINDING = 5
    PARTICLE_FIELD = 6


@dataclass(frozen=True)
class ReactionDefinition:
    name: str
    reaction_type: ReactionType

    trigger_particle_id: int

    partner_particle_id: int = 0

    product_a_id: int = 0
    product_b_id: int = 0

    rate: float = 0.0

    field_substrate_id: int = -1
    field_product_id: int = -1


class ReactionRegistry:

    def __init__(
        self,
        species_registry: SpeciesRegistry,
        definitions,
    ):

        self.species_registry = (
            species_registry
        )

        self.definitions = tuple(
            definitions
        )

        self._compile()


    @classmethod
    def from_yaml(
        cls,
        path,
        species_registry: SpeciesRegistry,
        voxel_volume_liters: float,
    ):

        path = Path(path)

        with path.open("r") as handle:
            raw = yaml.safe_load(handle)

        definitions = []


        def particle_id(name):

            species = (
                species_registry.get(
                    name
                )
            )

            if species.particle_id is None:

                raise ValueError(
                    f"{name} must be a particle species."
                )

            return species.particle_id


        def field_id(name):

            species = (
                species_registry.get(
                    name
                )
            )

            if species.field_id is None:

                raise ValueError(
                    f"{name} must be a field species."
                )

            return species.field_id


        def bimolecular_rate(
            macroscopic_rate,
        ):

            return (
                float(macroscopic_rate)
                /
                (
                    AVOGADRO
                    * voxel_volume_liters
                )
            )


        for item in raw.get(
            "reactions",
            []
        ):

            name = str(
                item["name"]
            )

            reaction_type = str(
                item["type"]
            )


            if reaction_type == "decay":

                definitions.append(
                    ReactionDefinition(
                        name=name,

                        reaction_type=(
                            ReactionType.DECAY
                        ),

                        trigger_particle_id=(
                            particle_id(
                                item["reactant"]
                            )
                        ),

                        rate=float(
                            item["rate_s"]
                        ),
                    )
                )


            elif reaction_type == "conversion":

                definitions.append(
                    ReactionDefinition(
                        name=name,

                        reaction_type=(
                            ReactionType.CONVERSION
                        ),

                        trigger_particle_id=(
                            particle_id(
                                item["reactant"]
                            )
                        ),

                        product_a_id=(
                            particle_id(
                                item["product"]
                            )
                        ),

                        rate=float(
                            item["rate_s"]
                        ),
                    )
                )


            elif reaction_type == "catalytic":

                definitions.append(
                    ReactionDefinition(
                        name=name,

                        reaction_type=(
                            ReactionType.CATALYTIC
                        ),

                        trigger_particle_id=(
                            particle_id(
                                item["substrate"]
                            )
                        ),

                        partner_particle_id=(
                            particle_id(
                                item["catalyst"]
                            )
                        ),

                        product_a_id=(
                            particle_id(
                                item["product"]
                            )
                        ),

                        rate=bimolecular_rate(
                            item["rate_M_inv_s"]
                        ),
                    )
                )


            elif reaction_type == "binding":

                definitions.append(
                    ReactionDefinition(
                        name=name,

                        reaction_type=(
                            ReactionType.BINDING
                        ),

                        trigger_particle_id=(
                            particle_id(
                                item["reactant"]
                            )
                        ),

                        partner_particle_id=(
                            particle_id(
                                item["partner"]
                            )
                        ),

                        product_a_id=(
                            particle_id(
                                item["product"]
                            )
                        ),

                        rate=bimolecular_rate(
                            item["rate_M_inv_s"]
                        ),
                    )
                )


            elif reaction_type == "unbinding":

                definitions.append(
                    ReactionDefinition(
                        name=name,

                        reaction_type=(
                            ReactionType.UNBINDING
                        ),

                        trigger_particle_id=(
                            particle_id(
                                item["reactant"]
                            )
                        ),

                        product_a_id=(
                            particle_id(
                                item["product_a"]
                            )
                        ),

                        product_b_id=(
                            particle_id(
                                item["product_b"]
                            )
                        ),

                        rate=float(
                            item["rate_s"]
                        ),
                    )
                )


            elif reaction_type == "particle_field":

                definitions.append(
                    ReactionDefinition(
                        name=name,

                        reaction_type=(
                            ReactionType.PARTICLE_FIELD
                        ),

                        trigger_particle_id=(
                            particle_id(
                                item["particle"]
                            )
                        ),

                        field_substrate_id=(
                            field_id(
                                item[
                                    "field_substrate"
                                ]
                            )
                        ),

                        field_product_id=(
                            field_id(
                                item[
                                    "field_product"
                                ]
                            )
                        ),

                        rate=float(
                            item[
                                "rate_per_particle_s"
                            ]
                        ),
                    )
                )


            else:

                raise ValueError(
                    f"Unknown reaction type: "
                    f"{reaction_type}"
                )


        return cls(
            species_registry=species_registry,
            definitions=definitions,
        )


    def _compile(self):

        particle_count = (
            self.species_registry
            .particle_count
        )


        grouped = [
            []
            for _ in range(
                particle_count + 1
            )
        ]


        for reaction in self.definitions:

            grouped[
                reaction.trigger_particle_id
            ].append(
                reaction
            )


        ordered = []

        offsets = np.zeros(
            particle_count + 2,
            dtype=np.int32,
        )


        for particle_id in range(
            particle_count + 1
        ):

            offsets[
                particle_id
            ] = len(
                ordered
            )

            ordered.extend(
                grouped[
                    particle_id
                ]
            )


        offsets[
            particle_count + 1
        ] = len(
            ordered
        )


        self.ordered_definitions = tuple(
            ordered
        )

        self.offsets = offsets


        n = len(
            ordered
        )


        self.types = np.zeros(
            n,
            dtype=np.uint8,
        )

        self.partners = np.zeros(
            n,
            dtype=np.uint16,
        )

        self.product_a = np.zeros(
            n,
            dtype=np.uint16,
        )

        self.product_b = np.zeros(
            n,
            dtype=np.uint16,
        )

        self.rates = np.zeros(
            n,
            dtype=np.float32,
        )

        self.field_substrate = np.full(
            n,
            -1,
            dtype=np.int32,
        )

        self.field_product = np.full(
            n,
            -1,
            dtype=np.int32,
        )


        for index, reaction in enumerate(
            ordered
        ):

            self.types[
                index
            ] = reaction.reaction_type

            self.partners[
                index
            ] = reaction.partner_particle_id

            self.product_a[
                index
            ] = reaction.product_a_id

            self.product_b[
                index
            ] = reaction.product_b_id

            self.rates[
                index
            ] = reaction.rate

            self.field_substrate[
                index
            ] = reaction.field_substrate_id

            self.field_product[
                index
            ] = reaction.field_product_id


    @property
    def reaction_count(self):

        return len(
            self.ordered_definitions
        )


    def reactions_for_species(
        self,
        species_name,
    ):

        species = (
            self.species_registry.get(
                species_name
            )
        )

        if species.particle_id is None:

            return tuple()


        start = int(
            self.offsets[
                species.particle_id
            ]
        )

        end = int(
            self.offsets[
                species.particle_id + 1
            ]
        )


        return self.ordered_definitions[
            start:end
        ]


    def describe(self):

        print()
        print("REACTION REGISTRY")
        print("--------------------------------------------")

        print(
            f"Total reactions:            "
            f"{self.reaction_count:,}"
        )


        counts = {
            reaction_type: 0
            for reaction_type
            in ReactionType
        }


        for reaction in (
            self.ordered_definitions
        ):

            counts[
                reaction.reaction_type
            ] += 1


        for reaction_type in ReactionType:

            print(
                f"{reaction_type.name:26s}"
                f"{counts[reaction_type]:,}"
            )


        multifunction_species = 0


        for species in (
            self.species_registry
            .particle_species
        ):

            if (
                len(
                    self.reactions_for_species(
                        species.name
                    )
                )
                >= 2
            ):

                multifunction_species += 1


        print(
            f"Multi-function species:     "
            f"{multifunction_species:,}"
        )


class ReactionEngine:

    THREADS = 256


    def __init__(
        self,
        particle_state: ParticleState,
        field_state: FieldState,
        registry: ReactionRegistry,
        seed: int,
    ):

        self.particle_state = (
            particle_state
        )

        self.field_state = (
            field_state
        )

        self.registry = registry

        self.seed = int(
            seed
        )

        self.step_count = 0

        self.starting_particle_count = (
            particle_state
            .total_particle_count
        )


        if (
            field_state.values.dtype
            != cp.float32
        ):

            raise ValueError(
                "ReactionEngine currently requires "
                "float32 concentration fields."
            )


        self.offsets = cp.asarray(
            registry.offsets
        )

        self.types = cp.asarray(
            registry.types
        )

        self.partners = cp.asarray(
            registry.partners
        )

        self.product_a = cp.asarray(
            registry.product_a
        )

        self.product_b = cp.asarray(
            registry.product_b
        )

        self.rates = cp.asarray(
            registry.rates
        )

        self.field_substrate = cp.asarray(
            registry.field_substrate
        )

        self.field_product = cp.asarray(
            registry.field_product
        )


        self.event_counts = cp.zeros(
            (
                particle_state.cell_voxels,
                6,
            ),
            dtype=cp.uint64,
        )


        kernel_path = (
            Path(__file__).parent
            / "kernels"
            / "reactions.cu"
        )

        source = (
            kernel_path.read_text()
        )


        module = cp.RawModule(
            code=source,
            options=(
                "--std=c++11",
            ),
        )


        self._reaction_step = (
            module.get_function(
                "reaction_step"
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


    def step(
        self,
        dt_s: float,
    ):

        if dt_s <= 0:

            raise ValueError(
                "Reaction timestep must be > 0."
            )


        particles = (
            self.particle_state
        )


        self._reaction_step(
            self._blocks(
                particles.cell_voxels
            ),

            (
                self.THREADS,
            ),

            (
                particles.primary,
                particles.primary_occupancy,

                particles.spill,
                particles.spill_occupancy,

                self.field_state.values,

                self.offsets,

                self.types,

                self.partners,
                self.product_a,
                self.product_b,

                self.rates,

                self.field_substrate,
                self.field_product,

                self.event_counts.reshape(
                    -1
                ),

                np.int32(
                    particles.primary_slots
                ),

                np.int32(
                    particles.spill_slots
                ),

                np.int64(
                    particles.cell_voxels
                ),

                np.float32(
                    dt_s
                ),

                np.uint64(
                    self.seed
                ),

                np.uint64(
                    self.step_count
                ),
            ),
        )


        self.step_count += 1


    def run_steps(
        self,
        steps: int,
        dt_s: float,
    ):

        if steps < 0:

            raise ValueError(
                "steps must be >= 0"
            )


        for _ in range(
            steps
        ):

            self.step(
                dt_s
            )


    @property
    def event_totals(self):

        return cp.asnumpy(
            cp.sum(
                self.event_counts,
                axis=0,
            )
        )


    @property
    def degradation_events(self):

        return int(
            self.event_totals[0]
        )


    @property
    def conversion_events(self):

        return int(
            self.event_totals[1]
        )


    @property
    def catalytic_events(self):

        return int(
            self.event_totals[2]
        )


    @property
    def binding_events(self):

        return int(
            self.event_totals[3]
        )


    @property
    def unbinding_events(self):

        return int(
            self.event_totals[4]
        )


    @property
    def particle_field_flux_evaluations(
        self
    ):

        return int(
            self.event_totals[5]
        )


    @property
    def expected_particle_count(self):

        return (
            self.starting_particle_count
            - self.degradation_events
            - self.binding_events
            + self.unbinding_events
        )


    def validate(self):

        self.particle_state.validate()


        actual = (
            self.particle_state
            .total_particle_count
        )

        expected = (
            self.expected_particle_count
        )


        if actual != expected:

            raise RuntimeError(
                "Reaction particle accounting failed: "
                f"expected {expected:,}, "
                f"found {actual:,}."
            )


        return True


    def describe(self):

        events = (
            self.event_totals
        )

        print()
        print("REACTION ENGINE")
        print("--------------------------------------------")

        print(
            f"Completed reaction steps:   "
            f"{self.step_count:,}"
        )

        print(
            f"Degradation events:         "
            f"{int(events[0]):,}"
        )

        print(
            f"Conversion events:          "
            f"{int(events[1]):,}"
        )

        print(
            f"Catalytic events:           "
            f"{int(events[2]):,}"
        )

        print(
            f"Binding events:             "
            f"{int(events[3]):,}"
        )

        print(
            f"Unbinding events:           "
            f"{int(events[4]):,}"
        )

        print(
            f"Particle-field flux evals:  "
            f"{int(events[5]):,}"
        )

        print(
            f"Expected particles:         "
            f"{self.expected_particle_count:,}"
        )

        print(
            f"Actual particles:           "
            f"{self.particle_state.total_particle_count:,}"
        )