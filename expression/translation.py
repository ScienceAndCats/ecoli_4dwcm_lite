from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import yaml

from chromosome.chromosome import (
    Chromosome,
)
from expression.mrna import (
    MRNAState,
)
from expression.transcription import (
    TranscriptionRegistry,
)
from spatial.species import (
    Representation,
    SpeciesRegistry,
)
from spatial.voxel_counts import (
    VoxelCountState,
)


AVOGADRO = 6.02214076e23


PRODUCT_PARTICLE = 1
PRODUCT_VOXEL_COUNT = 2


@dataclass(frozen=True)
class TranslationTemplate:

    transcription_template_index: int

    locus_tag: str
    gene_name: str | None

    protein_id: str | None
    product: str | None

    product_species_name: str
    product_representation: Representation
    product_numeric_id: int

    coding_length_nt: int
    protein_length_aa: int

    association_rate_per_ribosome_s: float
    elongation_aa_per_s: float

    max_ribosomes_per_mrna: int


class TranslationRegistry:

    def __init__(
        self,
        templates,
        transcription_template_count,
        ribosome_name,
        ribosome_voxel_count_id,
        max_active_complexes,
        cell_voxels,
        complex_volume_name,
        species_registry,
    ):

        self.templates = tuple(
            templates
        )


        self.transcription_template_count = int(
            transcription_template_count
        )


        self.ribosome_name = str(
            ribosome_name
        )


        self.ribosome_voxel_count_id = int(
            ribosome_voxel_count_id
        )


        self.max_active_complexes = int(
            max_active_complexes
        )


        self.cell_voxels = int(
            cell_voxels
        )


        self.complex_volume_name = str(
            complex_volume_name
        )


        self.species_registry = (
            species_registry
        )


        self._compile()


    @classmethod
    def from_yaml(
        cls,
        path,
        chromosome: Chromosome,
        transcription_registry: TranscriptionRegistry,
        species_registry: SpeciesRegistry,
        voxel_volume_liters: float,
        cell_voxels: int,
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


        section = raw[
            "translation"
        ]


        # ====================================================
        # Ribosome species
        # ====================================================

        ribosome_name = str(
            section[
                "ribosome_species"
            ]
        )


        ribosome = species_registry.get(
            ribosome_name
        )


        if (
            ribosome.representation
            is not Representation.VOXEL_COUNT
        ):

            raise ValueError(
                f"Ribosome species {ribosome_name} "
                "must use voxel_count representation."
            )


        if ribosome.voxel_count_id is None:

            raise RuntimeError(
                f"Ribosome species {ribosome_name} "
                "has no voxel-count ID."
            )


        # ====================================================
        # Global parameters
        # ====================================================

        max_active_complexes = int(
            section[
                "max_active_complexes"
            ]
        )


        if max_active_complexes <= 0:

            raise ValueError(
                "max_active_complexes must be > 0."
            )


        complex_volume_name = str(
            section[
                "complex_volume_name"
            ]
        )


        default_binding_rate = float(
            section[
                "default_ribosome_binding_rate_M_inv_s"
            ]
        )


        default_elongation = float(
            section[
                "default_elongation_aa_per_s"
            ]
        )


        footprint_nt = int(
            section[
                "ribosome_footprint_nt"
            ]
        )


        if default_binding_rate < 0:

            raise ValueError(
                "Ribosome binding rate cannot be negative."
            )


        if default_elongation <= 0:

            raise ValueError(
                "Translation elongation rate must be > 0."
            )


        if footprint_nt <= 0:

            raise ValueError(
                "ribosome_footprint_nt must be > 0."
            )


        if voxel_volume_liters <= 0:

            raise ValueError(
                "voxel_volume_liters must be > 0."
            )


        # ====================================================
        # Transcription lookup
        # ====================================================

        transcription_by_locus = {}


        for index, template in enumerate(
            transcription_registry.templates
        ):

            transcription_by_locus[
                template.locus_tag
            ] = (
                index,
                template,
            )


        # ====================================================
        # CDS lookup
        # ====================================================

        cds_by_locus = {}


        for cds in (
            chromosome.coding_sequences
        ):

            if cds.locus_tag is None:

                continue


            cds_by_locus.setdefault(
                cds.locus_tag,
                [],
            ).append(
                cds
            )


        # ====================================================
        # Translation templates
        # ====================================================

        templates = []

        seen_loci = set()


        for item in section.get(
            "templates",
            [],
        ):

            locus_tag = str(
                item[
                    "locus_tag"
                ]
            )


            if locus_tag in seen_loci:

                raise ValueError(
                    f"Duplicate translation template: "
                    f"{locus_tag}"
                )


            seen_loci.add(
                locus_tag
            )


            if locus_tag not in transcription_by_locus:

                raise KeyError(
                    f"Translation locus {locus_tag} "
                    "is not a transcription template."
                )


            if locus_tag not in cds_by_locus:

                raise KeyError(
                    f"No CDS annotation found for "
                    f"{locus_tag}."
                )


            cds_matches = (
                cds_by_locus[
                    locus_tag
                ]
            )


            if len(
                cds_matches
            ) != 1:

                raise ValueError(
                    f"Expected exactly one CDS for "
                    f"{locus_tag}, found "
                    f"{len(cds_matches)}."
                )


            cds = (
                cds_matches[
                    0
                ]
            )


            (
                transcription_index,
                transcription_template,
            ) = transcription_by_locus[
                locus_tag
            ]


            # =================================================
            # Product species
            # =================================================

            product_species_name = str(
                item[
                    "product_species"
                ]
            )


            product_species = species_registry.get(
                product_species_name
            )


            if (
                product_species.representation
                is Representation.FIELD
            ):

                raise ValueError(
                    f"{locus_tag}: translated protein "
                    f"{product_species_name} cannot use "
                    "continuous field representation."
                )


            if (
                product_species.representation
                is Representation.PARTICLE
            ):

                if product_species.particle_id is None:

                    raise RuntimeError(
                        f"{product_species_name} has no "
                        "particle ID."
                    )


                product_numeric_id = int(
                    product_species.particle_id
                )


            elif (
                product_species.representation
                is Representation.VOXEL_COUNT
            ):

                if product_species.voxel_count_id is None:

                    raise RuntimeError(
                        f"{product_species_name} has no "
                        "voxel-count ID."
                    )


                product_numeric_id = int(
                    product_species.voxel_count_id
                )


            else:

                raise ValueError(
                    f"Unsupported product representation "
                    f"for {product_species_name}."
                )


            # =================================================
            # Protein properties
            # =================================================

            coding_length_nt = int(
                cds.length_bp
            )


            default_protein_length = max(
                1,
                coding_length_nt
                // 3,
            )


            protein_length_aa = int(
                item.get(
                    "protein_length_aa",
                    default_protein_length,
                )
            )


            if protein_length_aa <= 0:

                raise ValueError(
                    f"{locus_tag}: protein length "
                    "must be > 0."
                )


            binding_rate_M_inv_s = float(
                item.get(
                    "ribosome_binding_rate_M_inv_s",
                    default_binding_rate,
                )
            )


            if binding_rate_M_inv_s < 0:

                raise ValueError(
                    f"{locus_tag}: ribosome binding "
                    "rate cannot be negative."
                )


            association_rate = (
                binding_rate_M_inv_s
                /
                (
                    AVOGADRO
                    * voxel_volume_liters
                )
            )


            elongation = float(
                item.get(
                    "elongation_aa_per_s",
                    default_elongation,
                )
            )


            if elongation <= 0:

                raise ValueError(
                    f"{locus_tag}: elongation rate "
                    "must be > 0."
                )


            default_max_ribosomes = max(
                1,
                coding_length_nt
                // footprint_nt,
            )


            max_ribosomes = int(
                item.get(
                    "max_ribosomes_per_mrna",
                    default_max_ribosomes,
                )
            )


            if max_ribosomes <= 0:

                raise ValueError(
                    f"{locus_tag}: maximum ribosomes "
                    "must be > 0."
                )


            templates.append(
                TranslationTemplate(
                    transcription_template_index=(
                        transcription_index
                    ),

                    locus_tag=locus_tag,

                    gene_name=(
                        transcription_template
                        .gene_name
                    ),

                    protein_id=(
                        cds.protein_id
                    ),

                    product=(
                        cds.product
                    ),

                    product_species_name=(
                        product_species_name
                    ),

                    product_representation=(
                        product_species
                        .representation
                    ),

                    product_numeric_id=(
                        product_numeric_id
                    ),

                    coding_length_nt=(
                        coding_length_nt
                    ),

                    protein_length_aa=(
                        protein_length_aa
                    ),

                    association_rate_per_ribosome_s=float(
                        association_rate
                    ),

                    elongation_aa_per_s=(
                        elongation
                    ),

                    max_ribosomes_per_mrna=(
                        max_ribosomes
                    ),
                )
            )


        return cls(
            templates=templates,

            transcription_template_count=(
                transcription_registry
                .template_count
            ),

            ribosome_name=ribosome_name,

            ribosome_voxel_count_id=(
                ribosome.voxel_count_id
            ),

            max_active_complexes=(
                max_active_complexes
            ),

            cell_voxels=cell_voxels,

            complex_volume_name=(
                complex_volume_name
            ),

            species_registry=species_registry,
        )


    # ========================================================
    # Compile GPU-friendly arrays
    # ========================================================

    def _compile(self):

        n = (
            self.transcription_template_count
        )


        self.enabled = np.zeros(
            n,
            dtype=np.uint8,
        )


        self.association_rates = np.zeros(
            n,
            dtype=np.float32,
        )


        self.protein_lengths_aa = np.zeros(
            n,
            dtype=np.float32,
        )


        self.elongation_rates = np.zeros(
            n,
            dtype=np.float32,
        )


        self.max_ribosomes_per_mrna = np.zeros(
            n,
            dtype=np.uint32,
        )


        self.product_representation_codes = np.zeros(
            n,
            dtype=np.uint8,
        )


        self.product_numeric_ids = np.full(
            n,
            -1,
            dtype=np.int32,
        )


        for template in (
            self.templates
        ):

            index = (
                template
                .transcription_template_index
            )


            self.enabled[
                index
            ] = 1


            self.association_rates[
                index
            ] = (
                template
                .association_rate_per_ribosome_s
            )


            self.protein_lengths_aa[
                index
            ] = (
                template
                .protein_length_aa
            )


            self.elongation_rates[
                index
            ] = (
                template
                .elongation_aa_per_s
            )


            self.max_ribosomes_per_mrna[
                index
            ] = (
                template
                .max_ribosomes_per_mrna
            )


            if (
                template.product_representation
                is Representation.PARTICLE
            ):

                code = (
                    PRODUCT_PARTICLE
                )


            elif (
                template.product_representation
                is Representation.VOXEL_COUNT
            ):

                code = (
                    PRODUCT_VOXEL_COUNT
                )


            else:

                raise RuntimeError(
                    "Unsupported compiled product "
                    "representation."
                )


            self.product_representation_codes[
                index
            ] = code


            self.product_numeric_ids[
                index
            ] = (
                template
                .product_numeric_id
            )


    @property
    def template_count(self):

        return len(
            self.templates
        )


    def describe(self):

        print()
        print("TRANSLATION REGISTRY")
        print("--------------------------------------------")


        print(
            f"Ribosome species:           "
            f"{self.ribosome_name}"
        )


        print(
            f"Translation templates:      "
            f"{self.template_count:,}"
        )


        print(
            f"Maximum active complexes:   "
            f"{self.max_active_complexes:,}"
        )


        print(
            f"Complex volume name:        "
            f"{self.complex_volume_name}"
        )


        for template in (
            self.templates
        ):

            print(
                f"{template.locus_tag:10s} "
                f"{template.protein_length_aa:6,d} aa  "
                f"{template.product_species_name:20s} "
                f"{template.product_representation.value}"
            )


class TranslationEngine:

    THREADS = 256


    def __init__(
        self,
        voxel_count_state: VoxelCountState,
        mrna_state: MRNAState,
        registry: TranslationRegistry,
        seed: int,
        crowding_synchronizer=None,
    ):

        self.voxel_count_state = (
            voxel_count_state
        )


        self.mrna_state = (
            mrna_state
        )


        self.registry = (
            registry
        )


        self.seed = int(
            seed
        )


        self.step_count = 0


        if (
            voxel_count_state.registry
            is not registry.species_registry
        ):

            raise ValueError(
                "Translation registry and voxel-count "
                "state do not share the same "
                "SpeciesRegistry."
            )


        if (
            mrna_state.template_count
            != registry.transcription_template_count
        ):

            raise ValueError(
                "mRNA template count does not match "
                "translation registry."
            )


        if (
            mrna_state.cell_voxels
            != registry.cell_voxels
        ):

            raise ValueError(
                "mRNA state and translation registry "
                "use different spatial grids."
            )


        self.ribosome_counts = (
            voxel_count_state.counts[
                registry.ribosome_voxel_count_id
            ]
        )


        # ====================================================
        # Active translation complexes
        # ====================================================

        maximum = (
            registry.max_active_complexes
        )


        # 0 = free
        # 1 = active
        # 2 = temporarily reserved

        self.complex_active = cp.zeros(
            maximum,
            dtype=cp.uint32,
        )


        self.complex_mrna_index = cp.full(
            maximum,
            -1,
            dtype=cp.int32,
        )


        self.complex_template_index = cp.full(
            maximum,
            -1,
            dtype=cp.int32,
        )


        self.complex_voxel_id = cp.full(
            maximum,
            -1,
            dtype=cp.int32,
        )


        self.complex_remaining_aa = cp.zeros(
            maximum,
            dtype=cp.float32,
        )


        # ====================================================
        # Completed-protein event queue
        #
        # A translation complex cannot disappear until it has
        # successfully reserved a product-event slot.
        #
        # This guarantees completed proteins are never silently
        # lost even if protein production is delayed.
        # ====================================================

        self.product_event_active = cp.zeros(
            maximum,
            dtype=cp.uint32,
        )


        self.product_event_template_index = cp.full(
            maximum,
            -1,
            dtype=cp.int32,
        )


        self.product_event_voxel_id = cp.full(
            maximum,
            -1,
            dtype=cp.int32,
        )


        # ====================================================
        # Registry arrays
        # ====================================================

        self.enabled = cp.asarray(
            registry.enabled
        )


        self.association_rates = cp.asarray(
            registry.association_rates
        )


        self.protein_lengths_aa = cp.asarray(
            registry.protein_lengths_aa
        )


        self.elongation_rates = cp.asarray(
            registry.elongation_rates
        )


        self.max_ribosomes_per_mrna = cp.asarray(
            registry.max_ribosomes_per_mrna
        )


        # ====================================================
        # Diagnostics
        # ====================================================

        self.initiated_by_template = cp.zeros(
            registry.transcription_template_count,
            dtype=cp.uint64,
        )


        self.completed_by_template = cp.zeros(
            registry.transcription_template_count,
            dtype=cp.uint64,
        )


        # 0 initiation
        # 1 completion
        # 2 translation-complex pool full
        # 3 mRNA ribosome-capacity reached
        # 4 completed-protein event queue full

        self.event_counts = cp.zeros(
            5,
            dtype=cp.uint64,
        )


        # ====================================================
        # Crowding integration
        # ====================================================

        self.crowding_synchronizer = (
            crowding_synchronizer
        )


        self.translation_complex_volume_nm3 = 0.0


        if (
            crowding_synchronizer
            is not None
        ):

            if (
                crowding_synchronizer
                .voxel_count_state
                is not None
                and
                crowding_synchronizer
                .voxel_count_state
                is not voxel_count_state
            ):

                raise ValueError(
                    "Translation and crowding synchronizer "
                    "use different VoxelCountState objects."
                )


            self.translation_complex_volume_nm3 = float(
                crowding_synchronizer
                .effective_volumes
                .complex_volume_nm3(
                    registry.complex_volume_name
                )
            )


            crowding_synchronizer.register_volume_provider(
                "translation_complexes",
                self.crowding_volume_by_voxel,
            )


        self.initial_ribosome_count = (
            self.ribosome_accounted_count
        )


        # ====================================================
        # CUDA
        # ====================================================

        kernel_path = (
            Path(__file__).parents[1]
            / "spatial"
            / "kernels"
            / "translation.cu"
        )


        module = cp.RawModule(
            code=kernel_path.read_text(),

            options=(
                "--std=c++11",
            ),
        )


        self._elongate = (
            module.get_function(
                "elongate_translation"
            )
        )


        self._initiate = (
            module.get_function(
                "initiate_translation"
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
    # Counts
    # ========================================================

    @property
    def free_ribosome_count(self):

        return int(
            cp.sum(
                self.ribosome_counts,
                dtype=cp.uint64,
            ).item()
        )


    @property
    def active_complex_count(self):

        return int(
            cp.count_nonzero(
                self.complex_active
                == 1
            ).item()
        )


    @property
    def pending_product_count(self):

        return int(
            cp.count_nonzero(
                self.product_event_active
                == 1
            ).item()
        )


    @property
    def ribosome_accounted_count(self):

        return (
            self.free_ribosome_count
            + self.active_complex_count
        )


    @property
    def initiation_events(self):

        return int(
            self.event_counts[
                0
            ].item()
        )


    @property
    def completion_events(self):

        return int(
            self.event_counts[
                1
            ].item()
        )


    @property
    def pool_full_events(self):

        return int(
            self.event_counts[
                2
            ].item()
        )


    @property
    def mrna_capacity_events(self):

        return int(
            self.event_counts[
                3
            ].item()
        )


    @property
    def product_queue_full_events(self):

        return int(
            self.event_counts[
                4
            ].item()
        )


    def completed_for_template(
        self,
        template_index,
    ):

        return int(
            self.completed_by_template[
                int(
                    template_index
                )
            ].item()
        )


    # ========================================================
    # Translation-complex crowding provider
    # ========================================================

    def crowding_volume_by_voxel(
        self,
    ):

        values = cp.zeros(
            self.registry.cell_voxels,
            dtype=cp.float32,
        )


        if (
            self.translation_complex_volume_nm3
            <= 0
        ):

            return values


        mask = (
            self.complex_active
            == 1
        )


        if not bool(
            cp.any(
                mask
            ).item()
        ):

            return values


        voxels = (
            self.complex_voxel_id[
                mask
            ]
        )


        counts = cp.bincount(
            voxels,
            minlength=(
                self.registry.cell_voxels
            ),
        ).astype(
            cp.float32
        )


        values[:] = (
            counts
            * self.translation_complex_volume_nm3
        )


        return values


    # ========================================================
    # One translation step
    # ========================================================

    def step(
        self,
        dt_s: float,
    ):

        if dt_s <= 0:

            raise ValueError(
                "Translation timestep must be > 0."
            )


        # ====================================================
        # 1. Elongation / completion
        # ====================================================

        self._elongate(
            self._blocks(
                self.registry
                .max_active_complexes
            ),

            (
                self.THREADS,
            ),

            (
                self.complex_active,
                self.complex_mrna_index,
                self.complex_template_index,
                self.complex_voxel_id,
                self.complex_remaining_aa,

                self.elongation_rates,

                self.ribosome_counts,

                self.mrna_state.bound_ribosomes,

                self.product_event_active,
                self.product_event_template_index,
                self.product_event_voxel_id,

                self.completed_by_template,
                self.event_counts,

                np.int32(
                    self.registry
                    .max_active_complexes
                ),

                np.float32(
                    dt_s
                ),
            ),
        )


        # ====================================================
        # 2. New local ribosome binding
        # ====================================================

        self._initiate(
            self._blocks(
                self.mrna_state
                .max_molecules
            ),

            (
                self.THREADS,
            ),

            (
                self.mrna_state.active,
                self.mrna_state.template_index,
                self.mrna_state.voxel_id,
                self.mrna_state.bound_ribosomes,

                self.ribosome_counts,

                self.enabled,
                self.association_rates,
                self.protein_lengths_aa,
                self.max_ribosomes_per_mrna,

                self.complex_active,
                self.complex_mrna_index,
                self.complex_template_index,
                self.complex_voxel_id,
                self.complex_remaining_aa,

                self.initiated_by_template,
                self.event_counts,

                np.int32(
                    self.mrna_state
                    .max_molecules
                ),

                np.int32(
                    self.registry
                    .max_active_complexes
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


        if (
            self.crowding_synchronizer
            is not None
        ):

            self.crowding_synchronizer.rebuild_dynamic()


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


    # ========================================================
    # Validation
    # ========================================================

    def validate(self):

        self.mrna_state.validate()


        self.voxel_count_state.validate(
            require_initial_totals=False
        )


        if (
            self.ribosome_accounted_count
            != self.initial_ribosome_count
        ):

            raise RuntimeError(
                "Ribosome conservation failure: "
                f"initial="
                f"{self.initial_ribosome_count:,}, "
                f"accounted="
                f"{self.ribosome_accounted_count:,}."
            )


        if bool(
            cp.any(
                self.complex_active
                > 1
            ).item()
        ):

            raise RuntimeError(
                "Persistent reserved translation "
                "complex detected."
            )


        active_complexes = (
            self.complex_active
            == 1
        )


        if bool(
            cp.any(
                self.complex_mrna_index[
                    active_complexes
                ]
                < 0
            ).item()
        ):

            raise RuntimeError(
                "Active translation complex has "
                "invalid mRNA reference."
            )


        if bool(
            cp.any(
                self.complex_voxel_id[
                    active_complexes
                ]
                < 0
            ).item()
        ):

            raise RuntimeError(
                "Active translation complex has "
                "invalid voxel."
            )


        if bool(
            cp.any(
                self.product_event_active
                > 1
            ).item()
        ):

            raise RuntimeError(
                "Persistent reserved product-event "
                "slot detected."
            )


        active_products = (
            self.product_event_active
            == 1
        )


        if bool(
            cp.any(
                self.product_event_template_index[
                    active_products
                ]
                < 0
            ).item()
        ):

            raise RuntimeError(
                "Pending protein product has "
                "invalid template."
            )


        if bool(
            cp.any(
                self.product_event_voxel_id[
                    active_products
                ]
                < 0
            ).item()
        ):

            raise RuntimeError(
                "Pending protein product has "
                "invalid voxel."
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
        print("TRANSLATION ENGINE")
        print("--------------------------------------------")


        print(
            f"Translation steps:          "
            f"{self.step_count:,}"
        )


        print(
            f"Free ribosomes:             "
            f"{self.free_ribosome_count:,}"
        )


        print(
            f"Bound ribosomes:            "
            f"{self.active_complex_count:,}"
        )


        print(
            f"Accounted ribosomes:        "
            f"{self.ribosome_accounted_count:,}"
        )


        print(
            f"Initiation events:          "
            f"{self.initiation_events:,}"
        )


        print(
            f"Completion events:          "
            f"{self.completion_events:,}"
        )


        print(
            f"Pending proteins:           "
            f"{self.pending_product_count:,}"
        )


        print(
            f"Product-queue-full events:  "
            f"{self.product_queue_full_events:,}"
        )