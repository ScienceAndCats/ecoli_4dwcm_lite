from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import yaml

from chromosome.chromosome import (
    Chromosome,
    Strand,
)
from chromosome.spatial import (
    ChromosomeSpatialState,
)
from expression.mrna import (
    MRNAState,
)
from expression.promoters import (
    PromoterRegistry,
)
from expression.tf_binding import (
    TFBindingRegistry,
    TFPromoterBindingEngine,
)
from spatial.particles import (
    ParticleState,
)
from spatial.species import (
    SpeciesRegistry,
)


AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class TranscriptionTemplate:

    locus_tag: str
    gene_name: str | None

    promoter_voxel_id: int
    completion_voxel_id: int

    strand: Strand

    transcript_length_bp: int

    rnap_particle_id: int

    association_rate_per_rnap_s: float

    elongation_bp_per_s: float

    mrna_degradation_rate_s: float


class TranscriptionRegistry:

    def __init__(
        self,
        templates,
        rnap_name,
        rnap_particle_id,
        max_active_complexes,
        max_mrna_molecules,
        cell_voxels,
        blocker_site_indices_by_template,
    ):

        self.templates = tuple(
            templates
        )

        self.rnap_name = str(
            rnap_name
        )

        self.rnap_particle_id = int(
            rnap_particle_id
        )

        self.max_active_complexes = int(
            max_active_complexes
        )

        self.max_mrna_molecules = int(
            max_mrna_molecules
        )

        self.cell_voxels = int(
            cell_voxels
        )

        self.blocker_site_indices_by_template = tuple(
            tuple(
                values
            )
            for values
            in blocker_site_indices_by_template
        )

        self._compile()


    @classmethod
    def from_yaml(
        cls,
        path,
        chromosome: Chromosome,
        chromosome_spatial: ChromosomeSpatialState,
        promoters: PromoterRegistry,
        species_registry: SpeciesRegistry,
        voxel_volume_liters: float,
        cell_voxels: int,
        tf_binding_registry: TFBindingRegistry | None = None,
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
            "transcription"
        ]


        rnap_name = str(
            section[
                "rnap_species"
            ]
        )


        rnap_species = (
            species_registry.get(
                rnap_name
            )
        )


        if rnap_species.particle_id is None:

            raise ValueError(
                f"RNAP species {rnap_name} must be "
                "represented as particles."
            )


        max_active_complexes = int(
            section[
                "max_active_complexes"
            ]
        )


        max_mrna_molecules = int(
            section[
                "max_mrna_molecules"
            ]
        )


        if max_active_complexes <= 0:

            raise ValueError(
                "max_active_complexes must be > 0"
            )


        if max_mrna_molecules <= 0:

            raise ValueError(
                "max_mrna_molecules must be > 0"
            )


        default_elongation = float(
            section[
                "default_elongation_bp_per_s"
            ]
        )


        default_half_life = float(
            section[
                "default_mrna_half_life_s"
            ]
        )


        if default_elongation <= 0:

            raise ValueError(
                "default_elongation_bp_per_s "
                "must be > 0"
            )


        if default_half_life <= 0:

            raise ValueError(
                "default_mrna_half_life_s "
                "must be > 0"
            )


        genes_by_locus = {
            gene.locus_tag: gene
            for gene
            in chromosome.genes
            if gene.locus_tag is not None
        }


        blocker_sites_by_locus = {}


        if tf_binding_registry is not None:

            for site_index, site in enumerate(
                tf_binding_registry.sites
            ):

                blocker_sites_by_locus.setdefault(
                    site.promoter_locus_tag,
                    [],
                ).append(
                    site_index
                )


        templates = []
        blockers = []


        for item in section.get(
            "templates",
            [],
        ):

            locus_tag = str(
                item[
                    "locus_tag"
                ]
            )


            if locus_tag not in genes_by_locus:

                raise KeyError(
                    f"Gene {locus_tag} not found "
                    "in chromosome annotation."
                )


            gene = genes_by_locus[
                locus_tag
            ]


            promoter = (
                promoters.get_by_locus_tag(
                    locus_tag
                )
            )


            macroscopic_rate = float(
                item[
                    "rnap_association_rate_M_inv_s"
                ]
            )


            association_rate = (
                macroscopic_rate
                /
                (
                    AVOGADRO
                    * voxel_volume_liters
                )
            )


            elongation_rate = float(
                item.get(
                    "elongation_bp_per_s",
                    default_elongation,
                )
            )


            half_life_s = float(
                item.get(
                    "mrna_half_life_s",
                    default_half_life,
                )
            )


            if association_rate < 0:

                raise ValueError(
                    "RNAP association rate "
                    "cannot be negative."
                )


            if elongation_rate <= 0:

                raise ValueError(
                    "Elongation rate must be > 0."
                )


            if half_life_s <= 0:

                raise ValueError(
                    "mRNA half-life must be > 0."
                )


            degradation_rate = (
                np.log(
                    2.0
                )
                / half_life_s
            )


            # =================================================
            # 3' END OF THE GENE
            #
            # Forward strand:
            #
            # start ----------------------> end
            #                              ^
            #                              3'
            #
            # Reverse strand:
            #
            # start <---------------------- end
            #   ^
            #   3'
            # =================================================

            if gene.strand == Strand.FORWARD:

                completion_bp = (
                    gene.end_bp
                    - 1
                )

            else:

                completion_bp = (
                    gene.start_bp
                )


            completion_voxel = (
                chromosome_spatial
                .voxel_for_bp(
                    completion_bp
                )
            )


            templates.append(
                TranscriptionTemplate(
                    locus_tag=locus_tag,

                    gene_name=(
                        gene.gene_name
                    ),

                    promoter_voxel_id=(
                        promoter.voxel_id
                    ),

                    completion_voxel_id=(
                        completion_voxel
                    ),

                    strand=(
                        gene.strand
                    ),

                    transcript_length_bp=(
                        gene.length_bp
                    ),

                    rnap_particle_id=(
                        rnap_species.particle_id
                    ),

                    association_rate_per_rnap_s=(
                        association_rate
                    ),

                    elongation_bp_per_s=(
                        elongation_rate
                    ),

                    mrna_degradation_rate_s=float(
                        degradation_rate
                    ),
                )
            )


            blockers.append(
                blocker_sites_by_locus.get(
                    locus_tag,
                    [],
                )
            )


        return cls(
            templates=templates,

            rnap_name=rnap_name,

            rnap_particle_id=(
                rnap_species.particle_id
            ),

            max_active_complexes=(
                max_active_complexes
            ),

            max_mrna_molecules=(
                max_mrna_molecules
            ),

            cell_voxels=cell_voxels,

            blocker_site_indices_by_template=(
                blockers
            ),
        )


    def _compile(self):

        template_count = len(
            self.templates
        )


        self.promoter_voxels = np.zeros(
            template_count,
            dtype=np.int32,
        )


        self.completion_voxels = np.zeros(
            template_count,
            dtype=np.int32,
        )


        self.rnap_species = np.zeros(
            template_count,
            dtype=np.uint16,
        )


        self.association_rates = np.zeros(
            template_count,
            dtype=np.float32,
        )


        self.transcript_lengths = np.zeros(
            template_count,
            dtype=np.float32,
        )


        self.elongation_rates = np.zeros(
            template_count,
            dtype=np.float32,
        )


        self.mrna_degradation_rates = np.zeros(
            template_count,
            dtype=np.float32,
        )


        templates_by_voxel = [
            []
            for _ in range(
                self.cell_voxels
            )
        ]


        for index, template in enumerate(
            self.templates
        ):

            self.promoter_voxels[
                index
            ] = (
                template.promoter_voxel_id
            )


            self.completion_voxels[
                index
            ] = (
                template.completion_voxel_id
            )


            self.rnap_species[
                index
            ] = (
                template.rnap_particle_id
            )


            self.association_rates[
                index
            ] = (
                template.association_rate_per_rnap_s
            )


            self.transcript_lengths[
                index
            ] = (
                template.transcript_length_bp
            )


            self.elongation_rates[
                index
            ] = (
                template.elongation_bp_per_s
            )


            self.mrna_degradation_rates[
                index
            ] = (
                template.mrna_degradation_rate_s
            )


            templates_by_voxel[
                template.promoter_voxel_id
            ].append(
                index
            )


        self.voxel_offsets = np.zeros(
            self.cell_voxels + 1,
            dtype=np.int32,
        )


        ordered_templates = []


        for voxel in range(
            self.cell_voxels
        ):

            self.voxel_offsets[
                voxel
            ] = len(
                ordered_templates
            )


            ordered_templates.extend(
                templates_by_voxel[
                    voxel
                ]
            )


        self.voxel_offsets[
            self.cell_voxels
        ] = len(
            ordered_templates
        )


        self.template_indices = np.asarray(
            ordered_templates,
            dtype=np.int32,
        )


        self.blocker_offsets = np.zeros(
            template_count + 1,
            dtype=np.int32,
        )


        ordered_blockers = []


        for template_index in range(
            template_count
        ):

            self.blocker_offsets[
                template_index
            ] = len(
                ordered_blockers
            )


            ordered_blockers.extend(
                self.blocker_site_indices_by_template[
                    template_index
                ]
            )


        self.blocker_offsets[
            template_count
        ] = len(
            ordered_blockers
        )


        self.blocker_site_indices = np.asarray(
            ordered_blockers,
            dtype=np.int32,
        )


    @property
    def template_count(self):

        return len(
            self.templates
        )


    def describe(self):

        print()
        print("TRANSCRIPTION REGISTRY")
        print("--------------------------------------------")


        print(
            f"RNAP species:               "
            f"{self.rnap_name}"
        )


        print(
            f"Templates:                  "
            f"{self.template_count:,}"
        )


        print(
            f"Maximum active RNAP:        "
            f"{self.max_active_complexes:,}"
        )


        print(
            f"Maximum mRNAs:              "
            f"{self.max_mrna_molecules:,}"
        )


        for template in self.templates:

            print(
                f"{template.locus_tag:10s} "
                f"{template.gene_name or '-':10s} "
                f"{template.transcript_length_bp:6,d} bp  "
                f"{template.elongation_bp_per_s:6.1f} bp/s  "
                f"3' voxel={template.completion_voxel_id}"
            )


class TranscriptionEngine:

    THREADS = 256


    def __init__(
        self,
        particle_state: ParticleState,
        registry: TranscriptionRegistry,
        mrna_state: MRNAState,
        seed: int,
        tf_binding_engine: TFPromoterBindingEngine | None = None,
    ):

        self.particle_state = (
            particle_state
        )

        self.registry = registry

        self.mrna_state = (
            mrna_state
        )

        self.seed = int(
            seed
        )

        self.step_count = 0


        if (
            mrna_state.template_count
            != registry.template_count
        ):

            raise ValueError(
                "mRNA state template count does not "
                "match transcription registry."
            )


        self.tf_binding_engine = (
            tf_binding_engine
        )


        self.complex_active = cp.zeros(
            registry.max_active_complexes,
            dtype=cp.uint32,
        )


        self.complex_template = cp.full(
            registry.max_active_complexes,
            -1,
            dtype=cp.int32,
        )


        self.complex_remaining_bp = cp.zeros(
            registry.max_active_complexes,
            dtype=cp.float32,
        )


        self.completed_transcripts = cp.zeros(
            registry.template_count,
            dtype=cp.uint64,
        )


        # 0 initiation
        # 1 completed transcript + mRNA created
        # 2 RNAP release blocked
        # 3 transcription-complex pool full
        # 4 mRNA pool full

        self.event_counts = cp.zeros(
            5,
            dtype=cp.uint64,
        )


        self.voxel_offsets = cp.asarray(
            registry.voxel_offsets
        )


        self.template_indices = cp.asarray(
            registry.template_indices
        )


        self.completion_voxels = cp.asarray(
            registry.completion_voxels
        )


        self.rnap_species = cp.asarray(
            registry.rnap_species
        )


        self.association_rates = cp.asarray(
            registry.association_rates
        )


        self.transcript_lengths = cp.asarray(
            registry.transcript_lengths
        )


        self.elongation_rates = cp.asarray(
            registry.elongation_rates
        )


        self.blocker_offsets = cp.asarray(
            registry.blocker_offsets
        )


        self.blocker_site_indices = cp.asarray(
            registry.blocker_site_indices
        )


        if tf_binding_engine is None:

            self.tf_bound = cp.zeros(
                1,
                dtype=cp.uint8,
            )

        else:

            self.tf_bound = (
                tf_binding_engine.bound
            )


        self.initial_rnap_count = (
            self.free_rnap_count
        )


        kernel_path = (
            Path(__file__).parents[1]
            / "spatial"
            / "kernels"
            / "transcription.cu"
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


        self._elongate = (
            module.get_function(
                "elongate_transcription"
            )
        )


        self._initiate = (
            module.get_function(
                "initiate_transcription"
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


    @property
    def free_rnap_count(self):

        rnap_id = (
            self.registry.rnap_particle_id
        )


        primary = cp.count_nonzero(
            self.particle_state.primary
            == rnap_id
        )


        spill = cp.count_nonzero(
            self.particle_state.spill
            == rnap_id
        )


        return int(
            (
                primary
                + spill
            ).item()
        )


    @property
    def active_complex_count(self):

        return int(
            cp.sum(
                self.complex_active
            ).item()
        )


    @property
    def rnap_accounted_count(self):

        return (
            self.free_rnap_count
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
    def blocked_release_events(self):

        return int(
            self.event_counts[
                2
            ].item()
        )


    @property
    def pool_full_events(self):

        return int(
            self.event_counts[
                3
            ].item()
        )


    @property
    def mrna_pool_full_events(self):

        return int(
            self.event_counts[
                4
            ].item()
        )


    def step(
        self,
        dt_s: float,
    ):

        if dt_s <= 0:

            raise ValueError(
                "Transcription timestep must be > 0."
            )


        state = (
            self.particle_state
        )


        self._elongate(
            self._blocks(
                self.registry
                .max_active_complexes
            ),

            (
                self.THREADS,
            ),

            (
                state.primary,
                state.primary_occupancy,

                state.spill,
                state.spill_occupancy,

                self.complex_active,
                self.complex_template,
                self.complex_remaining_bp,

                self.completion_voxels,
                self.rnap_species,
                self.elongation_rates,

                self.mrna_state.active,
                self.mrna_state.template_index,
                self.mrna_state.voxel_id,

                np.int32(
                    self.mrna_state
                    .max_molecules
                ),

                self.completed_transcripts,
                self.event_counts,

                np.int32(
                    state.primary_slots
                ),

                np.int32(
                    state.spill_slots
                ),

                np.int32(
                    self.registry
                    .max_active_complexes
                ),

                np.float32(
                    dt_s
                ),

                np.uint64(
                    self.step_count
                ),
            ),
        )


        self._initiate(
            self._blocks(
                state.cell_voxels
            ),

            (
                self.THREADS,
            ),

            (
                state.primary,
                state.primary_occupancy,

                state.spill,
                state.spill_occupancy,

                self.voxel_offsets,
                self.template_indices,

                self.association_rates,
                self.transcript_lengths,
                self.rnap_species,

                self.blocker_offsets,
                self.blocker_site_indices,
                self.tf_bound,

                self.complex_active,
                self.complex_template,
                self.complex_remaining_bp,

                self.event_counts,

                np.int32(
                    state.primary_slots
                ),

                np.int32(
                    state.spill_slots
                ),

                np.int64(
                    state.cell_voxels
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


    def completed_for_template(
        self,
        template_index,
    ):

        return int(
            self.completed_transcripts[
                template_index
            ].item()
        )


    def validate(self):

        self.particle_state.validate()
        self.mrna_state.validate()


        if (
            self.active_complex_count
            > self.registry.max_active_complexes
        ):

            raise RuntimeError(
                "Active transcription complex count "
                "exceeds capacity."
            )


        return True


    def describe(self):

        print()
        print("TRANSCRIPTION ENGINE")
        print("--------------------------------------------")


        print(
            f"Transcription steps:        "
            f"{self.step_count:,}"
        )


        print(
            f"Free RNAP:                  "
            f"{self.free_rnap_count:,}"
        )


        print(
            f"Elongating RNAP:            "
            f"{self.active_complex_count:,}"
        )


        print(
            f"RNAP accounted:             "
            f"{self.rnap_accounted_count:,}"
        )


        print(
            f"Initiations:                "
            f"{self.initiation_events:,}"
        )


        print(
            f"Completed transcripts:      "
            f"{self.completion_events:,}"
        )


        print(
            f"Active mRNAs:               "
            f"{self.mrna_state.active_count:,}"
        )


        print(
            f"Blocked RNAP releases:      "
            f"{self.blocked_release_events:,}"
        )


        print(
            f"Complex-pool-full events:   "
            f"{self.pool_full_events:,}"
        )


        print(
            f"mRNA-pool-full events:      "
            f"{self.mrna_pool_full_events:,}"
        )