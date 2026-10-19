import cupy as cp
import numpy as np

from chromosome.genome_loader import (
    GenomeLoader,
)
from chromosome.spatial import (
    ChromosomeSpatialState,
)
from expression.mrna import (
    MRNADegradationEngine,
    MRNAState,
)
from expression.protein_production import (
    ProteinProductionEngine,
)
from expression.promoters import (
    PromoterRegistry,
)
from expression.tf_binding import (
    TFBindingRegistry,
    TFPromoterBindingEngine,
)
from expression.transcription import (
    TranscriptionEngine,
    TranscriptionRegistry,
)
from expression.translation import (
    TranslationEngine,
    TranslationRegistry,
)
from spatial.config import (
    load_config,
)
from spatial.grid import (
    SpatialGrid,
)
from spatial.particles import (
    ParticleState,
)
from spatial.species import (
    SpeciesRegistry,
)
from spatial.voxel_counts import (
    VoxelCountState,
)


DT_S = 0.001


def build_expression_system(
    rnap_count=4,
    ribosome_count=4,
):

    config = load_config(
        "configs/default.yaml"
    )


    species = SpeciesRegistry.from_yaml(
        "tests/data/"
        "translation_species.yaml"
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    chromosome = GenomeLoader.from_genbank(
        "tests/data/"
        "synthetic_genome.gb"
    )


    chromosome_spatial = (
        ChromosomeSpatialState(
            chromosome=chromosome,

            grid=grid,

            spatial_config=(
                config.spatial
            ),

            chromosome_config=(
                config.chromosome
            ),
        )
    )


    promoters = PromoterRegistry.from_chromosome(
        chromosome=chromosome,

        spatial_state=(
            chromosome_spatial
        ),

        upstream_bp=(
            config.chromosome
            .promoter_upstream_bp
        ),
    )


    # ========================================================
    # Regulation
    # ========================================================

    tf_registry = TFBindingRegistry.from_yaml(
        path=(
            "tests/data/"
            "synthetic_regulation.yaml"
        ),

        promoters=promoters,

        species_registry=species,

        voxel_volume_liters=(
            config.spatial
            .voxel_volume_liters
        ),

        cell_voxels=(
            grid.cell_voxels
        ),
    )


    # ========================================================
    # Transcription definitions
    # ========================================================

    transcription_registry = (
        TranscriptionRegistry.from_yaml(
            path=(
                "tests/data/"
                "synthetic_transcription.yaml"
            ),

            chromosome=chromosome,

            chromosome_spatial=(
                chromosome_spatial
            ),

            promoters=promoters,

            species_registry=species,

            voxel_volume_liters=(
                config.spatial
                .voxel_volume_liters
            ),

            cell_voxels=(
                grid.cell_voxels
            ),

            tf_binding_registry=(
                tf_registry
            ),
        )
    )


    # ========================================================
    # Translation definitions
    # ========================================================

    translation_registry = (
        TranslationRegistry.from_yaml(
            path=(
                "tests/data/"
                "synthetic_translation.yaml"
            ),

            chromosome=chromosome,

            transcription_registry=(
                transcription_registry
            ),

            species_registry=species,

            voxel_volume_liters=(
                config.spatial
                .voxel_volume_liters
            ),

            cell_voxels=(
                grid.cell_voxels
            ),
        )
    )


    # ========================================================
    # Molecular state
    # ========================================================

    particles = ParticleState(
        grid=grid,

        registry=species,

        config=config.particles,

        seed=config.seed,
    )


    voxel_counts = VoxelCountState(
        grid=grid,

        registry=species,

        seed=config.seed,
    )


    mrna = MRNAState(
        template_count=(
            transcription_registry
            .template_count
        ),

        cell_voxels=(
            grid.cell_voxels
        ),

        max_molecules=(
            transcription_registry
            .max_mrna_molecules
        ),
    )


    # ========================================================
    # Remove randomized/default test initialization.
    # ========================================================

    particles.primary.fill(
        0
    )

    particles.spill.fill(
        0
    )

    particles.primary_occupancy.fill(
        0
    )

    particles.spill_occupancy.fill(
        0
    )


    voxel_counts.counts.fill(
        0
    )

    voxel_counts.counts_next.fill(
        0
    )


    # ========================================================
    # Focus the integrated test on TEST_0001.
    # ========================================================

    template_index = 0


    promoter_voxel = int(
        transcription_registry
        .templates[
            template_index
        ]
        .promoter_voxel_id
    )


    completion_voxel = int(
        transcription_registry
        .templates[
            template_index
        ]
        .completion_voxel_id
    )


    # ========================================================
    # Put RNAP directly at TEST_0001's promoter.
    # ========================================================

    rnap_id = species.get(
        "RNAP"
    ).particle_id


    if rnap_count > 0:

        particles.primary[
            promoter_voxel,
            :rnap_count
        ] = rnap_id


        particles.primary_occupancy[
            promoter_voxel
        ] = rnap_count


    # ========================================================
    # Put free ribosomes at the voxel where TEST_0001 mRNA
    # will be born.
    #
    # This deliberately validates LOCAL spatial handoff:
    #
    # gene 3' voxel
    #     → mRNA
    #     → ribosome
    #     → protein
    # ========================================================

    ribosome_id = (
        translation_registry
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        completion_voxel
    ] = ribosome_count


    # ========================================================
    # Engines
    # ========================================================

    tf_binding = TFPromoterBindingEngine(
        particle_state=particles,

        registry=tf_registry,

        seed=config.seed,
    )


    transcription = TranscriptionEngine(
        particle_state=particles,

        registry=transcription_registry,

        mrna_state=mrna,

        seed=config.seed,

        tf_binding_engine=(
            tf_binding
        ),
    )


    translation = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation_registry,

        seed=config.seed,
    )


    protein_production = ProteinProductionEngine(
        translation_engine=translation,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    degradation = MRNADegradationEngine(
        mrna_state=mrna,

        degradation_rates=(
            transcription_registry
            .mrna_degradation_rates
        ),

        seed=config.seed,
    )


    return {
        "config":
            config,

        "species":
            species,

        "grid":
            grid,

        "chromosome":
            chromosome,

        "transcription_registry":
            transcription_registry,

        "translation_registry":
            translation_registry,

        "particles":
            particles,

        "voxel_counts":
            voxel_counts,

        "mrna":
            mrna,

        "tf_binding":
            tf_binding,

        "transcription":
            transcription,

        "translation":
            translation,

        "protein_production":
            protein_production,

        "degradation":
            degradation,

        "template_index":
            template_index,

        "promoter_voxel":
            promoter_voxel,

        "completion_voxel":
            completion_voxel,

        "initial_rnap_count":
            rnap_count,

        "initial_ribosome_count":
            ribosome_count,
    }


def run_expression(
    system,
    steps,
    degrade=False,
):

    for _ in range(
        steps
    ):

        # DNA → mRNA

        system[
            "transcription"
        ].step(
            DT_S
        )


        # mRNA → translating ribosome

        system[
            "translation"
        ].step(
            DT_S
        )


        # completed translation → protein

        system[
            "protein_production"
        ].step()


        # Optional mRNA decay.

        if degrade:

            system[
                "degradation"
            ].step(
                DT_S
            )


def count_particle_species(
    particle_state,
    particle_id,
):

    primary = cp.count_nonzero(
        particle_state.primary
        == particle_id
    )


    spill = cp.count_nonzero(
        particle_state.spill
        == particle_id
    )


    return int(
        (
            primary
            + spill
        ).item()
    )


def count_particle_species_in_voxel(
    particle_state,
    voxel,
    particle_id,
):

    primary = cp.count_nonzero(
        particle_state.primary[
            voxel
        ]
        == particle_id
    )


    spill = cp.count_nonzero(
        particle_state.spill[
            voxel
        ]
        == particle_id
    )


    return int(
        (
            primary
            + spill
        ).item()
    )


def test_end_to_end_gene_expression_and_spatial_handoff():

    system = build_expression_system(
        rnap_count=4,
        ribosome_count=4,
    )


    transcription = system[
        "transcription"
    ]


    translation = system[
        "translation"
    ]


    production = system[
        "protein_production"
    ]


    mrna = system[
        "mrna"
    ]


    template_index = system[
        "template_index"
    ]


    completion_voxel = system[
        "completion_voxel"
    ]


    # ========================================================
    # Synthetic transcript is 30 bp and transcription
    # elongation is 1000 bp/s.
    #
    # Even an immediately initiated transcript therefore
    # cannot be complete after only 20 ms.
    # ========================================================

    run_expression(
        system,
        steps=20,
        degrade=False,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        transcription.completed_for_template(
            template_index
        )
        == 0
    )


    assert (
        production.produced_for_template(
            template_index
        )
        == 0
    )


    # ========================================================
    # Continue long enough for:
    #
    # transcription
    #     → mRNA
    #     → ribosome binding
    #     → translation
    #     → protein
    # ========================================================

    run_expression(
        system,
        steps=80,
        degrade=False,
    )


    cp.cuda.Stream.null.synchronize()


    completed_transcripts = (
        transcription
        .completed_for_template(
            template_index
        )
    )


    completed_translation = (
        translation
        .completed_for_template(
            template_index
        )
    )


    produced_proteins = (
        production
        .produced_for_template(
            template_index
        )
    )


    assert (
        completed_transcripts
        > 0
    )


    assert (
        completed_translation
        > 0
    )


    assert (
        produced_proteins
        > 0
    )


    # ========================================================
    # Every completed translation must have become a protein.
    # ========================================================

    assert (
        produced_proteins
        == completed_translation
    )


    assert (
        translation.pending_product_count
        == 0
    )


    # ========================================================
    # No degradation ran in this test, so every completed
    # transcript should still exist as an active mRNA.
    # ========================================================

    template_counts = (
        mrna.template_counts()
    )


    assert (
        int(
            template_counts[
                template_index
            ]
        )
        == completed_transcripts
    )


    # ========================================================
    # Spatial handoff:
    #
    # every TEST_0001 mRNA is born at the 3' completion voxel.
    # ========================================================

    mrna_voxels = (
        mrna.active_voxels(
            template_index=(
                template_index
            )
        )
    )


    assert (
        len(
            mrna_voxels
        )
        > 0
    )


    assert bool(
        np.all(
            mrna_voxels
            == completion_voxel
        )
    )


    # ========================================================
    # ProteinA is an explicit particle.
    #
    # Because we are not running protein diffusion here,
    # every ProteinA molecule must still be in the voxel where
    # translation completed.
    # ========================================================

    protein_a_id = system[
        "species"
    ].get(
        "ProteinA"
    ).particle_id


    total_protein_a = (
        count_particle_species(
            system[
                "particles"
            ],

            protein_a_id,
        )
    )


    local_protein_a = (
        count_particle_species_in_voxel(
            system[
                "particles"
            ],

            completion_voxel,

            protein_a_id,
        )
    )


    assert (
        total_protein_a
        == produced_proteins
    )


    assert (
        local_protein_a
        == total_protein_a
    )


    # ========================================================
    # RNAP is never created or destroyed by transcription.
    # It is either free or inside a transcription complex.
    # ========================================================

    assert (
        transcription.rnap_accounted_count
        == system[
            "initial_rnap_count"
        ]
    )


    # ========================================================
    # Ribosomes are likewise conserved:
    #
    # free + translating = initial total.
    # ========================================================

    assert (
        translation.ribosome_accounted_count
        == system[
            "initial_ribosome_count"
        ]
    )


    # ========================================================
    # No artificial storage bottleneck should have occurred.
    # ========================================================

    assert (
        transcription.pool_full_events
        == 0
    )


    assert (
        transcription.mrna_pool_full_events
        == 0
    )


    assert (
        translation.pool_full_events
        == 0
    )


    assert (
        translation.product_queue_full_events
        == 0
    )


    assert (
        production.particle_storage_block_events
        == 0
    )


    assert (
        production.voxel_count_overflow_events
        == 0
    )


    assert transcription.validate()

    assert translation.validate()

    assert production.validate()


def test_completed_transcripts_are_accounted_for_by_mrna_or_degradation():

    system = build_expression_system(
        rnap_count=6,
        ribosome_count=0,
    )


    # ========================================================
    # No ribosomes are present.
    #
    # Therefore mRNAs cannot become protected by translation
    # and are free to undergo their normal stochastic decay.
    # ========================================================

    run_expression(
        system,
        steps=500,
        degrade=True,
    )


    cp.cuda.Stream.null.synchronize()


    transcription = system[
        "transcription"
    ]


    degradation = system[
        "degradation"
    ]


    mrna = system[
        "mrna"
    ]


    template_index = system[
        "template_index"
    ]


    completed = (
        transcription
        .completed_for_template(
            template_index
        )
    )


    degraded = (
        degradation
        .degraded_for_template(
            template_index
        )
    )


    active = int(
        mrna.template_counts()[
            template_index
        ]
    )


    assert (
        completed
        > 0
    )


    assert (
        degraded
        > 0
    )


    # ========================================================
    # Every transcript ever completed must be in exactly one
    # of two places:
    #
    #     still active
    #     or
    #     already degraded
    # ========================================================

    assert (
        completed
        == active
        + degraded
    )


    assert (
        transcription.rnap_accounted_count
        == system[
            "initial_rnap_count"
        ]
    )


    assert degradation.validate()

    assert transcription.validate()


def test_tf_repression_propagates_all_the_way_to_protein_output():

    system = build_expression_system(
        rnap_count=4,
        ribosome_count=4,
    )


    tf_binding = system[
        "tf_binding"
    ]


    transcription = system[
        "transcription"
    ]


    production = system[
        "protein_production"
    ]


    template_index = system[
        "template_index"
    ]


    # ========================================================
    # The synthetic TF site belongs to TEST_0001.
    #
    # Force it into the bound state so this validation is
    # testing the regulatory consequence rather than the
    # stochastic chance of TF binding.
    # ========================================================

    tf_binding.bound[
        0
    ] = 1


    run_expression(
        system,
        steps=80,
        degrade=False,
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # Bound repressor:
    #
    # no TEST_0001 transcription
    # therefore no TEST_0001 protein.
    # ========================================================

    assert (
        transcription.completed_for_template(
            template_index
        )
        == 0
    )


    assert (
        production.produced_for_template(
            template_index
        )
        == 0
    )


    # ========================================================
    # Release repression.
    # ========================================================

    tf_binding.bound[
        0
    ] = 0


    run_expression(
        system,
        steps=100,
        degrade=False,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        transcription.completed_for_template(
            template_index
        )
        > 0
    )


    assert (
        production.produced_for_template(
            template_index
        )
        > 0
    )


    assert (
        transcription.rnap_accounted_count
        == system[
            "initial_rnap_count"
        ]
    )


    assert (
        system[
            "translation"
        ].ribosome_accounted_count
        == system[
            "initial_ribosome_count"
        ]
    )


    assert transcription.validate()

    assert system[
        "translation"
    ].validate()

    assert production.validate()
