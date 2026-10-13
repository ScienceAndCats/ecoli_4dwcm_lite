import cupy as cp

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


def build_spatial_genome(
    genome_path,
    config,
):

    chromosome = (
        GenomeLoader.from_genbank(
            genome_path
        )
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    chromosome_spatial = (
        ChromosomeSpatialState(
            chromosome=chromosome,
            grid=grid,
            spatial_config=config.spatial,
            chromosome_config=config.chromosome,
        )
    )


    promoters = (
        PromoterRegistry.from_chromosome(
            chromosome=chromosome,
            spatial_state=chromosome_spatial,
            upstream_bp=(
                config.chromosome
                .promoter_upstream_bp
            ),
        )
    )


    return (
        chromosome,
        grid,
        chromosome_spatial,
        promoters,
    )


def test_real_ecoli_transcription_templates_resolve():

    config = load_config(
        "configs/default.yaml"
    )


    species = SpeciesRegistry.from_yaml(
        config.biology.species_file
    )


    (
        chromosome,
        grid,
        chromosome_spatial,
        promoters,
    ) = build_spatial_genome(
        config.biology.genome_file,
        config,
    )


    tf_registry = (
        TFBindingRegistry.from_yaml(
            path=(
                config.biology
                .regulation_file
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
    )


    transcription = (
        TranscriptionRegistry.from_yaml(
            path=(
                config.biology
                .transcription_file
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


    assert (
        transcription.template_count
        == 2
    )


    assert (
        transcription.templates[
            0
        ].locus_tag
        == "b0002"
    )


    assert (
        transcription.templates[
            1
        ].locus_tag
        == "b0006"
    )


    assert (
        transcription.templates[
            0
        ].transcript_length_bp
        > 0
    )


def test_rnap_transcription_and_tf_blocking():

    config = load_config(
        "configs/default.yaml"
    )


    species = SpeciesRegistry.from_yaml(
        config.biology.species_file
    )


    (
        chromosome,
        grid,
        chromosome_spatial,
        promoters,
    ) = build_spatial_genome(
        "tests/data/synthetic_genome.gb",
        config,
    )


    tf_registry = (
        TFBindingRegistry.from_yaml(
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
    )


    state = ParticleState(
        grid=grid,
        registry=species,
        config=config.particles,
        seed=config.seed,
    )


    state.primary.fill(
        0
    )

    state.spill.fill(
        0
    )

    state.primary_occupancy.fill(
        0
    )

    state.spill_occupancy.fill(
        0
    )


    promoter = (
        promoters.get_by_locus_tag(
            "TEST_0001"
        )
    )


    rnap_id = (
        species.get(
            "RNAP"
        ).particle_id
    )


    state.primary[
        promoter.voxel_id,
        :20
    ] = rnap_id


    state.primary_occupancy[
        promoter.voxel_id
    ] = 20


    initial_rnap = 20


    tf_engine = (
        TFPromoterBindingEngine(
            particle_state=state,
            registry=tf_registry,
            seed=config.seed,
        )
    )


    registry = (
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


    mrna_state = MRNAState(
        template_count=(
            registry.template_count
        ),

        cell_voxels=(
            grid.cell_voxels
        ),

        max_molecules=(
            registry.max_mrna_molecules
        ),
    )


    transcription = (
        TranscriptionEngine(
            particle_state=state,
            registry=registry,
            mrna_state=mrna_state,
            seed=config.seed,
            tf_binding_engine=tf_engine,
        )
    )


    # ========================================================
    # Bound TF blocks initiation.
    # ========================================================

    tf_engine.bound[
        0
    ] = 1


    transcription.run_steps(
        steps=100,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        transcription.initiation_events
        == 0
    )


    # ========================================================
    # Release regulatory block.
    # ========================================================

    tf_engine.bound[
        0
    ] = 0


    transcription.run_steps(
        steps=300,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        transcription.initiation_events
        > 0
    )


    assert (
        transcription.completion_events
        > 0
    )


    assert (
        transcription.completed_for_template(
            0
        )
        > 0
    )


    # ========================================================
    # Every TEST_0001 mRNA must be born in the gene's
    # 3' completion voxel.
    # ========================================================

    expected_voxel = (
        registry.templates[
            0
        ].completion_voxel_id
    )


    observed_voxels = (
        mrna_state.active_voxels(
            template_index=0
        )
    )


    assert (
        len(
            observed_voxels
        )
        > 0
    )


    assert bool(
        (
            observed_voxels
            == expected_voxel
        ).all()
    )


    # ========================================================
    # RNAP conservation.
    # ========================================================

    assert (
        transcription.rnap_accounted_count
        == initial_rnap
    )


    assert (
        transcription.pool_full_events
        == 0
    )


    assert (
        transcription.mrna_pool_full_events
        == 0
    )


    assert transcription.validate()


def test_mrna_degradation():

    config = load_config(
        "configs/default.yaml"
    )


    species = SpeciesRegistry.from_yaml(
        config.biology.species_file
    )


    (
        chromosome,
        grid,
        chromosome_spatial,
        promoters,
    ) = build_spatial_genome(
        "tests/data/synthetic_genome.gb",
        config,
    )


    registry = (
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
        )
    )


    mrna_state = MRNAState(
        template_count=(
            registry.template_count
        ),

        cell_voxels=(
            grid.cell_voxels
        ),

        max_molecules=(
            registry.max_mrna_molecules
        ),
    )


    # ========================================================
    # Seed 20 artificial mRNAs at template 0's real 3' voxel.
    # ========================================================

    count = 20


    mrna_state.active[
        :count
    ] = 1


    mrna_state.template_index[
        :count
    ] = 0


    mrna_state.voxel_id[
        :count
    ] = (
        registry.templates[
            0
        ].completion_voxel_id
    )


    degradation = (
        MRNADegradationEngine(
            mrna_state=mrna_state,

            degradation_rates=(
                registry
                .mrna_degradation_rates
            ),

            seed=config.seed,
        )
    )


    degradation.run_steps(
        steps=500,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        degradation.degradation_events
        > 0
    )


    assert (
        mrna_state.active_count
        < count
    )


    assert (
        mrna_state.active_count
        + degradation.degradation_events
        == count
    )


    assert degradation.validate()