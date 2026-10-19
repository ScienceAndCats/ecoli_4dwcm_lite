import cupy as cp
import pytest

from chromosome.genome_loader import (
    GenomeLoader,
)
from chromosome.spatial import (
    ChromosomeSpatialState,
)
from expression.mrna import (
    MRNAState,
)
from expression.protein_production import (
    ProteinProductionEngine,
)
from expression.promoters import (
    PromoterRegistry,
)
from expression.transcription import (
    TranscriptionRegistry,
)
from expression.translation import (
    PRODUCT_PARTICLE,
    PRODUCT_VOXEL_COUNT,
    TranslationEngine,
    TranslationRegistry,
)
from spatial.config import (
    load_config,
)
from spatial.crowding import (
    CrowdingState,
)
from spatial.crowding_sync import (
    DynamicCrowdingSynchronizer,
)
from spatial.effective_volumes import (
    EffectiveVolumeRegistry,
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


def build_context():

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


    transcription = (
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


    translation = (
        TranslationRegistry.from_yaml(
            path=(
                "tests/data/"
                "synthetic_translation.yaml"
            ),

            chromosome=chromosome,

            transcription_registry=(
                transcription
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


    mrna = MRNAState(
        template_count=(
            transcription.template_count
        ),

        cell_voxels=(
            grid.cell_voxels
        ),

        max_molecules=(
            transcription.max_mrna_molecules
        ),
    )


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


    # Controlled test states.

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


    return (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        particles,
        voxel_counts,
    )


def seed_one_mrna(
    mrna,
    transcription,
    template_index,
):

    voxel = (
        transcription.templates[
            template_index
        ].completion_voxel_id
    )


    mrna.active[
        0
    ] = 1


    mrna.template_index[
        0
    ] = template_index


    mrna.voxel_id[
        0
    ] = voxel


    mrna.bound_ribosomes[
        0
    ] = 0


    return voxel


def count_particle_species(
    particles,
    particle_id,
):

    primary = cp.count_nonzero(
        particles.primary
        == particle_id
    )


    spill = cp.count_nonzero(
        particles.spill
        == particle_id
    )


    return int(
        (
            primary
            + spill
        ).item()
    )


def test_registry_maps_translation_to_product_representations():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        particles,
        voxel_counts,
    ) = build_context()


    first = (
        translation.templates[
            0
        ]
    )


    second = (
        translation.templates[
            1
        ]
    )


    assert (
        first.product_species_name
        == "ProteinA"
    )


    assert (
        second.product_species_name
        == "ProteinB"
    )


    assert (
        translation
        .product_representation_codes[
            first.transcription_template_index
        ]
        == PRODUCT_PARTICLE
    )


    assert (
        translation
        .product_representation_codes[
            second.transcription_template_index
        ]
        == PRODUCT_VOXEL_COUNT
    )


def test_particle_protein_is_created_in_translation_voxel():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        particles,
        voxel_counts,
    ) = build_context()


    template_index = 0


    voxel = seed_one_mrna(
        mrna,
        transcription,
        template_index,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 3


    translation_engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,
    )


    production = ProteinProductionEngine(
        translation_engine=translation_engine,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    for _ in range(
        30
    ):

        translation_engine.step(
            0.001
        )

        production.step()


    cp.cuda.Stream.null.synchronize()


    product_id = species.get(
        "ProteinA"
    ).particle_id


    total_products = count_particle_species(
        particles,
        product_id,
    )


    local_products = int(
        cp.count_nonzero(
            particles.primary[
                voxel
            ]
            == product_id
        ).item()
        +
        cp.count_nonzero(
            particles.spill[
                voxel
            ]
            == product_id
        ).item()
    )


    assert (
        production.total_produced
        > 0
    )


    assert (
        total_products
        == production.produced_for_template(
            template_index
        )
    )


    assert (
        total_products
        == translation_engine.completed_for_template(
            template_index
        )
    )


    # Every protein was born where its ribosome finished.

    assert (
        local_products
        == total_products
    )


    assert (
        production.pending_product_count
        == 0
    )


    assert production.validate()


def test_voxel_count_protein_is_created_in_translation_voxel():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        particles,
        voxel_counts,
    ) = build_context()


    template_index = 1


    voxel = seed_one_mrna(
        mrna,
        transcription,
        template_index,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 3


    translation_engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,
    )


    production = ProteinProductionEngine(
        translation_engine=translation_engine,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    for _ in range(
        30
    ):

        translation_engine.step(
            0.001
        )

        production.step()


    cp.cuda.Stream.null.synchronize()


    product_id = species.get(
        "ProteinB"
    ).voxel_count_id


    total_products = int(
        cp.sum(
            voxel_counts.counts[
                product_id
            ],

            dtype=cp.uint64,
        ).item()
    )


    local_products = int(
        voxel_counts.counts[
            product_id,
            voxel
        ].item()
    )


    assert (
        total_products
        > 0
    )


    assert (
        total_products
        == production.produced_for_template(
            template_index
        )
    )


    assert (
        total_products
        == translation_engine.completed_for_template(
            template_index
        )
    )


    assert (
        local_products
        == total_products
    )


    assert (
        production.pending_product_count
        == 0
    )


    assert production.validate()


def test_completed_product_waits_until_production_consumes_it():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        particles,
        voxel_counts,
    ) = build_context()


    template_index = 0


    voxel = seed_one_mrna(
        mrna,
        transcription,
        template_index,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 1


    translation_engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,
    )


    # Stop new initiations once the first ribosome binds.

    translation_engine.step(
        0.001
    )


    translation_engine.enabled[
        template_index
    ] = 0


    translation_engine.run_steps(
        steps=10,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        translation_engine.completion_events
        == 1
    )


    assert (
        translation_engine.pending_product_count
        == 1
    )


    product_id = species.get(
        "ProteinA"
    ).particle_id


    assert (
        count_particle_species(
            particles,
            product_id,
        )
        == 0
    )


    production = ProteinProductionEngine(
        translation_engine=translation_engine,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    production.step()


    cp.cuda.Stream.null.synchronize()


    assert (
        translation_engine.pending_product_count
        == 0
    )


    assert (
        count_particle_species(
            particles,
            product_id,
        )
        == 1
    )


    assert (
        production.total_produced
        == 1
    )


def test_protein_production_updates_crowding_without_double_counting():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        particles,
        voxel_counts,
    ) = build_context()


    template_index = 0


    voxel = seed_one_mrna(
        mrna,
        transcription,
        template_index,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 1


    volumes = EffectiveVolumeRegistry.from_yaml(
        path=(
            "tests/data/"
            "translation_effective_volumes.yaml"
        ),

        species_registry=species,
    )


    crowding = CrowdingState(
        grid=grid,

        movement_exponent=(
            config.crowding
            .movement_exponent
        ),
    )


    synchronizer = DynamicCrowdingSynchronizer(
        crowding_state=crowding,

        effective_volumes=volumes,

        particle_state=particles,

        voxel_count_state=voxel_counts,
    )


    # One free ribosome:
    #
    # 8000 nm^3

    assert (
        synchronizer.dynamic_volume_nm3
        == pytest.approx(
            8000.0
        )
    )


    translation_engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,

        crowding_synchronizer=(
            synchronizer
        ),
    )


    production = ProteinProductionEngine(
        translation_engine=translation_engine,

        particle_state=particles,

        voxel_count_state=voxel_counts,

        crowding_synchronizer=(
            synchronizer
        ),
    )


    # First step binds the ribosome.

    translation_engine.step(
        0.001
    )


    production.step()


    cp.cuda.Stream.null.synchronize()


    assert (
        translation_engine.active_complex_count
        == 1
    )


    # Translation complex:
    #
    # 8500 nm^3

    assert (
        synchronizer.dynamic_volume_nm3
        == pytest.approx(
            8500.0
        )
    )


    # Prevent the mRNA from immediately starting another
    # translation after this one completes.

    translation_engine.enabled[
        template_index
    ] = 0


    for _ in range(
        10
    ):

        translation_engine.step(
            0.001
        )

        production.step()


    cp.cuda.Stream.null.synchronize()


    assert (
        translation_engine.active_complex_count
        == 0
    )


    assert (
        production.total_produced
        == 1
    )


    assert (
        translation_engine.free_ribosome_count
        == 1
    )


    # Completed state:
    #
    # ribosome = 8000
    # ProteinA =  500
    # ----------------
    # total    = 8500 nm^3
    #
    # The bound translation complex disappeared, so it is not
    # being counted in addition to the free ribosome + protein.

    assert (
        synchronizer.dynamic_volume_nm3
        == pytest.approx(
            8500.0
        )
    )


    assert synchronizer.validate()

    assert production.validate()
