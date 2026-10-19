import cupy as cp
import pytest

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
from expression.transcription import (
    TranscriptionRegistry,
)
from expression.translation import (
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


    voxel_counts = VoxelCountState(
        grid=grid,

        registry=species,

        seed=config.seed,
    )


    # Controlled tests build ribosome distributions manually.

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
        voxel_counts,
    )


def seed_one_mrna(
    mrna,
    transcription,
    template_index=0,
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


def test_translation_registry_uses_cds_annotation():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        voxel_counts,
    ) = build_context()


    assert (
        translation.template_count
        == 2
    )


    first = (
        translation.templates[
            0
        ]
    )


    assert (
        first.locus_tag
        == "TEST_0001"
    )


    # Synthetic CDS is 30 nt.

    assert (
        first.coding_length_nt
        == 30
    )


    assert (
        first.protein_length_aa
        == 10
    )


    assert (
        first.protein_id
        == "TEST_PROT_0001"
    )


    assert (
        translation.ribosome_name
        == "Ribosome"
    )


def test_local_free_ribosome_binds_mrna():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        voxel_counts,
    ) = build_context()


    voxel = seed_one_mrna(
        mrna,
        transcription,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 20


    engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,
    )


    engine.step(
        dt_s=0.001
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        engine.initiation_events
        == 1
    )


    assert (
        engine.active_complex_count
        == 1
    )


    assert (
        int(
            mrna.bound_ribosomes[
                0
            ].item()
        )
        == 1
    )


    assert (
        engine.free_ribosome_count
        == 19
    )


    assert (
        engine.ribosome_accounted_count
        == 20
    )


    assert engine.validate()


def test_polysome_can_form_on_one_mrna():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        voxel_counts,
    ) = build_context()


    voxel = seed_one_mrna(
        mrna,
        transcription,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 20


    engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,
    )


    # Protein requires ~10 ms with our artificial test rate.
    #
    # Three 1-ms steps therefore allow three initiation
    # opportunities before the first ribosome finishes.

    engine.run_steps(
        steps=3,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        engine.active_complex_count
        == 3
    )


    assert (
        int(
            mrna.bound_ribosomes[
                0
            ].item()
        )
        == 3
    )


    assert (
        engine.free_ribosome_count
        == 17
    )


    assert (
        engine.ribosome_accounted_count
        == 20
    )


    assert engine.validate()


def test_translation_completion_releases_ribosomes():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        voxel_counts,
    ) = build_context()


    voxel = seed_one_mrna(
        mrna,
        transcription,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 5


    engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,
    )


    engine.run_steps(
        steps=30,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        engine.completion_events
        > 0
    )


    assert (
        engine.completed_for_template(
            0
        )
        > 0
    )


    # Free + bound must always equal the original five.

    assert (
        engine.ribosome_accounted_count
        == 5
    )


    assert engine.validate()


def test_bound_mrna_is_protected_from_degradation():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        voxel_counts,
    ) = build_context()


    seed_one_mrna(
        mrna,
        transcription,
    )


    # Pretend one translation complex currently references it.

    mrna.bound_ribosomes[
        0
    ] = 1


    degradation = MRNADegradationEngine(
        mrna_state=mrna,

        degradation_rates=(
            transcription
            .mrna_degradation_rates
        ),

        seed=config.seed,
    )


    degradation.run_steps(
        steps=1000,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    # Even though this test mRNA has an extremely short
    # half-life, it cannot disappear while a ribosome is bound.

    assert (
        int(
            mrna.active[
                0
            ].item()
        )
        == 1
    )


    assert (
        degradation.degradation_events
        == 0
    )


    # Remove the artificial bound ribosome.

    mrna.bound_ribosomes[
        0
    ] = 0


    degradation.run_steps(
        steps=1000,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    assert (
        int(
            mrna.active[
                0
            ].item()
        )
        == 0
    )


    assert (
        degradation.degradation_events
        == 1
    )


def test_translation_crowding_replaces_free_ribosome_volume():

    (
        config,
        species,
        grid,
        chromosome,
        transcription,
        translation,
        mrna,
        voxel_counts,
    ) = build_context()


    voxel = seed_one_mrna(
        mrna,
        transcription,
    )


    ribosome_id = (
        translation
        .ribosome_voxel_count_id
    )


    voxel_counts.counts[
        ribosome_id,
        voxel
    ] = 5


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

        voxel_count_state=voxel_counts,
    )


    # Five free ribosomes:
    #
    # 5 × 8000 = 40,000 nm^3

    assert (
        synchronizer.dynamic_volume_nm3
        == pytest.approx(
            40000.0
        )
    )


    engine = TranslationEngine(
        voxel_count_state=voxel_counts,

        mrna_state=mrna,

        registry=translation,

        seed=config.seed,

        crowding_synchronizer=(
            synchronizer
        ),
    )


    engine.step(
        dt_s=0.001
    )


    cp.cuda.Stream.null.synchronize()


    # One ribosome should now be bound:
    #
    # free:
    #     4 × 8000 = 32,000
    #
    # translation complex:
    #     1 × 8500 =  8,500
    #
    # total:
    #                 40,500 nm^3

    assert (
        engine.active_complex_count
        == 1
    )


    assert (
        engine.free_ribosome_count
        == 4
    )


    assert (
        synchronizer.dynamic_volume_nm3
        == pytest.approx(
            40500.0
        )
    )


    assert synchronizer.validate()

    assert engine.validate()
