import cupy as cp
import pytest

from chromosome.genome_loader import (
    GenomeLoader,
)
from chromosome.occupancy import (
    DNAOccupancyLayer,
)
from chromosome.spatial import (
    ChromosomeSpatialState,
)
from spatial.config import (
    load_config,
)
from spatial.crowding import (
    CrowdingState,
)
from spatial.grid import (
    SpatialGrid,
)


def build_context(
    genome_path,
):

    config = load_config(
        "configs/default.yaml"
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    chromosome = GenomeLoader.from_genbank(
        genome_path
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


    crowding = CrowdingState(
        grid=grid
    )


    dna = DNAOccupancyLayer(
        crowding_state=crowding,

        segment_bp=(
            config.chromosome
            .dna_occupancy_segment_bp
        ),

        volume_per_bp_nm3=(
            config.chromosome
            .dna_volume_per_bp_nm3
        ),
    )


    return (
        config,
        grid,
        chromosome,
        chromosome_spatial,
        crowding,
        dna,
    )


def test_synthetic_dna_volume_is_conserved():

    (
        config,
        grid,
        chromosome,
        chromosome_spatial,
        crowding,
        dna,
    ) = build_context(
        "tests/data/"
        "synthetic_genome.gb"
    )


    dna.rebuild_from_chromosome_spatial(
        chromosome_spatial
    )


    expected = (
        chromosome.length_bp
        * config.chromosome
        .dna_volume_per_bp_nm3
    )


    assert (
        dna.represented_bp
        == chromosome.length_bp
    )


    assert (
        crowding.total_dna_volume_nm3
        == pytest.approx(
            expected,
            rel=1e-5,
        )
    )


    assert (
        crowding.dna_occupied_voxel_count
        >= 1
    )


    assert dna.validate()


def test_dna_layer_is_separate_from_other_crowding():

    (
        config,
        grid,
        chromosome,
        chromosome_spatial,
        crowding,
        dna,
    ) = build_context(
        "tests/data/"
        "synthetic_genome.gb"
    )


    crowding.add_static_volume(
        voxel_ids=[
            0
        ],

        volumes_nm3=1000.0,
    )


    crowding.add_dynamic_volume(
        voxel_ids=[
            1
        ],

        volumes_nm3=2000.0,
    )


    dna.rebuild_from_chromosome_spatial(
        chromosome_spatial
    )


    assert (
        crowding.total_static_volume_nm3
        == pytest.approx(
            1000.0
        )
    )


    assert (
        crowding.total_dynamic_volume_nm3
        == pytest.approx(
            2000.0
        )
    )


    assert (
        crowding.total_dna_volume_nm3
        > 0
    )


    assert dna.validate()


def test_rebuilding_moved_dna_replaces_old_positions():

    (
        config,
        grid,
        chromosome,
        chromosome_spatial,
        crowding,
        dna,
    ) = build_context(
        "tests/data/"
        "synthetic_genome.gb"
    )


    # ========================================================
    # Pretend a future chromosome model has three movable
    # DNA segments.
    # ========================================================

    segment_lengths = [
        40,
        40,
        40,
    ]


    dna.rebuild_from_segments(
        voxel_ids=[
            10,
            20,
            30,
        ],

        segment_lengths_bp=(
            segment_lengths
        ),
    )


    first_total = (
        crowding.total_dna_volume_nm3
    )


    assert (
        float(
            crowding.dna_volume_nm3[
                10
            ].item()
        )
        > 0
    )


    # ========================================================
    # Now "move" every DNA segment somewhere else.
    # ========================================================

    dna.rebuild_from_segments(
        voxel_ids=[
            40,
            50,
            60,
        ],

        segment_lengths_bp=(
            segment_lengths
        ),
    )


    second_total = (
        crowding.total_dna_volume_nm3
    )


    # Old DNA positions were cleared.

    assert (
        float(
            crowding.dna_volume_nm3[
                10
            ].item()
        )
        == pytest.approx(
            0.0
        )
    )


    assert (
        float(
            crowding.dna_volume_nm3[
                20
            ].item()
        )
        == pytest.approx(
            0.0
        )
    )


    assert (
        float(
            crowding.dna_volume_nm3[
                30
            ].item()
        )
        == pytest.approx(
            0.0
        )
    )


    # New positions contain DNA.

    assert (
        float(
            crowding.dna_volume_nm3[
                40
            ].item()
        )
        > 0
    )


    assert (
        float(
            crowding.dna_volume_nm3[
                50
            ].item()
        )
        > 0
    )


    assert (
        float(
            crowding.dna_volume_nm3[
                60
            ].item()
        )
        > 0
    )


    # Moving DNA changes position, not total DNA volume.

    assert second_total == pytest.approx(
        first_total,
        rel=1e-6,
    )


def test_two_chromosome_copies_double_dna_volume():

    (
        config,
        grid,
        chromosome,
        chromosome_spatial,
        crowding,
        dna,
    ) = build_context(
        "tests/data/"
        "synthetic_genome.gb"
    )


    dna.rebuild_from_chromosome_spatial(
        [
            chromosome_spatial,
            chromosome_spatial,
        ]
    )


    expected = (
        2
        * chromosome.length_bp
        * config.chromosome
        .dna_volume_per_bp_nm3
    )


    assert (
        dna.chromosome_count
        == 2
    )


    assert (
        dna.represented_bp
        == (
            2
            * chromosome.length_bp
        )
    )


    assert (
        crowding.total_dna_volume_nm3
        == pytest.approx(
            expected,
            rel=1e-5,
        )
    )


def test_real_ecoli_dna_occupancy():

    (
        config,
        grid,
        chromosome,
        chromosome_spatial,
        crowding,
        dna,
    ) = build_context(
        load_config(
            "configs/default.yaml"
        ).biology.genome_file
    )


    dna.rebuild_from_chromosome_spatial(
        chromosome_spatial
    )


    expected = (
        chromosome.length_bp
        * config.chromosome
        .dna_volume_per_bp_nm3
    )


    assert (
        dna.represented_bp
        == chromosome.length_bp
    )


    assert (
        crowding.total_dna_volume_nm3
        == pytest.approx(
            expected,
            rel=1e-5,
        )
    )


    # The temporary chromosome should occupy many spatial
    # voxels instead of collapsing the genome into one voxel.

    assert (
        crowding.dna_occupied_voxel_count
        > 50
    )


    # Bare geometric DNA should fit within every occupied
    # 50-nm voxel for this temporary scaffold.

    assert (
        crowding.max_occupied_fraction
        < 1.0
    )


    assert dna.validate()
