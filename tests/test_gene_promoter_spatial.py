from chromosome.genome_loader import (
    GenomeLoader,
)
from chromosome.spatial import (
    ChromosomeSpatialState,
)
from expression.promoters import (
    PromoterRegistry,
)
from spatial.config import (
    load_config,
)
from spatial.grid import (
    SpatialGrid,
)


def test_gene_and_promoter_spatial_mapping():

    config = load_config(
        "configs/default.yaml"
    )


    chromosome = (
        GenomeLoader.from_genbank(
            "tests/data/synthetic_genome.gb"
        )
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    spatial_state = (
        ChromosomeSpatialState(
            chromosome=chromosome,
            grid=grid,
            spatial_config=config.spatial,
            chromosome_config=(
                config.chromosome
            ),
        )
    )


    promoters = (
        PromoterRegistry.from_chromosome(
            chromosome=chromosome,
            spatial_state=spatial_state,
            upstream_bp=(
                config.chromosome
                .promoter_upstream_bp
            ),
        )
    )


    # ========================================================
    # Synthetic fixture has two genes.
    # ========================================================

    assert (
        spatial_state.gene_count
        == 2
    )


    assert (
        promoters.count
        == 2
    )


    # ========================================================
    # Every gene must map into a real active voxel.
    # ========================================================

    for gene in (
        spatial_state.genes
    ):

        assert (
            0
            <= gene.voxel_id
            < grid.cell_voxels
        )


    # ========================================================
    # Every promoter must also map into the cell.
    # ========================================================

    for promoter in (
        promoters.sites
    ):

        assert (
            0
            <= promoter.voxel_id
            < grid.cell_voxels
        )


    # ========================================================
    # Verify strand-aware promoter coordinates.
    #
    # geneA:
    #
    # GenBank 10..39
    # → Python start = 9
    # → forward promoter = 9 - 40 mod 120
    # → 89
    # ========================================================

    promoter_a = (
        promoters.get_by_locus_tag(
            "TEST_0001"
        )
    )


    assert (
        promoter_a.promoter_bp
        == 89
    )


    # ========================================================
    # geneB:
    #
    # complement(70..99)
    #
    # Python coordinates:
    # start = 69
    # end   = 99
    #
    # reverse 5' base = 98
    # promoter = 98 + 40 mod 120
    # → 18
    # ========================================================

    promoter_b = (
        promoters.get_by_locus_tag(
            "TEST_0002"
        )
    )


    assert (
        promoter_b.promoter_bp
        == 18
    )


    # ========================================================
    # Circular coordinate wrapping should work.
    # ========================================================

    assert (
        spatial_state.voxel_for_bp(
            0
        )
        ==
        spatial_state.voxel_for_bp(
            chromosome.length_bp
        )
    )
