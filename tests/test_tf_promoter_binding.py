import cupy as cp

from chromosome.genome_loader import (
    GenomeLoader,
)
from chromosome.spatial import (
    ChromosomeSpatialState,
)
from expression.promoters import (
    PromoterRegistry,
)
from expression.tf_binding import (
    TFBindingRegistry,
    TFPromoterBindingEngine,
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


def build_promoters(
    chromosome,
    grid,
    config,
):

    spatial_state = (
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
            spatial_state=spatial_state,
            upstream_bp=(
                config.chromosome
                .promoter_upstream_bp
            ),
        )
    )


    return promoters


def test_real_ecoli_binding_rules_resolve():

    config = load_config(
        "configs/default.yaml"
    )


    species = SpeciesRegistry.from_yaml(
        config.biology.species_file
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    chromosome = (
        GenomeLoader.from_genbank(
            config.biology.genome_file
        )
    )


    promoters = build_promoters(
        chromosome,
        grid,
        config,
    )


    registry = (
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


    assert (
        registry.site_count
        == 2
    )


    for site in registry.sites:

        assert (
            0
            <= site.promoter_voxel_id
            < grid.cell_voxels
        )


def test_tf_binding_and_unbinding():

    config = load_config(
        "configs/default.yaml"
    )


    species = SpeciesRegistry.from_yaml(
        config.biology.species_file
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    chromosome = (
        GenomeLoader.from_genbank(
            "tests/data/synthetic_genome.gb"
        )
    )


    promoters = build_promoters(
        chromosome,
        grid,
        config,
    )


    registry = (
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


    # ========================================================
    # For this unit test, put 20 example_TF molecules directly
    # into the promoter voxel.
    #
    # This avoids waiting for random diffusion to happen to
    # bring a TF to one particular voxel.
    # ========================================================

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


    promoter_voxel = (
        registry.sites[
            0
        ].promoter_voxel_id
    )


    tf_id = (
        species.get(
            "example_TF"
        ).particle_id
    )


    state.primary[
        promoter_voxel,
        :20
    ] = tf_id


    state.primary_occupancy[
        promoter_voxel
    ] = 20


    initial_free_tf = 20


    binding = (
        TFPromoterBindingEngine(
            particle_state=state,
            registry=registry,
            seed=config.seed,
        )
    )


    # ========================================================
    # Run one second using 1-ms regulatory checks.
    # ========================================================

    binding.run_steps(
        steps=1000,
        dt_s=0.001,
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # Binding should actually have occurred.
    # ========================================================

    assert (
        binding.binding_events
        > 0
    )


    assert (
        binding.unbinding_events
        > 0
    )


    # ========================================================
    # Count remaining free TF molecules.
    # ========================================================

    free_primary = int(
        cp.count_nonzero(
            state.primary
            == tf_id
        ).item()
    )


    free_spill = int(
        cp.count_nonzero(
            state.spill
            == tf_id
        ).item()
    )


    final_free_tf = (
        free_primary
        + free_spill
    )


    # ========================================================
    # TF molecules must be conserved:
    #
    # free TF + promoter-bound TF = initial TF
    # ========================================================

    assert (
        final_free_tf
        + binding.bound_count
        == initial_free_tf
    )


    assert binding.validate()
