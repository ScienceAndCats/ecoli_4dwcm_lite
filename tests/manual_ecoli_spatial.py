import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


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


config = load_config(
    "configs/default.yaml"
)


chromosome = (
    GenomeLoader.from_genbank(
        config.biology.genome_file
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


chromosome.describe()
spatial_state.describe()
promoters.describe()


print()
print("FIRST 10 SPATIAL GENES")
print("--------------------------------------------")


for gene in spatial_state.genes[:10]:

    print(
        f"{gene.locus_tag or '-':12s} "
        f"{gene.gene_name or '-':10s} "
        f"bp={gene.anchor_bp:9,d} "
        f"voxel={gene.voxel_id:6d} "
        f"xyz=("
        f"{gene.x_um:6.3f}, "
        f"{gene.y_um:6.3f}, "
        f"{gene.z_um:6.3f})"
    )


print()
print("FIRST 10 PROMOTERS")
print("--------------------------------------------")


for promoter in promoters.sites[:10]:

    print(
        f"{promoter.locus_tag or '-':12s} "
        f"{promoter.gene_name or '-':10s} "
        f"bp={promoter.promoter_bp:9,d} "
        f"voxel={promoter.voxel_id:6d}"
    )
