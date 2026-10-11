from __future__ import annotations

from dataclasses import dataclass

from chromosome.chromosome import (
    Chromosome,
    Strand,
)
from chromosome.spatial import (
    ChromosomeSpatialState,
)


@dataclass(frozen=True)
class PromoterSite:
    """
    One spatial promoter anchor.

    For now promoter coordinates are derived from the
    annotated gene 5' end.

    These are placeholders until real promoter/TSS
    annotations are loaded.
    """

    promoter_id: str

    gene_feature_id: str

    locus_tag: str | None
    gene_name: str | None

    strand: Strand

    promoter_bp: int

    voxel_id: int

    x_um: float
    y_um: float
    z_um: float

    source: str


class PromoterRegistry:

    def __init__(
        self,
        sites,
    ):

        self.sites = tuple(
            sites
        )


    @classmethod
    def from_chromosome(
        cls,
        chromosome: Chromosome,
        spatial_state: ChromosomeSpatialState,
        upstream_bp: int,
    ):

        if upstream_bp < 0:

            raise ValueError(
                "upstream_bp must be >= 0"
            )


        sites = []


        for gene in (
            chromosome.genes
        ):

            # ====================================================
            # Approximate the transcription-facing end.
            #
            # Forward gene:
            #
            #        promoter → gene
            #
            # Reverse gene:
            #
            #        gene ← promoter
            #
            # ====================================================

            if (
                gene.strand
                == Strand.FORWARD
            ):

                five_prime_bp = (
                    gene.start_bp
                )


                promoter_bp = (
                    five_prime_bp
                    - upstream_bp
                ) % chromosome.length_bp


            else:

                five_prime_bp = (
                    gene.end_bp
                    - 1
                )


                promoter_bp = (
                    five_prime_bp
                    + upstream_bp
                ) % chromosome.length_bp


            voxel_id = (
                spatial_state.voxel_for_bp(
                    promoter_bp
                )
            )


            (
                x_um,
                y_um,
                z_um,
            ) = spatial_state.voxel_xyz_um(
                voxel_id
            )


            sites.append(
                PromoterSite(
                    promoter_id=(
                        "promoter:"
                        f"{gene.feature_id}"
                    ),

                    gene_feature_id=(
                        gene.feature_id
                    ),

                    locus_tag=(
                        gene.locus_tag
                    ),

                    gene_name=(
                        gene.gene_name
                    ),

                    strand=(
                        gene.strand
                    ),

                    promoter_bp=(
                        promoter_bp
                    ),

                    voxel_id=(
                        voxel_id
                    ),

                    x_um=x_um,
                    y_um=y_um,
                    z_um=z_um,

                    source=(
                        "derived_5prime_offset"
                    ),
                )
            )


        return cls(
            sites
        )


    @property
    def count(self):

        return len(
            self.sites
        )


    def get_by_locus_tag(
        self,
        locus_tag,
    ):

        for site in self.sites:

            if (
                site.locus_tag
                == locus_tag
            ):

                return site


        raise KeyError(
            locus_tag
        )


    def describe(self):

        unique_voxels = len(
            {
                site.voxel_id
                for site
                in self.sites
            }
        )


        print()
        print("PROMOTER REGISTRY")
        print("--------------------------------------------")

        print(
            f"Promoter anchors:           "
            f"{self.count:,}"
        )

        print(
            f"Occupied promoter voxels:   "
            f"{unique_voxels:,}"
        )

        print(
            "Source:                     "
            "derived 5' gene offset"
        )
