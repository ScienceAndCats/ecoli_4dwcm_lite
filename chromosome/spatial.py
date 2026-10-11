from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chromosome.chromosome import (
    Chromosome,
    Strand,
)
from spatial.config import (
    ChromosomeConfig,
    SpatialConfig,
)
from spatial.grid import SpatialGrid


@dataclass(frozen=True)
class SpatialGene:
    """
    Spatial representation of one annotated gene.
    """

    feature_id: str

    locus_tag: str | None
    gene_name: str | None

    start_bp: int
    end_bp: int

    strand: Strand

    anchor_bp: int

    voxel_id: int

    x_um: float
    y_um: float
    z_um: float


class ChromosomeSpatialState:
    """
    Coarse spatial representation of a chromosome.

    This is NOT intended to reproduce the true polymer
    conformation of a bacterial chromosome.

    Its purpose is to give genomic loci stable 3D positions
    inside the simulated cell so spatial transcription
    machinery can be developed.

    Later chromosome/replication models can replace the
    mapping without changing gene-expression interfaces.
    """

    def __init__(
        self,
        chromosome: Chromosome,
        grid: SpatialGrid,
        spatial_config: SpatialConfig,
        chromosome_config: ChromosomeConfig,
    ):

        self.chromosome = chromosome
        self.grid = grid

        self.spatial_config = (
            spatial_config
        )

        self.config = (
            chromosome_config
        )

        self._build_active_coordinates()

        self._voxel_cache = {}

        self.genes = (
            self._build_gene_locations()
        )


    # ========================================================
    # Active voxel coordinates
    # ========================================================

    def _build_active_coordinates(self):

        nx, ny, nz = (
            self.grid.shape
        )

        flat = (
            self.grid.active_flat
        )

        plane = (
            ny * nz
        )


        xi = (
            flat
            // plane
        )

        remainder = (
            flat
            - xi * plane
        )

        yi = (
            remainder
            // nz
        )

        zi = (
            remainder
            - yi * nz
        )


        dx = (
            self.grid.voxel_size_um
        )


        self.active_x_um = (
            xi
            - (nx - 1) / 2
        ) * dx

        self.active_y_um = (
            yi
            - (ny - 1) / 2
        ) * dx

        self.active_z_um = (
            zi
            - (nz - 1) / 2
        ) * dx


    # ========================================================
    # Coarse chromosome scaffold
    # ========================================================

    def target_position_um(
        self,
        bp: int,
    ):

        """
        Convert one genomic coordinate into a target
        3D position on a closed elongated scaffold.

        Because the chromosome is circular, bp=0 and bp=L
        map continuously to the same spatial location.
        """

        genome_length = (
            self.chromosome.length_bp
        )


        coordinate = (
            int(bp)
            % genome_length
        )


        fraction = (
            coordinate
            / genome_length
        )


        theta = (
            2.0
            * np.pi
            * fraction
        )


        half_length = (
            self.spatial_config
            .cell_length_um
            * self.config
            .nucleoid_length_fraction
            / 2.0
        )


        cell_radius = (
            self.spatial_config
            .cell_diameter_um
            / 2.0
        )


        nucleoid_radius = (
            cell_radius
            * self.config
            .nucleoid_radius_fraction
        )


        x_um = (
            half_length
            * np.cos(
                theta
            )
        )


        y_um = (
            nucleoid_radius
            * np.sin(
                theta
            )
        )


        z_um = (
            nucleoid_radius
            * 0.5
            * np.sin(
                self.config.scaffold_waves
                * theta
            )
        )


        return (
            float(
                x_um
            ),
            float(
                y_um
            ),
            float(
                z_um
            ),
        )


    # ========================================================
    # Genomic coordinate → active voxel
    # ========================================================

    def voxel_for_bp(
        self,
        bp: int,
    ):

        coordinate = (
            int(bp)
            % self.chromosome.length_bp
        )


        cached = (
            self._voxel_cache.get(
                coordinate
            )
        )


        if cached is not None:

            return cached


        (
            target_x,
            target_y,
            target_z,
        ) = self.target_position_um(
            coordinate
        )


        distance_squared = (
            (
                self.active_x_um
                - target_x
            ) ** 2

            + (
                self.active_y_um
                - target_y
            ) ** 2

            + (
                self.active_z_um
                - target_z
            ) ** 2
        )


        voxel_id = int(
            np.argmin(
                distance_squared
            )
        )


        self._voxel_cache[
            coordinate
        ] = voxel_id


        return voxel_id


    def voxel_xyz_um(
        self,
        voxel_id: int,
    ):

        return (
            float(
                self.active_x_um[
                    voxel_id
                ]
            ),

            float(
                self.active_y_um[
                    voxel_id
                ]
            ),

            float(
                self.active_z_um[
                    voxel_id
                ]
            ),
        )


    # ========================================================
    # Gene positions
    # ========================================================

    def _build_gene_locations(self):

        genes = []


        for gene in (
            self.chromosome.genes
        ):

            # Use the midpoint of the annotated gene as the
            # gene's initial coarse spatial anchor.

            anchor_bp = (
                gene.start_bp
                + gene.end_bp
                - 1
            ) // 2


            voxel_id = (
                self.voxel_for_bp(
                    anchor_bp
                )
            )


            (
                x_um,
                y_um,
                z_um,
            ) = self.voxel_xyz_um(
                voxel_id
            )


            genes.append(
                SpatialGene(
                    feature_id=(
                        gene.feature_id
                    ),

                    locus_tag=(
                        gene.locus_tag
                    ),

                    gene_name=(
                        gene.gene_name
                    ),

                    start_bp=(
                        gene.start_bp
                    ),

                    end_bp=(
                        gene.end_bp
                    ),

                    strand=(
                        gene.strand
                    ),

                    anchor_bp=(
                        anchor_bp
                    ),

                    voxel_id=(
                        voxel_id
                    ),

                    x_um=x_um,
                    y_um=y_um,
                    z_um=z_um,
                )
            )


        return tuple(
            genes
        )


    @property
    def gene_count(self):

        return len(
            self.genes
        )


    def describe(self):

        unique_voxels = len(
            {
                gene.voxel_id
                for gene
                in self.genes
            }
        )


        print()
        print("CHROMOSOME SPATIAL STATE")
        print("--------------------------------------------")

        print(
            f"Spatial genes:              "
            f"{self.gene_count:,}"
        )

        print(
            f"Occupied gene voxels:       "
            f"{unique_voxels:,}"
        )

        print(
            f"Nucleoid length fraction:   "
            f"{self.config.nucleoid_length_fraction:.2f}"
        )

        print(
            f"Nucleoid radius fraction:   "
            f"{self.config.nucleoid_radius_fraction:.2f}"
        )

        print(
            f"Scaffold waves:             "
            f"{self.config.scaffold_waves:.1f}"
        )
