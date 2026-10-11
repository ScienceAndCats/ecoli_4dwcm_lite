from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Strand(str, Enum):
    FORWARD = "+"
    REVERSE = "-"


class FeatureType(str, Enum):
    GENE = "gene"
    CDS = "CDS"
    TRNA = "tRNA"
    RRNA = "rRNA"
    NCRNA = "ncRNA"
    TMRNA = "tmRNA"
    OTHER = "other"


@dataclass(frozen=True)
class GenomeFeature:
    """
    One annotated genomic feature.

    Coordinates use Python conventions:

        start_bp:
            zero-based, inclusive

        end_bp:
            zero-based, exclusive

    Example:

        start_bp = 100
        end_bp   = 200

    corresponds to 100 bases:

        100 ... 199
    """

    feature_id: str

    feature_type: FeatureType

    start_bp: int
    end_bp: int

    strand: Strand

    locus_tag: str | None = None
    gene_name: str | None = None

    product: str | None = None

    protein_id: str | None = None

    parent_locus_tag: str | None = None


    @property
    def length_bp(self) -> int:

        return (
            self.end_bp
            - self.start_bp
        )


@dataclass(frozen=True)
class Chromosome:
    """
    Organism-independent chromosome representation.
    """

    name: str

    accession: str

    sequence: str

    circular: bool

    features: tuple[
        GenomeFeature,
        ...
    ]


    @property
    def length_bp(self) -> int:

        return len(
            self.sequence
        )


    @property
    def feature_count(self) -> int:

        return len(
            self.features
        )


    def features_of_type(
        self,
        feature_type: FeatureType,
    ):

        return tuple(
            feature
            for feature
            in self.features
            if (
                feature.feature_type
                == feature_type
            )
        )


    @property
    def genes(self):

        return self.features_of_type(
            FeatureType.GENE
        )


    @property
    def coding_sequences(self):

        return self.features_of_type(
            FeatureType.CDS
        )


    @property
    def trnas(self):

        return self.features_of_type(
            FeatureType.TRNA
        )


    @property
    def rrnas(self):

        return self.features_of_type(
            FeatureType.RRNA
        )


    def describe(self):

        print()
        print("CHROMOSOME")
        print("--------------------------------------------")

        print(
            f"Name:                       "
            f"{self.name}"
        )

        print(
            f"Accession:                  "
            f"{self.accession}"
        )

        print(
            f"Length:                     "
            f"{self.length_bp:,} bp"
        )

        print(
            f"Circular:                   "
            f"{self.circular}"
        )

        print(
            f"Annotated features:         "
            f"{self.feature_count:,}"
        )

        print(
            f"Genes:                      "
            f"{len(self.genes):,}"
        )

        print(
            f"CDS:                        "
            f"{len(self.coding_sequences):,}"
        )

        print(
            f"tRNA:                       "
            f"{len(self.trnas):,}"
        )

        print(
            f"rRNA:                       "
            f"{len(self.rrnas):,}"
        )
