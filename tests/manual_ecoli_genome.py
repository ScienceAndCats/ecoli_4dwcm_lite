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


from chromosome.chromosome import (
    FeatureType,
)
from chromosome.genome_loader import (
    GenomeLoader,
)


GENOME_PATH = (
    "data/genomes/"
    "ecoli_k12_mg1655/"
    "GCF_000005845.2_ASM584v2_genomic.gbff"
)


chromosome = (
    GenomeLoader.from_genbank(
        GENOME_PATH
    )
)


chromosome.describe()


print()
print("FIRST 10 GENES")
print("--------------------------------------------")


for gene in chromosome.genes[:10]:

    print(
        f"{gene.locus_tag or '-':12s} "
        f"{gene.gene_name or '-':10s} "
        f"{gene.start_bp:9,d} - "
        f"{gene.end_bp:9,d}   "
        f"{gene.strand.value}"
    )


print()
print("FEATURE COUNTS")
print("--------------------------------------------")


for feature_type in FeatureType:

    count = len(
        chromosome.features_of_type(
            feature_type
        )
    )

    print(
        f"{feature_type.value:10s} "
        f"{count:,}"
    )
