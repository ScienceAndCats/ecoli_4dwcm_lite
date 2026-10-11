from chromosome.chromosome import (
    FeatureType,
    Strand,
)
from chromosome.genome_loader import (
    GenomeLoader,
)


def test_synthetic_genome_loader():

    chromosome = (
        GenomeLoader.from_genbank(
            "tests/data/synthetic_genome.gb"
        )
    )


    # ========================================================
    # Basic chromosome metadata
    # ========================================================

    assert (
        chromosome.accession
        == "SYNTH0001.1"
    )


    assert (
        chromosome.length_bp
        == 120
    )


    assert chromosome.circular


    # ========================================================
    # Gene annotations
    # ========================================================

    genes = (
        chromosome.features_of_type(
            FeatureType.GENE
        )
    )


    assert len(
        genes
    ) == 2


    gene_a = next(
        gene
        for gene
        in genes
        if gene.locus_tag
        == "TEST_0001"
    )


    gene_b = next(
        gene
        for gene
        in genes
        if gene.locus_tag
        == "TEST_0002"
    )


    assert (
        gene_a.gene_name
        == "geneA"
    )


    assert (
        gene_a.start_bp
        == 9
    )


    assert (
        gene_a.end_bp
        == 39
    )


    assert (
        gene_a.strand
        == Strand.FORWARD
    )


    assert (
        gene_b.start_bp
        == 69
    )


    assert (
        gene_b.end_bp
        == 99
    )


    assert (
        gene_b.strand
        == Strand.REVERSE
    )


    # ========================================================
    # CDS annotations
    # ========================================================

    cds = (
        chromosome.features_of_type(
            FeatureType.CDS
        )
    )


    assert len(
        cds
    ) == 2


    cds_a = next(
        feature
        for feature
        in cds
        if feature.locus_tag
        == "TEST_0001"
    )


    assert (
        cds_a.product
        == "synthetic protein A"
    )


    assert (
        cds_a.protein_id
        == "TEST_PROT_0001"
    )
