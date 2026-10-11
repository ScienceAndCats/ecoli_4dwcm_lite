from __future__ import annotations

from pathlib import Path

from Bio import SeqIO

from chromosome.chromosome import (
    Chromosome,
    FeatureType,
    GenomeFeature,
    Strand,
)


class GenomeLoader:
    """
    Organism-independent genome/annotation loader.

    Current supported input:

        GenBank / RefSeq .gb / .gbk / .gbff

    The loader converts external annotation formats into the
    simulator's own internal Chromosome + GenomeFeature objects.
    """


    @classmethod
    def from_genbank(
        cls,
        path,
    ) -> Chromosome:

        path = Path(
            path
        )


        records = list(
            SeqIO.parse(
                path,
                "genbank",
            )
        )


        if len(records) == 0:

            raise ValueError(
                f"No GenBank records found in {path}"
            )


        if len(records) > 1:

            raise ValueError(
                "GenomeLoader.from_genbank currently expects "
                "one chromosome/replicon per file. "
                f"Found {len(records)} records in {path}."
            )


        record = records[0]


        sequence = str(
            record.seq
        ).upper()


        accession = (
            record.id
            or record.name
            or path.stem
        )


        name = (
            record.description
            or accession
        )


        topology = (
            record.annotations
            .get(
                "topology",
                "",
            )
            .lower()
        )


        circular = (
            topology
            == "circular"
        )


        features = []


        for index, feature in enumerate(
            record.features
        ):

            feature_type = (
                cls._map_feature_type(
                    feature.type
                )
            )


            # Skip features that have no meaningful genomic
            # interval.

            if feature.location is None:

                continue


            start_bp = int(
                feature.location.start
            )

            end_bp = int(
                feature.location.end
            )


            strand_value = (
                feature.location.strand
            )


            if strand_value == -1:

                strand = (
                    Strand.REVERSE
                )

            else:

                strand = (
                    Strand.FORWARD
                )


            qualifiers = (
                feature.qualifiers
            )


            locus_tag = (
                cls._first_qualifier(
                    qualifiers,
                    "locus_tag",
                )
            )


            gene_name = (
                cls._first_qualifier(
                    qualifiers,
                    "gene",
                )
            )


            product = (
                cls._first_qualifier(
                    qualifiers,
                    "product",
                )
            )


            protein_id = (
                cls._first_qualifier(
                    qualifiers,
                    "protein_id",
                )
            )


            feature_id = (
                locus_tag
                or protein_id
                or gene_name
                or (
                    f"{feature.type}_"
                    f"{index}"
                )
            )


            features.append(
                GenomeFeature(
                    feature_id=feature_id,

                    feature_type=(
                        feature_type
                    ),

                    start_bp=start_bp,

                    end_bp=end_bp,

                    strand=strand,

                    locus_tag=locus_tag,

                    gene_name=gene_name,

                    product=product,

                    protein_id=protein_id,

                    parent_locus_tag=(
                        locus_tag
                    ),
                )
            )


        return Chromosome(
            name=name,

            accession=accession,

            sequence=sequence,

            circular=circular,

            features=tuple(
                features
            ),
        )


    @staticmethod
    def _first_qualifier(
        qualifiers,
        key,
    ):

        values = qualifiers.get(
            key
        )


        if not values:

            return None


        return str(
            values[0]
        )


    @staticmethod
    def _map_feature_type(
        value,
    ):

        if value == "gene":

            return FeatureType.GENE


        if value == "CDS":

            return FeatureType.CDS


        if value == "tRNA":

            return FeatureType.TRNA


        if value == "rRNA":

            return FeatureType.RRNA


        if value == "ncRNA":

            return FeatureType.NCRNA


        if value == "tmRNA":

            return FeatureType.TMRNA


        return FeatureType.OTHER
