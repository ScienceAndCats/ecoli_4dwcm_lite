from __future__ import annotations

import numpy as np

from spatial.crowding import (
    CrowdingState,
)


class DNAOccupancyLayer:
    """
    Recomputable spatial DNA occupancy.

    The output is simply:

        DNA volume in each active voxel


    HOW the DNA positions were obtained is intentionally
    separate from this class.

    For now:

        genomic coordinate
            ↓
        temporary ChromosomeSpatialState scaffold
            ↓
        voxel


    Later:

        movable chromosome segments
            ↓
        current segment voxel positions
            ↓
        voxel


    Crowding therefore does not need to be redesigned when
    the chromosome model becomes dynamic.
    """

    def __init__(
        self,
        crowding_state: CrowdingState,
        segment_bp: int,
        volume_per_bp_nm3: float,
    ):

        self.crowding_state = (
            crowding_state
        )


        self.segment_bp = int(
            segment_bp
        )


        self.volume_per_bp_nm3 = float(
            volume_per_bp_nm3
        )


        if self.segment_bp <= 0:

            raise ValueError(
                "DNA occupancy segment_bp "
                "must be > 0."
            )


        if (
            not np.isfinite(
                self.volume_per_bp_nm3
            )
            or self.volume_per_bp_nm3 < 0
        ):

            raise ValueError(
                "DNA volume per bp must be "
                "finite and non-negative."
            )


        self.last_voxel_ids = np.empty(
            0,
            dtype=np.int32,
        )


        self.last_segment_lengths_bp = np.empty(
            0,
            dtype=np.int32,
        )


        self.chromosome_count = 0


    # ========================================================
    # Fundamental interface
    #
    # Future movable DNA models can call this directly.
    # ========================================================

    def rebuild_from_segments(
        self,
        voxel_ids,
        segment_lengths_bp,
        chromosome_count=1,
    ):

        voxel_ids = np.asarray(
            voxel_ids,
            dtype=np.int32,
        ).reshape(
            -1
        )


        segment_lengths_bp = np.asarray(
            segment_lengths_bp,
            dtype=np.int32,
        ).reshape(
            -1
        )


        if (
            voxel_ids.shape
            != segment_lengths_bp.shape
        ):

            raise ValueError(
                "voxel_ids and segment_lengths_bp "
                "must have matching shapes."
            )


        if (
            int(
                chromosome_count
            )
            < 0
        ):

            raise ValueError(
                "chromosome_count cannot be negative."
            )


        if (
            voxel_ids.size > 0
        ):

            if np.any(
                voxel_ids < 0
            ):

                raise ValueError(
                    "DNA voxel IDs cannot be negative."
                )


            if np.any(
                voxel_ids
                >= self.crowding_state.cell_voxels
            ):

                raise ValueError(
                    "DNA voxel ID exceeds active "
                    "cell grid."
                )


            if np.any(
                segment_lengths_bp <= 0
            ):

                raise ValueError(
                    "DNA segment lengths must be > 0."
                )


        # ====================================================
        # Replace the previous DNA map.
        #
        # This is what makes the layer "recomputable".
        # ====================================================

        self.crowding_state.clear_dna()


        volumes_nm3 = (
            segment_lengths_bp.astype(
                np.float64
            )
            * self.volume_per_bp_nm3
        )


        self.crowding_state.add_dna_volume(
            voxel_ids=voxel_ids,
            volumes_nm3=volumes_nm3,
        )


        self.last_voxel_ids = (
            voxel_ids.copy()
        )


        self.last_segment_lengths_bp = (
            segment_lengths_bp.copy()
        )


        self.chromosome_count = int(
            chromosome_count
        )


        return self


    # ========================================================
    # Temporary scaffold interface
    #
    # This converts the current ChromosomeSpatialState into
    # generic DNA segments, then calls rebuild_from_segments().
    # ========================================================

    def rebuild_from_chromosome_spatial(
        self,
        chromosome_spatial_states,
    ):

        # One spatial state may be supplied directly.

        if hasattr(
            chromosome_spatial_states,
            "voxel_for_bp",
        ):

            states = (
                chromosome_spatial_states,
            )


        else:

            states = tuple(
                chromosome_spatial_states
            )


        if len(states) == 0:

            raise ValueError(
                "At least one chromosome spatial "
                "state is required."
            )


        all_voxel_ids = []
        all_lengths = []


        for spatial_state in states:

            if (
                spatial_state.grid.cell_voxels
                != self.crowding_state.cell_voxels
            ):

                raise ValueError(
                    "Chromosome spatial state and "
                    "crowding state use different grids."
                )


            genome_length = (
                spatial_state
                .chromosome
                .length_bp
            )


            starts = np.arange(
                0,
                genome_length,
                self.segment_bp,
                dtype=np.int64,
            )


            ends = np.minimum(
                starts
                + self.segment_bp,
                genome_length,
            )


            lengths = (
                ends
                - starts
            ).astype(
                np.int32
            )


            # Midpoint represents the location of the
            # temporary genomic chunk on the current scaffold.

            midpoints = (
                (
                    starts
                    + ends
                    - 1
                )
                // 2
            )


            voxel_ids = np.asarray(
                [
                    spatial_state.voxel_for_bp(
                        int(
                            midpoint
                        )
                    )

                    for midpoint
                    in midpoints
                ],

                dtype=np.int32,
            )


            all_voxel_ids.append(
                voxel_ids
            )


            all_lengths.append(
                lengths
            )


        voxel_ids = np.concatenate(
            all_voxel_ids
        )


        lengths = np.concatenate(
            all_lengths
        )


        return self.rebuild_from_segments(
            voxel_ids=voxel_ids,

            segment_lengths_bp=lengths,

            chromosome_count=len(
                states
            ),
        )


    # ========================================================
    # Diagnostics
    # ========================================================

    @property
    def segment_count(
        self,
    ):

        return int(
            self.last_voxel_ids.size
        )


    @property
    def represented_bp(
        self,
    ):

        return int(
            np.sum(
                self.last_segment_lengths_bp,
                dtype=np.int64,
            )
        )


    @property
    def expected_dna_volume_nm3(
        self,
    ):

        return (
            self.represented_bp
            * self.volume_per_bp_nm3
        )


    @property
    def occupied_voxel_count(
        self,
    ):

        if (
            self.last_voxel_ids.size
            == 0
        ):

            return 0


        return int(
            np.unique(
                self.last_voxel_ids
            ).size
        )


    # ========================================================
    # Validation
    # ========================================================

    def validate(
        self,
    ):

        observed = (
            self.crowding_state
            .total_dna_volume_nm3
        )


        expected = (
            self.expected_dna_volume_nm3
        )


        tolerance = max(
            1e-3,
            expected
            * 1e-5,
        )


        if (
            abs(
                observed
                - expected
            )
            > tolerance
        ):

            raise RuntimeError(
                "DNA occupancy volume mismatch: "
                f"expected {expected:.3f} nm^3, "
                f"found {observed:.3f} nm^3."
            )


        self.crowding_state.validate()


        return True


    # ========================================================
    # Human-readable summary
    # ========================================================

    def describe(
        self,
    ):

        print()
        print("DNA OCCUPANCY LAYER")
        print("--------------------------------------------")


        print(
            f"Chromosome copies:          "
            f"{self.chromosome_count:,}"
        )


        print(
            f"DNA segments represented:   "
            f"{self.segment_count:,}"
        )


        print(
            f"Temporary sample size:      "
            f"{self.segment_bp:,} bp"
        )


        print(
            f"Represented DNA:            "
            f"{self.represented_bp:,} bp"
        )


        print(
            f"DNA volume / bp:            "
            f"{self.volume_per_bp_nm3:.3f} nm^3"
        )


        print(
            f"DNA-occupied voxels:        "
            f"{self.occupied_voxel_count:,}"
        )


        print(
            f"Total DNA volume:           "
            f"{self.crowding_state.total_dna_volume_nm3:,.1f} nm^3"
        )
