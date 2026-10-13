from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np


class MRNAState:
    """
    Unique spatial mRNA molecules.

    Each active slot represents one individual transcript.

    Arrays:

        active[index]
        template_index[index]
        voxel_id[index]

    The transcription engine creates these directly at the
    voxel containing the transcribed gene's 3' end.
    """

    def __init__(
        self,
        template_count: int,
        cell_voxels: int,
        max_molecules: int,
    ):

        if template_count <= 0:
            raise ValueError(
                "template_count must be > 0"
            )

        if cell_voxels <= 0:
            raise ValueError(
                "cell_voxels must be > 0"
            )

        if max_molecules <= 0:
            raise ValueError(
                "max_molecules must be > 0"
            )


        self.template_count = int(
            template_count
        )

        self.cell_voxels = int(
            cell_voxels
        )

        self.max_molecules = int(
            max_molecules
        )


        # 0 = free slot
        # 1 = active mRNA
        # 2 = temporarily reserved by a CUDA kernel

        self.active = cp.zeros(
            self.max_molecules,
            dtype=cp.uint32,
        )


        self.template_index = cp.full(
            self.max_molecules,
            -1,
            dtype=cp.int32,
        )


        self.voxel_id = cp.full(
            self.max_molecules,
            -1,
            dtype=cp.int32,
        )


    @property
    def active_count(self):

        return int(
            cp.count_nonzero(
                self.active == 1
            ).item()
        )


    @property
    def free_capacity(self):

        return (
            self.max_molecules
            - self.active_count
        )


    def template_counts(self):

        mask = (
            self.active
            == 1
        )


        if not bool(
            cp.any(
                mask
            ).item()
        ):

            return np.zeros(
                self.template_count,
                dtype=np.int64,
            )


        template_ids = (
            self.template_index[
                mask
            ]
        )


        counts = cp.bincount(
            template_ids,
            minlength=self.template_count,
        )


        return cp.asnumpy(
            counts
        )


    def active_voxels(
        self,
        template_index=None,
    ):

        mask = (
            self.active
            == 1
        )


        if template_index is not None:

            mask = (
                mask
                &
                (
                    self.template_index
                    == int(
                        template_index
                    )
                )
            )


        return cp.asnumpy(
            self.voxel_id[
                mask
            ]
        )


    def validate(self):

        # Reserved state must not persist between completed
        # CUDA operations.

        if bool(
            cp.any(
                self.active > 1
            ).item()
        ):

            raise RuntimeError(
                "Persistent reserved mRNA slot detected."
            )


        mask = (
            self.active
            == 1
        )


        if bool(
            cp.any(
                self.template_index[
                    mask
                ]
                < 0
            ).item()
        ):

            raise RuntimeError(
                "Active mRNA has invalid template."
            )


        if bool(
            cp.any(
                self.template_index[
                    mask
                ]
                >= self.template_count
            ).item()
        ):

            raise RuntimeError(
                "Active mRNA template exceeds registry."
            )


        if bool(
            cp.any(
                self.voxel_id[
                    mask
                ]
                < 0
            ).item()
        ):

            raise RuntimeError(
                "Active mRNA has invalid voxel."
            )


        if bool(
            cp.any(
                self.voxel_id[
                    mask
                ]
                >= self.cell_voxels
            ).item()
        ):

            raise RuntimeError(
                "Active mRNA voxel exceeds cell grid."
            )


        return True


    def describe(self):

        print()
        print("mRNA STATE")
        print("--------------------------------------------")

        print(
            f"Active mRNAs:               "
            f"{self.active_count:,}"
        )

        print(
            f"Maximum mRNAs:              "
            f"{self.max_molecules:,}"
        )

        print(
            f"Free capacity:              "
            f"{self.free_capacity:,}"
        )


class MRNADegradationEngine:
    """
    Stochastic first-order degradation of individual mRNAs.
    """

    THREADS = 256


    def __init__(
        self,
        mrna_state: MRNAState,
        degradation_rates,
        seed: int,
    ):

        self.mrna_state = (
            mrna_state
        )

        self.seed = int(
            seed
        )

        self.step_count = 0


        rates = np.asarray(
            degradation_rates,
            dtype=np.float32,
        )


        if rates.shape != (
            mrna_state.template_count,
        ):

            raise ValueError(
                "mRNA degradation-rate array shape "
                "does not match template count."
            )


        if np.any(
            rates < 0
        ):

            raise ValueError(
                "mRNA degradation rates cannot be negative."
            )


        self.degradation_rates = cp.asarray(
            rates
        )


        self.degraded_by_template = cp.zeros(
            mrna_state.template_count,
            dtype=cp.uint64,
        )


        self.event_count = cp.zeros(
            1,
            dtype=cp.uint64,
        )


        kernel_path = (
            Path(__file__).parents[1]
            / "spatial"
            / "kernels"
            / "mrna.cu"
        )


        source = (
            kernel_path.read_text()
        )


        module = cp.RawModule(
            code=source,
            options=(
                "--std=c++11",
            ),
        )


        self._degrade = (
            module.get_function(
                "degrade_mrna"
            )
        )


    @classmethod
    def _blocks(
        cls,
        total,
    ):

        return (
            int(
                (
                    total
                    + cls.THREADS
                    - 1
                )
                // cls.THREADS
            ),
        )


    def step(
        self,
        dt_s: float,
    ):

        if dt_s <= 0:

            raise ValueError(
                "mRNA degradation timestep must be > 0."
            )


        state = (
            self.mrna_state
        )


        self._degrade(
            self._blocks(
                state.max_molecules
            ),

            (
                self.THREADS,
            ),

            (
                state.active,
                state.template_index,
                state.voxel_id,

                self.degradation_rates,

                self.degraded_by_template,
                self.event_count,

                np.int32(
                    state.max_molecules
                ),

                np.float32(
                    dt_s
                ),

                np.uint64(
                    self.seed
                ),

                np.uint64(
                    self.step_count
                ),
            ),
        )


        self.step_count += 1


    def run_steps(
        self,
        steps: int,
        dt_s: float,
    ):

        if steps < 0:

            raise ValueError(
                "steps must be >= 0"
            )


        for _ in range(
            steps
        ):

            self.step(
                dt_s
            )


    @property
    def degradation_events(self):

        return int(
            self.event_count[
                0
            ].item()
        )


    def degraded_for_template(
        self,
        template_index,
    ):

        return int(
            self.degraded_by_template[
                template_index
            ].item()
        )


    def validate(self):

        return (
            self.mrna_state.validate()
        )


    def describe(self):

        print()
        print("mRNA DEGRADATION")
        print("--------------------------------------------")

        print(
            f"Degradation steps:          "
            f"{self.step_count:,}"
        )

        print(
            f"Degradation events:         "
            f"{self.degradation_events:,}"
        )
