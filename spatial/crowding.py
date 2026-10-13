from __future__ import annotations

import cupy as cp


class CrowdingState:

    def __init__(
        self,
        grid,
        movement_exponent=1.0,
    ):

        self.grid = grid

        self.cell_voxels = (
            grid.cell_voxels
        )


        self.voxel_volume_nm3 = float(
            grid.voxel_volume_um3
            * 1e9
        )


        if self.voxel_volume_nm3 <= 0:

            raise ValueError(
                "Voxel volume must be > 0."
            )


        self.movement_exponent = float(
            movement_exponent
        )


        if self.movement_exponent <= 0:

            raise ValueError(
                "movement_exponent must be > 0."
            )


        self.static_volume_nm3 = cp.zeros(
            self.cell_voxels,
            dtype=cp.float32,
        )


        self.dna_volume_nm3 = cp.zeros(
            self.cell_voxels,
            dtype=cp.float32,
        )


        self.dynamic_volume_nm3 = cp.zeros(
            self.cell_voxels,
            dtype=cp.float32,
        )


        # ====================================================
        # Derived movement accessibility.
        #
        # 1.0 = completely accessible
        # 0.0 = inaccessible
        # ====================================================

        self.movement_accessibility = cp.ones(
            self.cell_voxels,
            dtype=cp.float32,
        )


        self.refresh_movement_accessibility()


    # ========================================================
    # Internal helpers
    # ========================================================

    def _prepare_addition(
        self,
        voxel_ids,
        volumes_nm3,
    ):

        voxel_ids = cp.asarray(
            voxel_ids,
            dtype=cp.int32,
        ).reshape(
            -1
        )


        if voxel_ids.size == 0:

            return (
                voxel_ids,
                cp.empty(
                    0,
                    dtype=cp.float32,
                ),
            )


        if bool(
            cp.any(
                voxel_ids < 0
            ).item()
        ):

            raise ValueError(
                "Voxel IDs cannot be negative."
            )


        if bool(
            cp.any(
                voxel_ids
                >= self.cell_voxels
            ).item()
        ):

            raise ValueError(
                "Voxel ID exceeds active cell grid."
            )


        volumes = cp.asarray(
            volumes_nm3,
            dtype=cp.float32,
        )


        if volumes.ndim == 0:

            volumes = cp.full(
                voxel_ids.shape,
                float(
                    volumes.item()
                ),
                dtype=cp.float32,
            )


        else:

            volumes = volumes.reshape(
                -1
            )


            if (
                volumes.shape
                != voxel_ids.shape
            ):

                raise ValueError(
                    "voxel_ids and volumes_nm3 must "
                    "have matching lengths."
                )


        if bool(
            cp.any(
                volumes < 0
            ).item()
        ):

            raise ValueError(
                "Occupied volume cannot be negative."
            )


        if bool(
            cp.any(
                ~cp.isfinite(
                    volumes
                )
            ).item()
        ):

            raise ValueError(
                "Occupied volume must be finite."
            )


        return (
            voxel_ids,
            volumes,
        )


    def _add_volume(
        self,
        target,
        voxel_ids,
        volumes_nm3,
    ):

        (
            voxel_ids,
            volumes,
        ) = self._prepare_addition(
            voxel_ids,
            volumes_nm3,
        )


        if voxel_ids.size == 0:

            return


        cp.add.at(
            target,
            voxel_ids,
            volumes,
        )


    # ========================================================
    # Static
    # ========================================================

    def add_static_volume(
        self,
        voxel_ids,
        volumes_nm3,
    ):

        self._add_volume(
            self.static_volume_nm3,
            voxel_ids,
            volumes_nm3,
        )


    def clear_static(self):

        self.static_volume_nm3.fill(
            0
        )


    # ========================================================
    # DNA
    # ========================================================

    def add_dna_volume(
        self,
        voxel_ids,
        volumes_nm3,
    ):

        self._add_volume(
            self.dna_volume_nm3,
            voxel_ids,
            volumes_nm3,
        )


    def clear_dna(self):

        self.dna_volume_nm3.fill(
            0
        )


    # ========================================================
    # Dynamic
    # ========================================================

    def add_dynamic_volume(
        self,
        voxel_ids,
        volumes_nm3,
    ):

        self._add_volume(
            self.dynamic_volume_nm3,
            voxel_ids,
            volumes_nm3,
        )


    def clear_dynamic(self):

        self.dynamic_volume_nm3.fill(
            0
        )


    # ========================================================
    # Occupancy
    # ========================================================

    @property
    def total_volume_nm3(self):

        return (
            self.static_volume_nm3
            + self.dna_volume_nm3
            + self.dynamic_volume_nm3
        )


    @property
    def occupied_fraction(self):

        return (
            self.total_volume_nm3
            / self.voxel_volume_nm3
        )


    @property
    def free_fraction(self):

        return (
            1.0
            - self.occupied_fraction
        )


    @property
    def available_volume_nm3(self):

        return (
            self.voxel_volume_nm3
            - self.total_volume_nm3
        )


    # ========================================================
    # Crowding-dependent accessibility
    # ========================================================

    def refresh_movement_accessibility(
        self,
    ):

        free = cp.clip(
            self.free_fraction,
            0.0,
            1.0,
        )


        cp.power(
            free,
            self.movement_exponent,
            out=self.movement_accessibility,
        )


        return self.movement_accessibility


    # ========================================================
    # Statistics
    # ========================================================

    @property
    def mean_occupied_fraction(self):

        return float(
            cp.mean(
                self.occupied_fraction,
                dtype=cp.float64,
            ).item()
        )


    @property
    def max_occupied_fraction(self):

        return float(
            cp.max(
                self.occupied_fraction
            ).item()
        )


    @property
    def mean_movement_accessibility(self):

        return float(
            cp.mean(
                self.movement_accessibility,
                dtype=cp.float64,
            ).item()
        )


    @property
    def minimum_movement_accessibility(self):

        return float(
            cp.min(
                self.movement_accessibility
            ).item()
        )


    @property
    def occupied_voxel_count(self):

        return int(
            cp.count_nonzero(
                self.total_volume_nm3
                > 0
            ).item()
        )


    @property
    def dna_occupied_voxel_count(self):

        return int(
            cp.count_nonzero(
                self.dna_volume_nm3
                > 0
            ).item()
        )


    @property
    def total_static_volume_nm3(self):

        return float(
            cp.sum(
                self.static_volume_nm3,
                dtype=cp.float64,
            ).item()
        )


    @property
    def total_dna_volume_nm3(self):

        return float(
            cp.sum(
                self.dna_volume_nm3,
                dtype=cp.float64,
            ).item()
        )


    @property
    def total_dynamic_volume_nm3(self):

        return float(
            cp.sum(
                self.dynamic_volume_nm3,
                dtype=cp.float64,
            ).item()
        )


    @property
    def state_bytes(self):

        return (
            self.static_volume_nm3.nbytes
            + self.dna_volume_nm3.nbytes
            + self.dynamic_volume_nm3.nbytes
            + self.movement_accessibility.nbytes
        )


    @property
    def state_kib(self):

        return (
            self.state_bytes
            / 1024
        )


    # ========================================================
    # Validation
    # ========================================================

    def validate(
        self,
        enforce_physical_capacity=True,
    ):

        expected_shape = (
            self.cell_voxels,
        )


        layers = (
            self.static_volume_nm3,
            self.dna_volume_nm3,
            self.dynamic_volume_nm3,
        )


        for values in layers:

            if (
                values.shape
                != expected_shape
            ):

                raise RuntimeError(
                    "Crowding-state shape mismatch."
                )


            if bool(
                cp.any(
                    ~cp.isfinite(
                        values
                    )
                ).item()
            ):

                raise RuntimeError(
                    "Non-finite occupied volume detected."
                )


            if bool(
                cp.any(
                    values < 0
                ).item()
            ):

                raise RuntimeError(
                    "Negative occupied volume detected."
                )


        if enforce_physical_capacity:

            tolerance = (
                self.voxel_volume_nm3
                * 1e-6
            )


            if bool(
                cp.any(
                    self.total_volume_nm3
                    >
                    self.voxel_volume_nm3
                    + tolerance
                ).item()
            ):

                raise RuntimeError(
                    "Voxel occupied volume exceeds "
                    "physical voxel capacity."
                )


        self.refresh_movement_accessibility()


        if bool(
            cp.any(
                self.movement_accessibility
                < 0
            ).item()
        ):

            raise RuntimeError(
                "Negative movement accessibility."
            )


        if bool(
            cp.any(
                self.movement_accessibility
                > 1
            ).item()
        ):

            raise RuntimeError(
                "Movement accessibility exceeds 1."
            )


        return True


    # ========================================================
    # Description
    # ========================================================

    def describe(self):

        self.refresh_movement_accessibility()


        print()
        print("CROWDING STATE")
        print("--------------------------------------------")


        print(
            f"Active voxels:              "
            f"{self.cell_voxels:,}"
        )


        print(
            f"Voxel volume:               "
            f"{self.voxel_volume_nm3:,.1f} nm^3"
        )


        print(
            f"Occupied voxels:            "
            f"{self.occupied_voxel_count:,}"
        )


        print(
            f"DNA-occupied voxels:        "
            f"{self.dna_occupied_voxel_count:,}"
        )


        print(
            f"Mean occupied fraction:     "
            f"{self.mean_occupied_fraction:.4f}"
        )


        print(
            f"Maximum occupied fraction:  "
            f"{self.max_occupied_fraction:.4f}"
        )


        print(
            f"Mean accessibility:         "
            f"{self.mean_movement_accessibility:.4f}"
        )


        print(
            f"Minimum accessibility:      "
            f"{self.minimum_movement_accessibility:.4f}"
        )


        print(
            f"Movement exponent:          "
            f"{self.movement_exponent:.3f}"
        )


        print(
            f"Static occupied volume:     "
            f"{self.total_static_volume_nm3:,.1f} nm^3"
        )


        print(
            f"DNA occupied volume:        "
            f"{self.total_dna_volume_nm3:,.1f} nm^3"
        )


        print(
            f"Dynamic occupied volume:    "
            f"{self.total_dynamic_volume_nm3:,.1f} nm^3"
        )


        print(
            f"State memory:               "
            f"{self.state_kib:.2f} KiB"
        )