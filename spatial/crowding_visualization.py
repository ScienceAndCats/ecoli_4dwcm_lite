from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np


class CrowdingVisualizer:
    """
    Convert active-voxel crowding arrays back into the
    full 3-D Cartesian cell grid and visualize them.

    The simulator stores crowding only for active cell voxels:

        [active_voxel]

    For visualization we reconstruct:

        [x, y, z]

    using SpatialGrid.active_flat.


    The default visualization uses a maximum projection:

        3-D cell
            ↓
        maximum value through one axis
            ↓
        2-D occupancy map

    A maximum projection is useful for crowding because a
    strongly crowded local voxel is not averaged away.
    """

    VALID_COMPONENTS = (
        "total",
        "dna",
        "dynamic",
        "static",
        "accessibility",
    )


    def __init__(
        self,
        crowding_state,
    ):

        self.crowding_state = (
            crowding_state
        )

        self.grid = (
            crowding_state.grid
        )


    # ========================================================
    # Active voxel values
    # ========================================================

    def active_values(
        self,
        component="total",
    ):

        component = str(
            component
        ).lower()


        if component == "total":

            values = (
                self.crowding_state
                .occupied_fraction
            )


        elif component == "dna":

            values = (
                self.crowding_state
                .dna_volume_nm3
                / self.crowding_state
                .voxel_volume_nm3
            )


        elif component == "dynamic":

            values = (
                self.crowding_state
                .dynamic_volume_nm3
                / self.crowding_state
                .voxel_volume_nm3
            )


        elif component == "static":

            values = (
                self.crowding_state
                .static_volume_nm3
                / self.crowding_state
                .voxel_volume_nm3
            )


        elif component == "accessibility":

            self.crowding_state.refresh_movement_accessibility()

            values = (
                self.crowding_state
                .movement_accessibility
            )


        else:

            raise ValueError(
                "Unknown crowding visualization "
                f"component: {component}. "
                "Expected one of: "
                + ", ".join(
                    self.VALID_COMPONENTS
                )
            )


        return cp.asnumpy(
            values
        ).astype(
            np.float32,
            copy=False,
        )


    # ========================================================
    # Active array → full XYZ grid
    # ========================================================

    def volume_grid(
        self,
        component="total",
    ):

        values = self.active_values(
            component
        )


        expected = (
            self.grid.cell_voxels,
        )


        if (
            values.shape
            != expected
        ):

            raise RuntimeError(
                "Crowding visualization state shape "
                f"mismatch: expected {expected}, "
                f"found {values.shape}."
            )


        flat = np.zeros(
            self.grid.bounding_voxels,
            dtype=np.float32,
        )


        flat[
            self.grid.active_flat
        ] = values


        return flat.reshape(
            self.grid.shape
        )


    # ========================================================
    # Cell mask in XYZ form
    # ========================================================

    def mask_grid(
        self,
    ):

        return np.asarray(
            self.grid.mask,
            dtype=bool,
        )


    # ========================================================
    # Maximum projection
    # ========================================================

    def max_projection(
        self,
        component="total",
        axis="z",
    ):

        axis = str(
            axis
        ).lower()


        axis_lookup = {
            "x": 0,
            "y": 1,
            "z": 2,
        }


        if axis not in axis_lookup:

            raise ValueError(
                "Projection axis must be "
                "'x', 'y', or 'z'."
            )


        axis_index = (
            axis_lookup[
                axis
            ]
        )


        values = self.volume_grid(
            component
        )


        mask = self.mask_grid()


        projection = np.max(
            values,
            axis=axis_index,
        )


        projection_mask = np.any(
            mask,
            axis=axis_index,
        )


        projection = projection.astype(
            np.float32,
            copy=True,
        )


        projection[
            ~projection_mask
        ] = np.nan


        return projection


    # ========================================================
    # Physical plotting extent
    # ========================================================

    def projection_extent_um(
        self,
        axis="z",
    ):

        axis = str(
            axis
        ).lower()


        nx, ny, nz = (
            self.grid.shape
        )


        dx = float(
            self.grid.voxel_size_um
        )


        lengths = {
            "x": nx * dx,
            "y": ny * dx,
            "z": nz * dx,
        }


        if axis == "z":

            horizontal = (
                lengths["x"]
            )

            vertical = (
                lengths["y"]
            )


        elif axis == "y":

            horizontal = (
                lengths["x"]
            )

            vertical = (
                lengths["z"]
            )


        elif axis == "x":

            horizontal = (
                lengths["y"]
            )

            vertical = (
                lengths["z"]
            )


        else:

            raise ValueError(
                "Projection axis must be "
                "'x', 'y', or 'z'."
            )


        return (
            -horizontal / 2.0,
            horizontal / 2.0,
            -vertical / 2.0,
            vertical / 2.0,
        )


    # ========================================================
    # Save overview image
    # ========================================================

    def save_overview(
        self,
        path,
        axis="z",
        dpi=180,
    ):

        # Import matplotlib only when actual plotting is
        # requested. Numerical simulation code therefore does
        # not require matplotlib merely to import this module.

        import matplotlib.pyplot as plt


        path = Path(
            path
        )


        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        panels = (
            (
                "total",
                "Total occupancy",
            ),

            (
                "dna",
                "DNA occupancy",
            ),

            (
                "dynamic",
                "Dynamic occupancy",
            ),

            (
                "accessibility",
                "Movement accessibility",
            ),
        )


        extent = self.projection_extent_um(
            axis
        )


        figure, axes = plt.subplots(
            2,
            2,
            figsize=(
                10,
                6,
            ),
            constrained_layout=True,
        )


        for plot_axis, (
            component,
            title,
        ) in zip(
            axes.flat,
            panels,
        ):

            projection = self.max_projection(
                component=component,
                axis=axis,
            )


            # max_projection retains array ordering from the
            # simulation grid. Transpose for conventional
            # horizontal/vertical image display.

            image = plot_axis.imshow(
                projection.T,

                origin="lower",

                extent=extent,

                vmin=0.0,
                vmax=1.0,

                aspect="equal",
            )


            plot_axis.set_title(
                title
            )


            plot_axis.set_xlabel(
                "Position (µm)"
            )


            plot_axis.set_ylabel(
                "Position (µm)"
            )


            colorbar = figure.colorbar(
                image,
                ax=plot_axis,
            )


            if (
                component
                == "accessibility"
            ):

                colorbar.set_label(
                    "Movement accessibility"
                )


            else:

                colorbar.set_label(
                    "Occupied fraction"
                )


        figure.suptitle(
            "Voxel crowding — "
            f"maximum projection along {axis.upper()}"
        )


        figure.savefig(
            path,
            dpi=int(
                dpi
            ),
        )


        plt.close(
            figure
        )


        return path
