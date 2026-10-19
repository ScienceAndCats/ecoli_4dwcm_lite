import matplotlib

matplotlib.use(
    "Agg"
)

import numpy as np
import pytest

from spatial.config import load_config
from spatial.crowding import CrowdingState
from spatial.crowding_visualization import (
    CrowdingVisualizer,
)
from spatial.grid import SpatialGrid


def build_context():

    config = load_config(
        "configs/default.yaml"
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    crowding = CrowdingState(
        grid=grid,

        movement_exponent=(
            config.crowding
            .movement_exponent
        ),
    )


    visualizer = CrowdingVisualizer(
        crowding
    )


    return (
        config,
        grid,
        crowding,
        visualizer,
    )


def test_active_values_return_correct_components():

    (
        config,
        grid,
        crowding,
        visualizer,
    ) = build_context()


    voxel = 100


    # ========================================================
    # 50-nm voxel:
    #
    # volume = 125,000 nm^3
    #
    # DNA:
    #     25,000 = 20%
    #
    # Dynamic:
    #     37,500 = 30%
    #
    # Total:
    #     62,500 = 50%
    #
    # Accessibility at exponent 1:
    #     50%
    # ========================================================

    crowding.add_dna_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=25000.0,
    )


    crowding.add_dynamic_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=37500.0,
    )


    crowding.refresh_movement_accessibility()


    assert (
        visualizer.active_values(
            "dna"
        )[
            voxel
        ]
        == pytest.approx(
            0.20
        )
    )


    assert (
        visualizer.active_values(
            "dynamic"
        )[
            voxel
        ]
        == pytest.approx(
            0.30
        )
    )


    assert (
        visualizer.active_values(
            "total"
        )[
            voxel
        ]
        == pytest.approx(
            0.50
        )
    )


    assert (
        visualizer.active_values(
            "accessibility"
        )[
            voxel
        ]
        == pytest.approx(
            0.50
        )
    )


def test_active_voxel_maps_to_correct_xyz_position():

    (
        config,
        grid,
        crowding,
        visualizer,
    ) = build_context()


    voxel = 250


    crowding.add_dynamic_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=62500.0,
    )


    volume_grid = (
        visualizer.volume_grid(
            "dynamic"
        )
    )


    assert (
        volume_grid.shape
        == grid.shape
    )


    flat_id = int(
        grid.active_flat[
            voxel
        ]
    )


    xyz = np.unravel_index(
        flat_id,
        grid.shape,
    )


    assert (
        volume_grid[
            xyz
        ]
        == pytest.approx(
            0.50
        )
    )


def test_inactive_grid_positions_are_zero():

    (
        config,
        grid,
        crowding,
        visualizer,
    ) = build_context()


    values = (
        visualizer.volume_grid(
            "total"
        )
    )


    assert np.all(
        values[
            ~grid.mask
        ]
        == 0
    )


def test_max_projection_preserves_peak_occupancy():

    (
        config,
        grid,
        crowding,
        visualizer,
    ) = build_context()


    voxel = 300


    crowding.add_dynamic_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=100000.0,
    )


    projection = (
        visualizer.max_projection(
            component="dynamic",
            axis="z",
        )
    )


    assert (
        np.nanmax(
            projection
        )
        == pytest.approx(
            0.80
        )
    )


def test_overview_image_is_written(
    tmp_path,
):

    (
        config,
        grid,
        crowding,
        visualizer,
    ) = build_context()


    crowding.add_dna_volume(
        voxel_ids=[
            100,
            200,
            300,
        ],

        volumes_nm3=[
            10000.0,
            20000.0,
            30000.0,
        ],
    )


    crowding.add_dynamic_volume(
        voxel_ids=[
            400,
            500,
        ],

        volumes_nm3=[
            40000.0,
            50000.0,
        ],
    )


    output = (
        tmp_path
        / "crowding_overview.png"
    )


    result = visualizer.save_overview(
        output
    )


    assert result.exists()


    assert (
        result.stat().st_size
        > 0
    )
