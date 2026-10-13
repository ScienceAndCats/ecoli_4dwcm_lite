import cupy as cp
import pytest

from spatial.config import (
    load_config,
)
from spatial.crowding import (
    CrowdingState,
)
from spatial.grid import (
    SpatialGrid,
)


def build_crowding_state():

    config = load_config(
        "configs/default.yaml"
    )


    grid = SpatialGrid.from_config(
        config.spatial
    )


    crowding = CrowdingState(
        grid=grid
    )


    return (
        config,
        grid,
        crowding,
    )


def test_crowding_state_initializes_empty():

    (
        config,
        grid,
        crowding,
    ) = build_crowding_state()


    # ========================================================
    # 50-nm voxel:
    #
    # 50^3 = 125,000 nm^3
    # ========================================================

    assert (
        crowding.voxel_volume_nm3
        == pytest.approx(
            125000.0
        )
    )


    assert (
        crowding.static_volume_nm3.shape
        ==
        (
            grid.cell_voxels,
        )
    )


    assert (
        crowding.dynamic_volume_nm3.shape
        ==
        (
            grid.cell_voxels,
        )
    )


    assert (
        crowding.occupied_voxel_count
        == 0
    )


    assert (
        crowding.mean_occupied_fraction
        == pytest.approx(
            0.0
        )
    )


    assert (
        crowding.max_occupied_fraction
        == pytest.approx(
            0.0
        )
    )


    assert crowding.validate()


def test_static_and_dynamic_volume_accumulate():

    (
        config,
        grid,
        crowding,
    ) = build_crowding_state()


    voxel = 100


    # ========================================================
    # Static:
    #
    # 25,000 nm^3
    # = 20% of a 50-nm voxel
    # ========================================================

    crowding.add_static_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=25000.0,
    )


    # ========================================================
    # Dynamic:
    #
    # 12,500 nm^3
    # = 10%
    #
    # Total occupancy should therefore be 30%.
    # ========================================================

    crowding.add_dynamic_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=12500.0,
    )


    occupied = float(
        crowding.occupied_fraction[
            voxel
        ].item()
    )


    free = float(
        crowding.free_fraction[
            voxel
        ].item()
    )


    assert occupied == pytest.approx(
        0.30,
        rel=1e-6,
    )


    assert free == pytest.approx(
        0.70,
        rel=1e-6,
    )


    assert (
        crowding.occupied_voxel_count
        == 1
    )


    assert crowding.validate()


def test_duplicate_voxel_additions_accumulate():

    (
        config,
        grid,
        crowding,
    ) = build_crowding_state()


    voxel = 200


    # Three separate objects all occupy the same voxel.

    crowding.add_dynamic_volume(
        voxel_ids=[
            voxel,
            voxel,
            voxel,
        ],

        volumes_nm3=[
            1000.0,
            2000.0,
            3000.0,
        ],
    )


    observed = float(
        crowding.dynamic_volume_nm3[
            voxel
        ].item()
    )


    assert observed == pytest.approx(
        6000.0
    )


    assert crowding.validate()


def test_dynamic_volume_can_be_rebuilt():

    (
        config,
        grid,
        crowding,
    ) = build_crowding_state()


    voxel = 300


    crowding.add_static_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=10000.0,
    )


    crowding.add_dynamic_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=20000.0,
    )


    assert (
        float(
            crowding.total_volume_nm3[
                voxel
            ].item()
        )
        == pytest.approx(
            30000.0
        )
    )


    # Dynamic occupancy can be cleared/rebuilt every time
    # mobile molecular state changes.

    crowding.clear_dynamic()


    assert (
        float(
            crowding.static_volume_nm3[
                voxel
            ].item()
        )
        == pytest.approx(
            10000.0
        )
    )


    assert (
        float(
            crowding.dynamic_volume_nm3[
                voxel
            ].item()
        )
        == pytest.approx(
            0.0
        )
    )


    assert (
        float(
            crowding.total_volume_nm3[
                voxel
            ].item()
        )
        == pytest.approx(
            10000.0
        )
    )


    assert crowding.validate()


def test_overfilled_voxel_is_detected():

    (
        config,
        grid,
        crowding,
    ) = build_crowding_state()


    voxel = 400


    # Put 150,000 nm^3 into a voxel whose physical volume is
    # only 125,000 nm^3.

    crowding.add_dynamic_volume(
        voxel_ids=[
            voxel
        ],

        volumes_nm3=150000.0,
    )


    with pytest.raises(
        RuntimeError,
        match="exceeds physical voxel capacity",
    ):

        crowding.validate()


def test_multiple_voxels_have_independent_occupancy():

    (
        config,
        grid,
        crowding,
    ) = build_crowding_state()


    crowding.add_dynamic_volume(
        voxel_ids=[
            10,
            20,
            30,
        ],

        volumes_nm3=[
            12500.0,
            25000.0,
            62500.0,
        ],
    )


    fractions = cp.asnumpy(
        crowding.occupied_fraction[
            [
                10,
                20,
                30,
            ]
        ]
    )


    assert fractions[
        0
    ] == pytest.approx(
        0.10
    )


    assert fractions[
        1
    ] == pytest.approx(
        0.20
    )


    assert fractions[
        2
    ] == pytest.approx(
        0.50
    )


    assert crowding.validate()
