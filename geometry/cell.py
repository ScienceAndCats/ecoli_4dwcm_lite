import numpy as np


def build_spherocylinder_mask(
    voxel_size_um: float,
    cell_length_um: float,
    cell_diameter_um: float,
):
    """
    Build an E. coli-like spherocylinder on a regular
    Cartesian voxel grid.

    Returns
    -------
    mask : ndarray[bool]
        True for voxels inside the cell.

    coordinates : tuple
        x, y, z coordinate arrays.
    """

    radius = cell_diameter_um / 2.0

    half_cylinder = (
        cell_length_um
        - cell_diameter_um
    ) / 2.0

    nx = int(
        np.ceil(
            (
                cell_length_um
                + 2 * voxel_size_um
            )
            / voxel_size_um
        )
    )

    ny = int(
        np.ceil(
            (
                cell_diameter_um
                + 2 * voxel_size_um
            )
            / voxel_size_um
        )
    )

    nz = ny

    x = (
        np.arange(nx)
        - (nx - 1) / 2
    ) * voxel_size_um

    y = (
        np.arange(ny)
        - (ny - 1) / 2
    ) * voxel_size_um

    z = (
        np.arange(nz)
        - (nz - 1) / 2
    ) * voxel_size_um

    X, Y, Z = np.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    axial_excess = np.maximum(
        np.abs(X) - half_cylinder,
        0.0,
    )

    mask = (
        axial_excess**2
        + Y**2
        + Z**2
    ) <= radius**2

    return mask, (x, y, z)
