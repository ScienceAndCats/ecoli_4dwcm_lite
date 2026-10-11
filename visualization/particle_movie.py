import argparse
import sys
from pathlib import Path

import cupy as cp
import matplotlib

# Allows movie generation without opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


from coupling.engine import SimulationEngine
from spatial.config import load_config


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    default="configs/default.yaml",
)

parser.add_argument(
    "--output",
    default="outputs/particle_diffusion.mp4",
)

parser.add_argument(
    "--frames",
    type=int,
    default=120,
)

parser.add_argument(
    "--steps-per-frame",
    type=int,
    default=20,
)

parser.add_argument(
    "--fps",
    type=int,
    default=20,
)

args = parser.parse_args()


# ============================================================
# Build simulation
# ============================================================

config = load_config(
    args.config
)

engine = SimulationEngine(
    config
)

state = engine.particle_state
grid = engine.grid
diffusion = engine.particle_diffusion


# ============================================================
# Output path
# ============================================================

output_path = Path(
    args.output
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Convert compact voxel IDs into physical XYZ coordinates
# ============================================================

nx, ny, nz = grid.shape

flat = grid.active_flat

plane = ny * nz

xi = flat // plane

remainder = (
    flat
    - xi * plane
)

yi = remainder // nz

zi = (
    remainder
    - yi * nz
)


dx = grid.voxel_size_um


voxel_x = (
    xi
    - (nx - 1) / 2
) * dx

voxel_y = (
    yi
    - (ny - 1) / 2
) * dx

voxel_z = (
    zi
    - (nz - 1) / 2
) * dx


# ============================================================
# Build a simple 2D cell background
#
# We are viewing the cell in the X-Y plane.
# Z is projected away.
# ============================================================

cell_xy = np.unique(
    np.column_stack(
        (
            voxel_x,
            voxel_y,
        )
    ),
    axis=0,
)


# ============================================================
# Extract current particles from packed storage
# ============================================================

def extract_particles():

    primary = cp.asnumpy(
        state.primary
    )

    spill = cp.asnumpy(
        state.spill
    )

    primary_occ = cp.asnumpy(
        state.primary_occupancy
    )

    spill_occ = cp.asnumpy(
        state.spill_occupancy
    )


    particle_voxels = []
    particle_species = []


    for voxel in range(
        state.cell_voxels
    ):

        np_primary = int(
            primary_occ[
                voxel
            ]
        )

        if np_primary > 0:

            species_here = primary[
                voxel,
                :np_primary
            ]

            particle_voxels.extend(
                [voxel]
                * np_primary
            )

            particle_species.extend(
                species_here.tolist()
            )


        np_spill = int(
            spill_occ[
                voxel
            ]
        )

        if np_spill > 0:

            species_here = spill[
                voxel,
                :np_spill
            ]

            particle_voxels.extend(
                [voxel]
                * np_spill
            )

            particle_species.extend(
                species_here.tolist()
            )


    return (
        np.asarray(
            particle_voxels,
            dtype=np.int32,
        ),
        np.asarray(
            particle_species,
            dtype=np.uint16,
        ),
    )


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)


# Cell voxel background.

ax.scatter(
    cell_xy[:, 0],
    cell_xy[:, 1],
    s=9,
    color="lightgray",
    alpha=0.4,
    linewidths=0,
)


ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    "Cell length (µm)"
)

ax.set_ylabel(
    "Cell width (µm)"
)


margin = (
    grid.voxel_size_um
    * 2
)

ax.set_xlim(
    voxel_x.min() - margin,
    voxel_x.max() + margin,
)

ax.set_ylim(
    voxel_y.min() - margin,
    voxel_y.max() + margin,
)


# ============================================================
# One scatter object per particle species
# ============================================================

particle_scatters = {}


colors = plt.cm.tab10(
    np.linspace(
        0,
        1,
        max(
            engine.species.particle_count,
            1,
        ),
    )
)


for index, species in enumerate(
    engine.species.particle_species
):

    scatter = ax.scatter(
        [],
        [],
        s=28,
        alpha=0.85,
        color=colors[index],
        label=species.name,
    )

    particle_scatters[
        species.particle_id
    ] = scatter


ax.legend(
    loc="upper right",
    fontsize=8,
)


title = ax.set_title(
    "Particle diffusion"
)


# ============================================================
# Movie metadata
# ============================================================

writer = FFMpegWriter(
    fps=args.fps,
    metadata={
        "title":
            "Spatial particle diffusion",

        "artist":
            "ecoli_4dwcm_lite",
    },
    bitrate=3000,
)


# ============================================================
# Render movie
# ============================================================

print()
print("PARTICLE MOVIE")
print("--------------------------------------------")

print(
    f"Frames:                     "
    f"{args.frames:,}"
)

print(
    f"Diffusion steps / frame:    "
    f"{args.steps_per_frame:,}"
)

print(
    f"FPS:                        "
    f"{args.fps}"
)

print(
    f"Output:                     "
    f"{output_path}"
)

print()


with writer.saving(
    fig,
    str(output_path),
    dpi=140,
):

    for frame in range(
        args.frames
    ):

        # ====================================================
        # Advance simulation
        # ====================================================

        if frame > 0:

            diffusion.run_steps(
                args.steps_per_frame
            )


        cp.cuda.Stream.null.synchronize()


        # ====================================================
        # Get current positions
        # ====================================================

        (
            particle_voxels,
            particle_species,
        ) = extract_particles()


        # ====================================================
        # Update species scatter plots
        # ====================================================

        for species in (
            engine.species.particle_species
        ):

            mask = (
                particle_species
                == species.particle_id
            )


            if np.any(mask):

                voxels = (
                    particle_voxels[
                        mask
                    ]
                )

                positions = np.column_stack(
                    (
                        voxel_x[
                            voxels
                        ],

                        voxel_y[
                            voxels
                        ],
                    )
                )

            else:

                positions = np.empty(
                    (0, 2)
                )


            particle_scatters[
                species.particle_id
            ].set_offsets(
                positions
            )


        # ====================================================
        # Simulation-time label
        # ====================================================

        biological_time = (
            diffusion.step_count
            * engine.base_dt_s
        )


        title.set_text(
            "Particle diffusion\n"
            f"biological time = "
            f"{biological_time:.4f} s   |   "
            f"particles = "
            f"{state.total_particle_count:,}"
        )


        writer.grab_frame()


        if (
            frame % 20 == 0
            or frame
            == args.frames - 1
        ):

            print(
                f"Rendered frame "
                f"{frame + 1:4d}"
                f"/{args.frames}"
            )


plt.close(
    fig
)


# ============================================================
# Final validation
# ============================================================

diffusion.validate()


print()
print("DONE")
print("--------------------------------------------")

print(
    f"Movie written to:           "
    f"{output_path}"
)

print(
    f"Final particles:            "
    f"{state.total_particle_count:,}"
)

print(
    f"Diffusion steps:            "
    f"{diffusion.step_count:,}"
)

print(
    f"Biological time shown:      "
    f"{diffusion.step_count * engine.base_dt_s:.4f} s"
)

print(
    f"Hard overflow events:       "
    f"{diffusion.hard_overflow_count:,}"
)
