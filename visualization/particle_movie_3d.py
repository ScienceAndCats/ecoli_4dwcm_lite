import argparse
import sys
from pathlib import Path

import cupy as cp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


# ============================================================
# Project imports
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


from chromosome.genome_loader import GenomeLoader
from chromosome.spatial import ChromosomeSpatialState
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
    default="outputs/particle_diffusion_3d.mp4",
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


parser.add_argument(
    "--dna-step-bp",
    type=int,
    default=5000,
    help=(
        "Genomic spacing between DNA scaffold samples."
    ),
)


parser.add_argument(
    "--rotate",
    action="store_true",
    help=(
        "Slowly rotate the camera during the movie."
    ),
)


args = parser.parse_args()


# ============================================================
# Configuration + simulation
# ============================================================

config = load_config(
    args.config
)


engine = SimulationEngine(
    config
)


state = (
    engine.particle_state
)

grid = (
    engine.grid
)

diffusion = (
    engine.particle_diffusion
)


# ============================================================
# Load chromosome
# ============================================================

chromosome = (
    GenomeLoader.from_genbank(
        config.biology.genome_file
    )
)


chromosome_spatial = (
    ChromosomeSpatialState(
        chromosome=chromosome,
        grid=grid,
        spatial_config=config.spatial,
        chromosome_config=config.chromosome,
    )
)


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
# Active voxel XYZ coordinates
# ============================================================

nx, ny, nz = (
    grid.shape
)


flat = (
    grid.active_flat
)


plane = (
    ny * nz
)


xi = (
    flat
    // plane
)


remainder = (
    flat
    - xi * plane
)


yi = (
    remainder
    // nz
)


zi = (
    remainder
    - yi * nz
)


dx = (
    grid.voxel_size_um
)


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
# Sample chromosome every N bp
#
# Each genomic position gets mapped onto the coarse chromosome
# scaffold, then snapped to its corresponding active 50-nm
# voxel.
# ============================================================

dna_bp_samples = np.arange(
    0,
    chromosome.length_bp,
    args.dna_step_bp,
    dtype=np.int64,
)


dna_voxel_samples = np.asarray(
    [
        chromosome_spatial.voxel_for_bp(
            int(bp)
        )

        for bp
        in dna_bp_samples
    ],
    dtype=np.int32,
)


# Many 5-kb genomic samples can occupy the same 50-nm voxel.
#
# The blue points therefore represent unique voxels containing
# our coarse chromosome scaffold.

dna_voxels = np.unique(
    dna_voxel_samples
)


dna_x = (
    voxel_x[
        dna_voxels
    ]
)


dna_y = (
    voxel_y[
        dna_voxels
    ]
)


dna_z = (
    voxel_z[
        dna_voxels
    ]
)


# ============================================================
# Extract current particles
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

        n_primary = int(
            primary_occ[
                voxel
            ]
        )


        if n_primary > 0:

            species_here = primary[
                voxel,
                :n_primary
            ]


            particle_voxels.extend(
                [voxel]
                * n_primary
            )


            particle_species.extend(
                species_here.tolist()
            )


        n_spill = int(
            spill_occ[
                voxel
            ]
        )


        if n_spill > 0:

            species_here = spill[
                voxel,
                :n_spill
            ]


            particle_voxels.extend(
                [voxel]
                * n_spill
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

fig = plt.figure(
    figsize=(10, 7)
)


ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# Faint whole-cell voxel cloud
# ============================================================

ax.scatter(
    voxel_x,
    voxel_y,
    voxel_z,
    s=1,
    color="gray",
    alpha=0.02,
)


# ============================================================
# Chromosome scaffold
#
# Blue = active voxels containing sampled DNA.
# ============================================================

ax.scatter(
    dna_x,
    dna_y,
    dna_z,
    s=16,
    color="blue",
    alpha=0.65,
    label=(
        f"DNA scaffold "
        f"({args.dna_step_bp:,} bp sampling)"
    ),
)


# ============================================================
# Axes
# ============================================================

ax.set_xlabel(
    "X / cell length (µm)"
)


ax.set_ylabel(
    "Y / width (µm)"
)


ax.set_zlabel(
    "Z / width (µm)"
)


margin = (
    grid.voxel_size_um
    * 2
)


ax.set_xlim(
    voxel_x.min()
    - margin,

    voxel_x.max()
    + margin,
)


ax.set_ylim(
    voxel_y.min()
    - margin,

    voxel_y.max()
    + margin,
)


ax.set_zlim(
    voxel_z.min()
    - margin,

    voxel_z.max()
    + margin,
)


# ============================================================
# Preserve rod-like proportions
# ============================================================

x_range = (
    voxel_x.max()
    - voxel_x.min()
)


y_range = (
    voxel_y.max()
    - voxel_y.min()
)


z_range = (
    voxel_z.max()
    - voxel_z.min()
)


ax.set_box_aspect(
    (
        x_range,
        y_range,
        z_range,
    )
)


# ============================================================
# Initial camera
# ============================================================

ax.view_init(
    elev=22,
    azim=-60,
)


# ============================================================
# Particle species
#
# Avoid blue so the chromosome remains visually distinct.
# ============================================================

particle_colors = [
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
]


particle_scatters = {}


for index, species in enumerate(
    engine.species.particle_species
):

    color = (
        particle_colors[
            index
            % len(
                particle_colors
            )
        ]
    )


    scatter = ax.scatter(
        [],
        [],
        [],
        s=30,
        alpha=0.9,
        color=color,
        label=species.name,
        depthshade=True,
    )


    particle_scatters[
        species.particle_id
    ] = scatter


ax.legend(
    loc="upper right",
    fontsize=8,
)


title = ax.set_title(
    "3D particle diffusion + chromosome scaffold"
)


# ============================================================
# Movie writer
# ============================================================

writer = FFMpegWriter(
    fps=args.fps,

    metadata={
        "title":
            "3D spatial particle diffusion "
            "with chromosome scaffold",
    },

    bitrate=3500,
)


# ============================================================
# Diagnostics
# ============================================================

print()
print("3D PARTICLE + DNA MOVIE")
print("--------------------------------------------")


print(
    f"Frames:                     "
    f"{args.frames:,}"
)


print(
    f"Steps / frame:              "
    f"{args.steps_per_frame:,}"
)


print(
    f"Genome length:              "
    f"{chromosome.length_bp:,} bp"
)


print(
    f"DNA sample spacing:         "
    f"{args.dna_step_bp:,} bp"
)


print(
    f"DNA genomic samples:        "
    f"{len(dna_bp_samples):,}"
)


print(
    f"Unique DNA voxels:          "
    f"{len(dna_voxels):,}"
)


print(
    f"Output:                     "
    f"{output_path}"
)


print()


# ============================================================
# Render
# ============================================================

with writer.saving(
    fig,
    str(output_path),
    dpi=140,
):

    for frame in range(
        args.frames
    ):

        # ====================================================
        # Advance particle diffusion
        # ====================================================

        if frame > 0:

            diffusion.run_steps(
                args.steps_per_frame
            )


        cp.cuda.Stream.null.synchronize()


        # ====================================================
        # Extract current particle positions
        # ====================================================

        (
            particle_voxels,
            particle_species,
        ) = extract_particles()


        # ====================================================
        # Update particle scatter plots
        # ====================================================

        for species in (
            engine.species.particle_species
        ):

            mask = (
                particle_species
                == species.particle_id
            )


            if np.any(
                mask
            ):

                voxels = (
                    particle_voxels[
                        mask
                    ]
                )


                xs = (
                    voxel_x[
                        voxels
                    ]
                )


                ys = (
                    voxel_y[
                        voxels
                    ]
                )


                zs = (
                    voxel_z[
                        voxels
                    ]
                )


            else:

                xs = np.empty(
                    0
                )

                ys = np.empty(
                    0
                )

                zs = np.empty(
                    0
                )


            particle_scatters[
                species.particle_id
            ]._offsets3d = (
                xs,
                ys,
                zs,
            )


        # ====================================================
        # Optional camera rotation
        # ====================================================

        if args.rotate:

            ax.view_init(
                elev=22,

                azim=(
                    -60
                    + frame * 0.75
                ),
            )


        # ====================================================
        # Simulation-time label
        # ====================================================

        biological_time = (
            diffusion.step_count
            * engine.particle_dt_s
        )


        title.set_text(
            "3D particle diffusion + chromosome scaffold\n"

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
    f"Movie:                      "
    f"{output_path}"
)


print(
    f"DNA voxels shown:           "
    f"{len(dna_voxels):,}"
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
    f"Biological time:            "
    f"{diffusion.step_count * engine.particle_dt_s:.4f} s"
)


print(
    f"Hard overflow events:       "
    f"{diffusion.hard_overflow_count:,}"
)