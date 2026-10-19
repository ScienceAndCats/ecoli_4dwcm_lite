import json
import shutil
from pathlib import Path

import cupy as cp
import numpy as np


class CheckpointManager:
    """
    Full restartable simulator checkpoints.

    Checkpoints are deliberately uncompressed because our
    Phase 0.5 benchmark showed that compression cost much more
    CPU time than the storage savings justified.
    """

    FORMAT_VERSION = 1


    def __init__(
        self,
        output_dir,
        compression=False,
    ):

        if compression:

            raise ValueError(
                "Compressed checkpoints are currently disabled. "
                "Our benchmark showed poor compression benefit "
                "relative to the write-time cost."
            )


        self.output_dir = Path(
            output_dir
        )


        self.checkpoint_dir = (
            self.output_dir
            / "checkpoints"
        )


    def save(
        self,
        engine,
    ):

        cp.cuda.Stream.null.synchronize()


        tick = (
            engine.scheduler.current_tick
        )


        final_path = (
            self.checkpoint_dir
            /
            f"checkpoint_tick_{tick:012d}"
        )


        temporary_path = (
            self.checkpoint_dir
            /
            f".checkpoint_tick_{tick:012d}.tmp"
        )


        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        if temporary_path.exists():

            shutil.rmtree(
                temporary_path
            )


        temporary_path.mkdir(
            parents=True
        )


        particles = (
            engine.particle_state
        )


        fields = (
            engine.field_state
        )


        reactions = (
            engine.reaction_engine
        )


        # ====================================================
        # Dynamic arrays
        # ====================================================

        np.save(
            temporary_path
            / "particle_primary.npy",

            cp.asnumpy(
                particles.primary
            ),

            allow_pickle=False,
        )


        np.save(
            temporary_path
            / "particle_spill.npy",

            cp.asnumpy(
                particles.spill
            ),

            allow_pickle=False,
        )


        np.save(
            temporary_path
            / "particle_primary_occupancy.npy",

            cp.asnumpy(
                particles.primary_occupancy
            ),

            allow_pickle=False,
        )


        np.save(
            temporary_path
            / "particle_spill_occupancy.npy",

            cp.asnumpy(
                particles.spill_occupancy
            ),

            allow_pickle=False,
        )


        np.save(
            temporary_path
            / "fields.npy",

            cp.asnumpy(
                fields.values
            ),

            allow_pickle=False,
        )


        np.save(
            temporary_path
            / "reaction_event_counts.npy",

            cp.asnumpy(
                reactions.event_counts
            ),

            allow_pickle=False,
        )


        # ====================================================
        # Metadata and counters
        # ====================================================

        metadata = {
            "format_version":
                self.FORMAT_VERSION,

            "current_tick":
                int(
                    engine.scheduler.current_tick
                ),

            "base_dt_s":
                float(
                    engine.base_dt_s
                ),

            "voxel_size_nm":
                float(
                    engine.config.spatial.voxel_size_nm
                ),

            "cell_voxels":
                int(
                    engine.grid.cell_voxels
                ),

            "particle_species_names":
                [
                    species.name
                    for species
                    in engine.species.particle_species
                ],

            "field_species_names":
                [
                    species.name
                    for species
                    in engine.species.field_species
                ],

            "reaction_names":
                [
                    reaction.name
                    for reaction
                    in engine.reactions.ordered_definitions
                ],

            "primary_slots":
                int(
                    particles.primary_slots
                ),

            "spill_slots":
                int(
                    particles.spill_slots
                ),

            "particle_initial_count":
                int(
                    particles.initial_particle_count
                ),

            "particle_diffusion_step_count":
                int(
                    engine.particle_diffusion.step_count
                    if engine.particle_diffusion
                    is not None
                    else 0
                ),

            "particle_diffusion_spill_writes":
                int(
                    engine.particle_diffusion.spill_write_count
                    if engine.particle_diffusion
                    is not None
                    else 0
                ),

            # Keep the metadata key name unchanged so existing
            # format-version-1 checkpoints remain compatible.
            #
            # Internally this now represents slot-capacity
            # rejected moves.

            "particle_diffusion_rejected_moves":
                int(
                    engine.particle_diffusion.rejected_move_count
                    if engine.particle_diffusion
                    is not None
                    else 0
                ),

            "particle_diffusion_hard_overflow":
                int(
                    engine.particle_diffusion.hard_overflow_count
                    if engine.particle_diffusion
                    is not None
                    else 0
                ),

            "field_diffusion_step_count":
                int(
                    engine.field_diffusion.step_count
                    if engine.field_diffusion
                    is not None
                    else 0
                ),

            "reaction_step_count":
                int(
                    reactions.step_count
                ),

            "reaction_starting_particle_count":
                int(
                    reactions.starting_particle_count
                ),

            "scheduler_task_calls":
                {
                    task.name:
                        int(
                            task.calls
                        )

                    for task
                    in engine.scheduler.tasks
                },
        }


        with (
            temporary_path
            / "metadata.json"
        ).open(
            "w"
        ) as handle:

            json.dump(
                metadata,
                handle,
                indent=2,
            )


        # ====================================================
        # Atomic-ish finalization
        # ====================================================

        if final_path.exists():

            shutil.rmtree(
                final_path
            )


        temporary_path.rename(
            final_path
        )


        return final_path


    def load(
        self,
        engine,
        path,
    ):

        path = Path(
            path
        )


        with (
            path
            / "metadata.json"
        ).open(
            "r"
        ) as handle:

            metadata = json.load(
                handle
            )


        # ====================================================
        # Compatibility checks
        # ====================================================

        if (
            metadata["format_version"]
            != self.FORMAT_VERSION
        ):

            raise ValueError(
                "Unsupported checkpoint format."
            )


        if (
            metadata["cell_voxels"]
            != engine.grid.cell_voxels
        ):

            raise ValueError(
                "Checkpoint cell grid does not match "
                "the current simulation."
            )


        if (
            metadata["primary_slots"]
            != engine.particle_state.primary_slots
        ):

            raise ValueError(
                "Checkpoint primary slot count does not match."
            )


        if (
            metadata["spill_slots"]
            != engine.particle_state.spill_slots
        ):

            raise ValueError(
                "Checkpoint spill slot count does not match."
            )


        particle_names = [
            species.name
            for species
            in engine.species.particle_species
        ]


        field_names = [
            species.name
            for species
            in engine.species.field_species
        ]


        reaction_names = [
            reaction.name
            for reaction
            in engine.reactions.ordered_definitions
        ]


        if (
            metadata["particle_species_names"]
            != particle_names
        ):

            raise ValueError(
                "Checkpoint particle species do not match."
            )


        if (
            metadata["field_species_names"]
            != field_names
        ):

            raise ValueError(
                "Checkpoint field species do not match."
            )


        if (
            metadata["reaction_names"]
            != reaction_names
        ):

            raise ValueError(
                "Checkpoint reactions do not match."
            )


        # ====================================================
        # Restore particles
        # ====================================================

        particles = (
            engine.particle_state
        )


        particles.primary[...] = cp.asarray(
            np.load(
                path
                / "particle_primary.npy"
            )
        )


        particles.spill[...] = cp.asarray(
            np.load(
                path
                / "particle_spill.npy"
            )
        )


        particles.primary_occupancy[...] = cp.asarray(
            np.load(
                path
                / "particle_primary_occupancy.npy"
            )
        )


        particles.spill_occupancy[...] = cp.asarray(
            np.load(
                path
                / "particle_spill_occupancy.npy"
            )
        )


        particles.clear_next()


        # ====================================================
        # Restore concentration fields
        # ====================================================

        engine.field_state.values[...] = cp.asarray(
            np.load(
                path
                / "fields.npy"
            )
        )


        engine.field_state.clear_next()


        # ====================================================
        # Restore reaction state
        # ====================================================

        engine.reaction_engine.event_counts[...] = (
            cp.asarray(
                np.load(
                    path
                    / "reaction_event_counts.npy"
                )
            )
        )


        engine.reaction_engine.step_count = int(
            metadata[
                "reaction_step_count"
            ]
        )


        engine.reaction_engine.starting_particle_count = int(
            metadata[
                "reaction_starting_particle_count"
            ]
        )


        # ====================================================
        # Restore diffusion counters
        # ====================================================

        if (
            engine.particle_diffusion
            is not None
        ):

            engine.particle_diffusion.step_count = int(
                metadata[
                    "particle_diffusion_step_count"
                ]
            )


            engine.particle_diffusion.spill_writes[
                0
            ] = np.uint64(
                metadata[
                    "particle_diffusion_spill_writes"
                ]
            )


            # The old checkpoint field is retained, but the
            # corresponding ParticleDiffusion array is now
            # named slot_rejected_moves.

            engine.particle_diffusion.slot_rejected_moves[
                0
            ] = np.uint64(
                metadata[
                    "particle_diffusion_rejected_moves"
                ]
            )


            engine.particle_diffusion.hard_overflow[
                0
            ] = np.uint64(
                metadata[
                    "particle_diffusion_hard_overflow"
                ]
            )


        if (
            engine.field_diffusion
            is not None
        ):

            engine.field_diffusion.step_count = int(
                metadata[
                    "field_diffusion_step_count"
                ]
            )


        # ====================================================
        # Restore simulation clock
        # ====================================================

        engine.scheduler.current_tick = int(
            metadata[
                "current_tick"
            ]
        )


        saved_calls = (
            metadata[
                "scheduler_task_calls"
            ]
        )


        for task in (
            engine.scheduler.tasks
        ):

            task.calls = int(
                saved_calls.get(
                    task.name,
                    0,
                )
            )


        engine.validate()


        return path