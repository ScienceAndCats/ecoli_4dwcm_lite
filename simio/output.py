from pathlib import Path

import cupy as cp
import numpy as np


class OutputManager:
    """
    Lightweight analysis output.

    Only explicitly selected species are written.

    This output is intended for analysis/visualization,
    not restarting the simulation.
    """

    def __init__(
        self,
        output_dir,
        selected_fields,
        selected_particles,
    ):

        self.output_dir = Path(
            output_dir
        )

        self.analysis_dir = (
            self.output_dir
            / "analysis"
        )

        self.selected_fields = tuple(
            selected_fields
        )

        self.selected_particles = tuple(
            selected_particles
        )

        self._resolved = False

        self.field_ids = ()
        self.particle_ids = ()


    def _resolve_species(
        self,
        engine,
    ):

        field_ids = []

        for name in self.selected_fields:

            species = (
                engine.species.get(
                    name
                )
            )

            if species.field_id is None:

                raise ValueError(
                    f"{name} is not a field species."
                )

            field_ids.append(
                species.field_id
            )


        particle_ids = []

        for name in self.selected_particles:

            species = (
                engine.species.get(
                    name
                )
            )

            if species.particle_id is None:

                raise ValueError(
                    f"{name} is not a particle species."
                )

            particle_ids.append(
                species.particle_id
            )


        self.field_ids = tuple(
            field_ids
        )

        self.particle_ids = tuple(
            particle_ids
        )

        self._resolved = True


    def _write_manifest(
        self,
        engine,
    ):

        path = (
            self.output_dir
            / "manifest.npz"
        )

        if path.exists():
            return


        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        np.savez(
            path,

            voxel_size_um=np.asarray(
                engine.grid.voxel_size_um,
                dtype=np.float64,
            ),

            grid_shape=np.asarray(
                engine.grid.shape,
                dtype=np.int32,
            ),

            active_flat=(
                engine.grid.active_flat
            ),

            field_names=np.asarray(
                self.selected_fields
            ),

            particle_names=np.asarray(
                self.selected_particles
            ),
        )


    def _extract_particles(
        self,
        engine,
    ):

        state = (
            engine.particle_state
        )


        if not self.particle_ids:

            return (
                np.empty(
                    0,
                    dtype=np.int32,
                ),

                np.empty(
                    0,
                    dtype=np.uint16,
                ),
            )


        all_voxels = []
        all_species = []


        # ====================================================
        # Primary storage
        # ====================================================

        slot_numbers = cp.arange(
            state.primary_slots,
            dtype=cp.uint32,
        )[None, :]


        valid = (
            slot_numbers
            <
            state.primary_occupancy[
                :,
                None
            ]
        )


        species = (
            state.primary[
                valid
            ]
        )

        voxels = (
            cp.nonzero(
                valid
            )[0]
            .astype(
                cp.int32
            )
        )


        selected = cp.asarray(
            self.particle_ids,
            dtype=cp.uint16,
        )


        keep = cp.isin(
            species,
            selected,
        )


        all_voxels.append(
            voxels[
                keep
            ]
        )

        all_species.append(
            species[
                keep
            ]
        )


        # ====================================================
        # Spill storage
        # ====================================================

        if state.spill_slots > 0:

            slot_numbers = cp.arange(
                state.spill_slots,
                dtype=cp.uint32,
            )[None, :]


            valid = (
                slot_numbers
                <
                state.spill_occupancy[
                    :,
                    None
                ]
            )


            species = (
                state.spill[
                    valid
                ]
            )

            voxels = (
                cp.nonzero(
                    valid
                )[0]
                .astype(
                    cp.int32
                )
            )


            keep = cp.isin(
                species,
                selected,
            )


            all_voxels.append(
                voxels[
                    keep
                ]
            )

            all_species.append(
                species[
                    keep
                ]
            )


        particle_voxels = cp.concatenate(
            all_voxels
        )

        particle_species = cp.concatenate(
            all_species
        )


        return (
            cp.asnumpy(
                particle_voxels
            ),

            cp.asnumpy(
                particle_species
            ),
        )


    def write_analysis(
        self,
        engine,
    ):

        if not self._resolved:

            self._resolve_species(
                engine
            )


        self._write_manifest(
            engine
        )


        self.analysis_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        cp.cuda.Stream.null.synchronize()


        # ====================================================
        # Selected concentration fields
        # ====================================================

        if self.field_ids:

            fields = cp.asnumpy(
                engine.field_state.values[
                    list(
                        self.field_ids
                    )
                ]
            )

        else:

            fields = np.empty(
                (
                    0,
                    engine.grid.cell_voxels,
                ),
                dtype=np.float32,
            )


        # ====================================================
        # Selected discrete particles
        # ====================================================

        (
            particle_voxels,
            particle_species,
        ) = self._extract_particles(
            engine
        )


        tick = (
            engine.scheduler.current_tick
        )


        path = (
            self.analysis_dir
            /
            f"analysis_tick_{tick:012d}.npz"
        )


        np.savez(
            path,

            tick=np.asarray(
                tick,
                dtype=np.int64,
            ),

            biological_time_s=np.asarray(
                engine.scheduler.current_time_s,
                dtype=np.float64,
            ),

            field_names=np.asarray(
                self.selected_fields
            ),

            field_ids=np.asarray(
                self.field_ids,
                dtype=np.int32,
            ),

            field_values=fields,

            particle_names=np.asarray(
                self.selected_particles
            ),

            particle_ids=np.asarray(
                self.particle_ids,
                dtype=np.uint16,
            ),

            particle_voxel_ids=(
                particle_voxels
            ),

            particle_species_ids=(
                particle_species
            ),
        )


        return path