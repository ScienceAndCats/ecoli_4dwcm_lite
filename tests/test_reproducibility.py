import cupy as cp
import numpy as np

from coupling.engine import SimulationEngine
from simio.checkpoint import CheckpointManager
from spatial.config import load_config


def canonical_particle_state(
    engine,
):
    """
    Convert packed GPU particle storage into a canonical
    biological representation.

    For each voxel, we count how many particles of every
    species are present.

    This intentionally ignores which internal primary/spill
    slot a particle happened to claim on the GPU.
    """

    state = engine.particle_state

    nvox = state.cell_voxels

    nspecies = (
        engine.species.particle_count
        + 1
    )


    counts = np.zeros(
        (
            nvox,
            nspecies,
        ),
        dtype=np.uint32,
    )


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


    for voxel in range(
        nvox
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


            species_counts = np.bincount(
                species_here,
                minlength=nspecies,
            )


            counts[
                voxel
            ] += species_counts.astype(
                np.uint32
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


            species_counts = np.bincount(
                species_here,
                minlength=nspecies,
            )


            counts[
                voxel
            ] += species_counts.astype(
                np.uint32
            )


    return counts


def snapshot_engine(
    engine,
):

    return {
        "tick":
            engine.scheduler.current_tick,

        # Physical particle state:
        # species counts in every voxel.
        "particle_state":
            canonical_particle_state(
                engine
            ),

        "fields":
            cp.asnumpy(
                engine.field_state.values
            ).copy(),

        "reaction_events":
            cp.asnumpy(
                engine.reaction_engine
                .event_counts
            ).copy(),

        "particle_diffusion_steps":
            engine.particle_diffusion
            .step_count,

        "field_diffusion_steps":
            engine.field_diffusion
            .step_count,

        "reaction_steps":
            engine.reaction_engine
            .step_count,

        "spill_writes":
            engine.particle_diffusion
            .spill_write_count,

        "rejected_moves":
            engine.particle_diffusion
            .rejected_move_count,

        "hard_overflow":
            engine.particle_diffusion
            .hard_overflow_count,

        "task_calls":
            {
                task.name:
                    task.calls

                for task
                in engine.scheduler.tasks
            },
    }


def assert_same_physical_state(
    a,
    b,
):

    assert (
        a["tick"]
        == b["tick"]
    )


    # ========================================================
    # Particle state
    #
    # Compare species counts per voxel rather than arbitrary
    # GPU storage-slot ordering.
    # ========================================================

    assert np.array_equal(
        a["particle_state"],
        b["particle_state"],
    )


    # ========================================================
    # Concentration fields
    # ========================================================

    assert np.array_equal(
        a["fields"],
        b["fields"],
    )


    # ========================================================
    # Reaction history
    # ========================================================

    assert np.array_equal(
        a["reaction_events"],
        b["reaction_events"],
    )


    # ========================================================
    # Simulation counters
    # ========================================================

    assert (
        a["particle_diffusion_steps"]
        == b["particle_diffusion_steps"]
    )


    assert (
        a["field_diffusion_steps"]
        == b["field_diffusion_steps"]
    )


    assert (
        a["reaction_steps"]
        == b["reaction_steps"]
    )


    assert (
        a["rejected_moves"]
        == b["rejected_moves"]
    )


    assert (
        a["hard_overflow"]
        == b["hard_overflow"]
    )


    assert (
        a["task_calls"]
        == b["task_calls"]
    )


def test_same_seed_same_physical_state():

    config = load_config(
        "configs/default.yaml"
    )


    engine_a = SimulationEngine(
        config
    )

    engine_b = SimulationEngine(
        config
    )


    engine_a.run(
        biological_seconds=0.02
    )


    engine_b.run(
        biological_seconds=0.02
    )


    cp.cuda.Stream.null.synchronize()


    state_a = snapshot_engine(
        engine_a
    )

    state_b = snapshot_engine(
        engine_b
    )


    assert_same_physical_state(
        state_a,
        state_b,
    )


def test_checkpoint_resume_same_physical_state(
    tmp_path,
):

    config = load_config(
        "configs/default.yaml"
    )


    # ========================================================
    # Continuous reference trajectory
    # ========================================================

    continuous = SimulationEngine(
        config
    )


    continuous.run(
        biological_seconds=0.02
    )


    cp.cuda.Stream.null.synchronize()


    continuous_state = (
        snapshot_engine(
            continuous
        )
    )


    # ========================================================
    # Split trajectory
    # ========================================================

    first_half = SimulationEngine(
        config
    )


    first_half.run(
        biological_seconds=0.01
    )


    cp.cuda.Stream.null.synchronize()


    checkpoint_manager = (
        CheckpointManager(
            output_dir=tmp_path,
            compression=False,
        )
    )


    checkpoint_path = (
        checkpoint_manager.save(
            first_half
        )
    )


    # ========================================================
    # Brand-new engine
    # ========================================================

    resumed = SimulationEngine(
        config
    )


    checkpoint_manager.load(
        resumed,
        checkpoint_path,
    )


    resumed.run(
        biological_seconds=0.01
    )


    cp.cuda.Stream.null.synchronize()


    resumed_state = (
        snapshot_engine(
            resumed
        )
    )


    # ========================================================
    # Compare actual biological/spatial state, not arbitrary
    # packed GPU slot ordering.
    # ========================================================

    assert_same_physical_state(
        continuous_state,
        resumed_state,
    )


    assert resumed.validate()