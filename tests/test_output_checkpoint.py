import cupy as cp
import numpy as np

from coupling.engine import SimulationEngine
from simio.checkpoint import CheckpointManager
from simio.output import OutputManager
from spatial.config import load_config


def test_output_and_checkpoint_round_trip(
    tmp_path,
):

    config = load_config(
        "configs/default.yaml"
    )

    engine = SimulationEngine(
        config
    )


    # Let the system evolve a little first.

    engine.run(
        biological_seconds=0.01
    )

    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # Lightweight analysis output
    # ========================================================

    output = OutputManager(
        output_dir=tmp_path,

        selected_fields=(
            "ATP",
            "ADP",
        ),

        selected_particles=(
            "DnaA",
            "example_mRNA",
        ),
    )


    analysis_path = (
        output.write_analysis(
            engine
        )
    )


    assert analysis_path.exists()


    with np.load(
        analysis_path
    ) as data:

        assert (
            "field_values"
            in data
        )

        assert (
            "particle_voxel_ids"
            in data
        )

        assert (
            data["field_values"].shape[0]
            == 2
        )


    # ========================================================
    # Save full checkpoint
    # ========================================================

    checkpoints = CheckpointManager(
        output_dir=tmp_path,
        compression=False,
    )


    checkpoint_path = (
        checkpoints.save(
            engine
        )
    )


    assert checkpoint_path.exists()


    saved_tick = (
        engine.scheduler.current_tick
    )


    saved_primary = cp.asnumpy(
        engine.particle_state.primary
    ).copy()


    saved_spill = cp.asnumpy(
        engine.particle_state.spill
    ).copy()


    saved_fields = cp.asnumpy(
        engine.field_state.values
    ).copy()


    saved_reaction_steps = (
        engine.reaction_engine.step_count
    )


    saved_particle_steps = (
        engine.particle_diffusion.step_count
    )


    saved_field_steps = (
        engine.field_diffusion.step_count
    )


    # ========================================================
    # Change the simulation after the checkpoint
    # ========================================================

    engine.run(
        biological_seconds=0.01
    )


    assert (
        engine.scheduler.current_tick
        > saved_tick
    )


    # ========================================================
    # Reload checkpoint
    # ========================================================

    checkpoints.load(
        engine,
        checkpoint_path,
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # Exact state should be back
    # ========================================================

    assert (
        engine.scheduler.current_tick
        == saved_tick
    )


    assert np.array_equal(
        cp.asnumpy(
            engine.particle_state.primary
        ),
        saved_primary,
    )


    assert np.array_equal(
        cp.asnumpy(
            engine.particle_state.spill
        ),
        saved_spill,
    )


    assert np.array_equal(
        cp.asnumpy(
            engine.field_state.values
        ),
        saved_fields,
    )


    assert (
        engine.reaction_engine.step_count
        == saved_reaction_steps
    )


    assert (
        engine.particle_diffusion.step_count
        == saved_particle_steps
    )


    assert (
        engine.field_diffusion.step_count
        == saved_field_steps
    )


    assert engine.validate()
