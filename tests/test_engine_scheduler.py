from coupling.engine import SimulationEngine
from spatial.config import load_config


def test_engine_multirate_scheduler():

    config = load_config(
        "configs/default.yaml"
    )

    engine = SimulationEngine(
        config
    )


    # ========================================================
    # Confirm tasks were registered
    # ========================================================

    task_names = {
        task.name
        for task
        in engine.scheduler.tasks
    }


    assert (
        "particle_diffusion"
        in task_names
    )

    assert (
        "field_diffusion"
        in task_names
    )

    assert (
        "reactions"
        in task_names
    )


    # ========================================================
    # Run a short integrated simulation
    # ========================================================

    engine.run(
        biological_seconds=0.01
    )


    # ========================================================
    # Every subsystem should have run
    # ========================================================

    assert (
        engine.particle_diffusion
        .step_count
        > 0
    )

    assert (
        engine.field_diffusion
        .step_count
        > 0
    )

    assert (
        engine.reaction_engine
        .step_count
        > 0
    )


    # ========================================================
    # Controller counts should agree with scheduler counts
    # ========================================================

    assert (
        engine.particle_diffusion
        .step_count
        ==
        engine.particle_task.calls
    )

    assert (
        engine.field_diffusion
        .step_count
        ==
        engine.field_task.calls
    )

    assert (
        engine.reaction_engine
        .step_count
        ==
        engine.reaction_task.calls
    )


    # ========================================================
    # Clock should have advanced
    # ========================================================

    assert (
        engine.scheduler
        .current_time_s
        > 0
    )


    # ========================================================
    # Full integrated state remains valid
    # ========================================================

    assert engine.validate()
