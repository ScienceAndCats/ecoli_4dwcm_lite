import cupy as cp
import numpy as np
import pytest

from coupling.engine import SimulationEngine
from simio.checkpoint import CheckpointManager
from spatial.config import load_config


def execution_snapshot(
    engine,
):

    return {
        "tick":
            engine.scheduler.current_tick,

        "particle_diffusion_steps":
            engine.particle_diffusion.step_count,

        "field_diffusion_steps":
            engine.field_diffusion.step_count,

        "reaction_steps":
            engine.reaction_engine.step_count,

        "hard_overflow":
            engine.particle_diffusion.hard_overflow_count,

        "task_calls":
            {
                task.name:
                    task.calls

                for task
                in engine.scheduler.tasks
            },
    }


def total_field_amount(
    engine,
):

    return float(
        cp.sum(
            engine.field_state.values,
            dtype=cp.float64,
        ).item()
    )


def exact_checkpoint_state(
    engine,
):

    return {
        "tick":
            engine.scheduler.current_tick,

        "primary":
            cp.asnumpy(
                engine.particle_state.primary
            ).copy(),

        "spill":
            cp.asnumpy(
                engine.particle_state.spill
            ).copy(),

        "primary_occupancy":
            cp.asnumpy(
                engine.particle_state
                .primary_occupancy
            ).copy(),

        "spill_occupancy":
            cp.asnumpy(
                engine.particle_state
                .spill_occupancy
            ).copy(),

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
            engine.particle_diffusion.step_count,

        "field_diffusion_steps":
            engine.field_diffusion.step_count,

        "reaction_steps":
            engine.reaction_engine.step_count,

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


def assert_exact_checkpoint_state(
    observed,
    expected,
):

    assert (
        observed["tick"]
        == expected["tick"]
    )


    assert np.array_equal(
        observed["primary"],
        expected["primary"],
    )


    assert np.array_equal(
        observed["spill"],
        expected["spill"],
    )


    assert np.array_equal(
        observed["primary_occupancy"],
        expected["primary_occupancy"],
    )


    assert np.array_equal(
        observed["spill_occupancy"],
        expected["spill_occupancy"],
    )


    assert np.array_equal(
        observed["fields"],
        expected["fields"],
    )


    assert np.array_equal(
        observed["reaction_events"],
        expected["reaction_events"],
    )


    assert (
        observed["particle_diffusion_steps"]
        == expected["particle_diffusion_steps"]
    )


    assert (
        observed["field_diffusion_steps"]
        == expected["field_diffusion_steps"]
    )


    assert (
        observed["reaction_steps"]
        == expected["reaction_steps"]
    )


    assert (
        observed["spill_writes"]
        == expected["spill_writes"]
    )


    assert (
        observed["rejected_moves"]
        == expected["rejected_moves"]
    )


    assert (
        observed["hard_overflow"]
        == expected["hard_overflow"]
    )


    assert (
        observed["task_calls"]
        == expected["task_calls"]
    )


def test_repeated_runs_follow_same_schedule_and_remain_valid():

    config = load_config(
        "configs/default.yaml"
    )


    engine_a = SimulationEngine(
        config
    )


    engine_b = SimulationEngine(
        config
    )


    initial_field_a = total_field_amount(
        engine_a
    )


    initial_field_b = total_field_amount(
        engine_b
    )


    engine_a.run(
        biological_seconds=0.02
    )


    engine_b.run(
        biological_seconds=0.02
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # CUDA atomics mean that two stochastic GPU runs are not
    # required to produce an identical molecule-by-molecule
    # trajectory, even with the same random seed.
    #
    # What MUST be deterministic is the simulation schedule.
    # ========================================================

    assert (
        execution_snapshot(
            engine_a
        )
        ==
        execution_snapshot(
            engine_b
        )
    )


    # ========================================================
    # Field diffusion + ATP→ADP conversion should conserve the
    # total amount represented by our continuous fields, apart
    # from normal float32 numerical drift.
    # ========================================================

    assert (
        total_field_amount(
            engine_a
        )
        ==
        pytest.approx(
            initial_field_a,
            rel=5e-5,
        )
    )


    assert (
        total_field_amount(
            engine_b
        )
        ==
        pytest.approx(
            initial_field_b,
            rel=5e-5,
        )
    )


    assert (
        engine_a.particle_diffusion
        .hard_overflow_count
        == 0
    )


    assert (
        engine_b.particle_diffusion
        .hard_overflow_count
        == 0
    )


    assert engine_a.validate()

    assert engine_b.validate()


def test_checkpoint_resume_restores_exact_state_and_continues_validly(
    tmp_path,
):

    config = load_config(
        "configs/default.yaml"
    )


    # ========================================================
    # Continuous trajectory used only as an execution-schedule
    # reference.
    # ========================================================

    continuous = SimulationEngine(
        config
    )


    continuous.run(
        biological_seconds=0.02
    )


    cp.cuda.Stream.null.synchronize()


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


    state_at_checkpoint = (
        exact_checkpoint_state(
            first_half
        )
    )


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
    # Load into a brand-new engine.
    # ========================================================

    resumed = SimulationEngine(
        config
    )


    checkpoint_manager.load(
        resumed,
        checkpoint_path,
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # At the instant of loading, checkpoint restoration SHOULD
    # be exact.
    # ========================================================

    restored_state = (
        exact_checkpoint_state(
            resumed
        )
    )


    assert_exact_checkpoint_state(
        restored_state,
        state_at_checkpoint,
    )


    # ========================================================
    # Continue for the remaining biological time.
    # ========================================================

    resumed.run(
        biological_seconds=0.01
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # Exact future stochastic replay is deliberately NOT
    # required. GPU atomic ordering can produce a different
    # valid trajectory.
    #
    # The execution schedule must nevertheless be identical.
    # ========================================================

    assert (
        execution_snapshot(
            resumed
        )
        ==
        execution_snapshot(
            continuous
        )
    )


    # Continuous-field total is a conserved quantity and
    # therefore should agree within numerical tolerance.

    assert (
        total_field_amount(
            resumed
        )
        ==
        pytest.approx(
            total_field_amount(
                continuous
            ),
            rel=5e-5,
        )
    )


    assert (
        resumed.particle_diffusion
        .hard_overflow_count
        == 0
    )


    assert resumed.validate()

    assert continuous.validate()