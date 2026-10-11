from dataclasses import replace

import cupy as cp
import numpy as np

from coupling.engine import SimulationEngine
from spatial.config import load_config


def test_10_second_end_to_end_smoke(
    tmp_path,
):

    config = load_config(
        "configs/default.yaml"
    )


    output_config = replace(
        config.output,

        directory=str(
            tmp_path
        ),

        analysis_interval_s=1.0,

        checkpoint_interval_s=5.0,
    )


    config = replace(
        config,

        biological_duration_s=10.0,

        output=output_config,
    )


    engine = SimulationEngine(
        config
    )


    # ========================================================
    # Starting state
    # ========================================================

    initial_field_total = float(
        cp.sum(
            engine.field_state.values,
            dtype=cp.float64,
        ).item()
    )


    initial_particles = (
        engine.particle_state
        .total_particle_count
    )


    # ========================================================
    # Full coupled 10-second simulation
    # ========================================================

    engine.run(
        biological_seconds=10.0
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # General validity
    # ========================================================

    assert engine.validate()


    assert np.isclose(
        engine.scheduler.current_time_s,
        10.0,
        rtol=0.0,
        atol=engine.base_dt_s,
    )


    # ========================================================
    # All major subsystems actually ran
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
    # No particle-storage failures
    # ========================================================

    assert (
        engine.particle_diffusion
        .hard_overflow_count
        == 0
    )


    # ========================================================
    # Particle reaction accounting
    # ========================================================

    expected_particles = (
        initial_particles

        - engine.reaction_engine
        .degradation_events

        - engine.reaction_engine
        .binding_events

        + engine.reaction_engine
        .unbinding_events
    )


    assert (
        engine.particle_state
        .total_particle_count
        == expected_particles
    )


    # ========================================================
    # Long-run field conservation
    #
    # Concentration fields are float32. Tens of thousands of
    # diffusion steps plus millions of local field updates
    # accumulate small floating-point rounding errors.
    #
    # 5e-5 allows 50 ppm drift while remaining strict enough
    # to catch meaningful mass-loss bugs.
    # ========================================================

    final_field_total = float(
        cp.sum(
            engine.field_state.values,
            dtype=cp.float64,
        ).item()
    )


    assert np.isclose(
        final_field_total,
        initial_field_total,
        rtol=5e-5,
        atol=1e-4,
    )


    # ========================================================
    # Scheduled output
    # ========================================================

    analysis_files = list(
        (
            tmp_path
            / "analysis"
        ).glob(
            "analysis_tick_*.npz"
        )
    )


    checkpoint_paths = list(
        (
            tmp_path
            / "checkpoints"
        ).glob(
            "checkpoint_tick_*"
        )
    )


    assert len(
        analysis_files
    ) == 10


    assert len(
        checkpoint_paths
    ) == 2
