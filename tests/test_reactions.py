import cupy as cp
import pytest

from coupling.engine import SimulationEngine
from spatial.config import load_config


def test_multifunction_reaction_engine():

    config = load_config(
        "configs/default.yaml"
    )

    engine = SimulationEngine(
        config
    )


    # ========================================================
    # Verify multi-function species definitions
    # ========================================================

    mrna_reactions = (
        engine.reactions
        .reactions_for_species(
            "example_mRNA"
        )
    )

    tf_reactions = (
        engine.reactions
        .reactions_for_species(
            "example_TF"
        )
    )


    assert len(
        mrna_reactions
    ) >= 2

    assert len(
        tf_reactions
    ) >= 2


    # ========================================================
    # Initial state
    # ========================================================

    initial_particles = (
        engine.particle_state
        .total_particle_count
    )


    initial_field_total = float(
        cp.sum(
            engine.field_state.values
        ).item()
    )


    # ========================================================
    # Run 1 biological second of reaction updates.
    #
    # 1000 × 1 ms
    # ========================================================

    reaction_dt = 0.001


    engine.reaction_engine.run_steps(
        steps=1000,
        dt_s=reaction_dt,
    )


    cp.cuda.Stream.null.synchronize()


    # ========================================================
    # Reactions actually occurred
    # ========================================================

    discrete_events = (
        engine.reaction_engine
        .degradation_events

        + engine.reaction_engine
        .conversion_events

        + engine.reaction_engine
        .catalytic_events

        + engine.reaction_engine
        .binding_events

        + engine.reaction_engine
        .unbinding_events
    )


    assert discrete_events > 0


    assert (
        engine.reaction_engine
        .particle_field_flux_evaluations
        > 0
    )


    # ========================================================
    # Particle accounting
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


    assert (
        engine.reaction_engine.validate()
    )


    # ========================================================
    # Particle-field chemistry should transfer ATP -> ADP,
    # not create or destroy total field concentration.
    # ========================================================

    final_field_total = float(
        cp.sum(
            engine.field_state.values
        ).item()
    )


    assert final_field_total == pytest.approx(
        initial_field_total,
        rel=1e-5,
    )
