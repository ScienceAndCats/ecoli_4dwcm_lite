import cupy as cp
import numpy as np
import pytest

from coupling.engine import SimulationEngine
from spatial.config import load_config


def particle_species_counts(
    engine,
):

    state = engine.particle_state

    n_species = (
        engine.species.particle_count
        + 1
    )


    primary_counts = cp.bincount(
        state.primary.reshape(-1),
        minlength=n_species,
    )


    spill_counts = cp.bincount(
        state.spill.reshape(-1),
        minlength=n_species,
    )


    return cp.asnumpy(
        primary_counts
        + spill_counts
    )


def test_integrated_conservation():

    config = load_config(
        "configs/default.yaml"
    )

    engine = SimulationEngine(
        config
    )


    # ========================================================
    # Initial particle state
    # ========================================================

    initial_particle_count = (
        engine.particle_state
        .total_particle_count
    )


    initial_species_counts = (
        particle_species_counts(
            engine
        )
    )


    # ========================================================
    # Initial concentration-field totals
    # ========================================================

    initial_field_total = float(
        cp.sum(
            engine.field_state.values,
            dtype=cp.float64,
        ).item()
    )


    atp_id = (
        engine.species
        .get("ATP")
        .field_id
    )

    adp_id = (
        engine.species
        .get("ADP")
        .field_id
    )


    initial_atp_adp = float(
        cp.sum(
            engine.field_state.values[
                [atp_id, adp_id]
            ],
            dtype=cp.float64,
        ).item()
    )


    # ========================================================
    # Run coupled diffusion + reactions
    # ========================================================

    engine.run(
        biological_seconds=0.1
    )

    cp.cuda.Stream.null.synchronize()


    assert engine.validate()


    # ========================================================
    # Particle accounting
    # ========================================================

    degradation = (
        engine.reaction_engine
        .degradation_events
    )

    binding = (
        engine.reaction_engine
        .binding_events
    )

    unbinding = (
        engine.reaction_engine
        .unbinding_events
    )


    expected_particle_count = (
        initial_particle_count
        - degradation
        - binding
        + unbinding
    )


    actual_particle_count = (
        engine.particle_state
        .total_particle_count
    )


    assert (
        actual_particle_count
        == expected_particle_count
    )


    # ========================================================
    # Diffusion must not cause unexplained species changes.
    #
    # We cannot expect every individual species to stay fixed,
    # because reactions convert species.
    #
    # But the reaction engine's total particle accounting must
    # exactly explain the change in total particle number.
    # ========================================================

    final_species_counts = (
        particle_species_counts(
            engine
        )
    )


    assert (
        int(
            np.sum(
                final_species_counts[
                    1:
                ]
            )
        )
        == actual_particle_count
    )


    assert (
        int(
            np.sum(
                initial_species_counts[
                    1:
                ]
            )
        )
        == initial_particle_count
    )


    # ========================================================
    # Entire field system conservation
    #
    # Our current demo field chemistry only transfers
    # concentration between fields.
    # ========================================================

    final_field_total = float(
        cp.sum(
            engine.field_state.values,
            dtype=cp.float64,
        ).item()
    )


    assert final_field_total == pytest.approx(
        initial_field_total,
        rel=1e-6,
        abs=1e-5,
    )


    # ========================================================
    # ATP + ADP specifically should remain conserved.
    #
    # ATP can become ADP, but that transfer should not create
    # or destroy ATP+ADP concentration.
    # ========================================================

    final_atp_adp = float(
        cp.sum(
            engine.field_state.values[
                [atp_id, adp_id]
            ],
            dtype=cp.float64,
        ).item()
    )


    assert final_atp_adp == pytest.approx(
        initial_atp_adp,
        rel=1e-6,
        abs=1e-5,
    )


    # ========================================================
    # Diffusion overflow must remain zero.
    # ========================================================

    assert (
        engine.particle_diffusion
        .hard_overflow_count
        == 0
    )
