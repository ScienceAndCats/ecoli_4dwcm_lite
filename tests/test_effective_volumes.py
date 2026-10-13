import numpy as np
import pytest

from spatial.effective_volumes import (
    EffectiveVolumeRegistry,
)
from spatial.species import (
    SpeciesRegistry,
)


def build_registry():

    species = SpeciesRegistry.from_yaml(
        "tests/data/"
        "voxel_count_species.yaml"
    )


    volumes = (
        EffectiveVolumeRegistry.from_yaml(
            path=(
                "tests/data/"
                "effective_volumes.yaml"
            ),

            species_registry=species,
        )
    )


    return (
        species,
        volumes,
    )


def test_effective_volume_registry_loads():

    (
        species,
        volumes,
    ) = build_registry()


    assert (
        volumes.species_volume_count
        == 2
    )


    assert (
        volumes.complex_volume_count
        == 2
    )


    assert (
        volumes.species_volume_nm3(
            "example_TF"
        )
        == pytest.approx(
            100.0
        )
    )


    assert (
        volumes.species_volume_nm3(
            "Ribosome"
        )
        == pytest.approx(
            8000.0
        )
    )


    assert (
        volumes.complex_volume_nm3(
            "test_transcription_complex"
        )
        == pytest.approx(
            1000.0
        )
    )


def test_volume_for_multiple_molecules():

    (
        species,
        volumes,
    ) = build_registry()


    assert (
        volumes.volume_for_species_count(
            "Ribosome",
            5,
        )
        == pytest.approx(
            40000.0
        )
    )


def test_particle_volume_lookup_table():

    (
        species,
        volumes,
    ) = build_registry()


    tf = species.get(
        "example_TF"
    )


    # Particle ID zero is the empty-slot sentinel.

    assert (
        volumes.particle_volume_lut_nm3[
            0
        ]
        == pytest.approx(
            0.0
        )
    )


    assert (
        volumes.particle_volume_lut_nm3[
            tf.particle_id
        ]
        == pytest.approx(
            100.0
        )
    )


def test_voxel_count_volume_lookup_table():

    (
        species,
        volumes,
    ) = build_registry()


    ribosome = species.get(
        "Ribosome"
    )


    assert (
        volumes.voxel_count_volume_lut_nm3[
            ribosome.voxel_count_id
        ]
        == pytest.approx(
            8000.0
        )
    )


def test_all_discrete_species_have_test_volumes():

    (
        species,
        volumes,
    ) = build_registry()


    # The test registry contains:
    #
    #   example_TF  -> particle
    #   Ribosome    -> voxel_count
    #   ATP         -> continuous field
    #
    # Only the first two require discrete crowding volumes.

    assert (
        volumes.missing_discrete_species()
        == ()
    )


    assert (
        volumes.validate_complete_discrete()
    )


def test_field_species_cannot_use_discrete_volume():

    (
        species,
        volumes,
    ) = build_registry()


    with pytest.raises(
        ValueError,
        match="continuous",
    ):

        volumes.species_volume_nm3(
            "ATP"
        )


def test_unknown_volume_lookup_fails():

    (
        species,
        volumes,
    ) = build_registry()


    with pytest.raises(
        KeyError,
        match="No effective volume",
    ):

        volumes.complex_volume_nm3(
            "does_not_exist"
        )
