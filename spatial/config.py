from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SpatialConfig:
    voxel_size_nm: float
    cell_length_um: float
    cell_diameter_um: float
    max_total_hop_probability: float

    @property
    def voxel_size_um(self) -> float:
        return self.voxel_size_nm / 1000.0

    @property
    def voxel_volume_um3(self) -> float:
        return self.voxel_size_um ** 3

    @property
    def voxel_volume_liters(self) -> float:
        return self.voxel_volume_um3 * 1e-15

    @property
    def diffusion_q(self) -> float:
        return (
            self.max_total_hop_probability
            / 6.0
        )

    def diffusion_timestep_s(
        self,
        diffusion_um2_s: float,
    ) -> float:

        if diffusion_um2_s <= 0:
            raise ValueError(
                "diffusion_um2_s must be > 0"
            )

        return (
            self.diffusion_q
            * self.voxel_size_um ** 2
            / diffusion_um2_s
        )


@dataclass(frozen=True)
class ChromosomeConfig:
    nucleoid_length_fraction: float
    nucleoid_radius_fraction: float
    scaffold_waves: float

    promoter_upstream_bp: int

    dna_occupancy_segment_bp: int
    dna_volume_per_bp_nm3: float


@dataclass(frozen=True)
class CrowdingConfig:
    movement_exponent: float


@dataclass(frozen=True)
class ParticleConfig:
    primary_slots_per_voxel: int
    spill_slots_per_voxel: int

    @property
    def total_slots_per_voxel(self) -> int:
        return (
            self.primary_slots_per_voxel
            + self.spill_slots_per_voxel
        )


@dataclass(frozen=True)
class FieldConfig:
    dtype: str


@dataclass(frozen=True)
class SchedulerConfig:
    reaction_interval_ms: float


@dataclass(frozen=True)
class OutputConfig:
    directory: str

    analysis_interval_s: float
    checkpoint_interval_s: float

    checkpoint_compression: bool

    analysis_fields: tuple[str, ...]
    analysis_particles: tuple[str, ...]


@dataclass(frozen=True)
class BiologyConfig:
    species_file: str
    reactions_file: str

    genome_file: str
    regulation_file: str
    transcription_file: str


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    biological_duration_s: float

    biology: BiologyConfig
    spatial: SpatialConfig
    chromosome: ChromosomeConfig
    crowding: CrowdingConfig
    particles: ParticleConfig
    fields: FieldConfig
    scheduler: SchedulerConfig
    output: OutputConfig


def load_config(path) -> SimulationConfig:

    path = Path(path)

    with path.open("r") as handle:
        raw = yaml.safe_load(handle)

    return SimulationConfig(
        seed=int(
            raw["simulation"]["seed"]
        ),

        biological_duration_s=float(
            raw["simulation"]["biological_duration_s"]
        ),

        biology=BiologyConfig(
            species_file=str(
                raw["biology"]["species_file"]
            ),

            reactions_file=str(
                raw["biology"]["reactions_file"]
            ),

            genome_file=str(
                raw["biology"]["genome_file"]
            ),

            regulation_file=str(
                raw["biology"]["regulation_file"]
            ),

            transcription_file=str(
                raw["biology"]["transcription_file"]
            ),
        ),

        spatial=SpatialConfig(
            voxel_size_nm=float(
                raw["spatial"]["voxel_size_nm"]
            ),

            cell_length_um=float(
                raw["spatial"]["cell_length_um"]
            ),

            cell_diameter_um=float(
                raw["spatial"]["cell_diameter_um"]
            ),

            max_total_hop_probability=float(
                raw["spatial"][
                    "max_total_hop_probability"
                ]
            ),
        ),

        chromosome=ChromosomeConfig(
            nucleoid_length_fraction=float(
                raw["chromosome"][
                    "nucleoid_length_fraction"
                ]
            ),

            nucleoid_radius_fraction=float(
                raw["chromosome"][
                    "nucleoid_radius_fraction"
                ]
            ),

            scaffold_waves=float(
                raw["chromosome"][
                    "scaffold_waves"
                ]
            ),

            promoter_upstream_bp=int(
                raw["chromosome"][
                    "promoter_upstream_bp"
                ]
            ),

            dna_occupancy_segment_bp=int(
                raw["chromosome"][
                    "dna_occupancy_segment_bp"
                ]
            ),

            dna_volume_per_bp_nm3=float(
                raw["chromosome"][
                    "dna_volume_per_bp_nm3"
                ]
            ),
        ),

        crowding=CrowdingConfig(
            movement_exponent=float(
                raw["crowding"][
                    "movement_exponent"
                ]
            )
        ),

        particles=ParticleConfig(
            primary_slots_per_voxel=int(
                raw["particles"][
                    "primary_slots_per_voxel"
                ]
            ),

            spill_slots_per_voxel=int(
                raw["particles"][
                    "spill_slots_per_voxel"
                ]
            ),
        ),

        fields=FieldConfig(
            dtype=str(
                raw["fields"]["dtype"]
            )
        ),

        scheduler=SchedulerConfig(
            reaction_interval_ms=float(
                raw["scheduler"][
                    "reaction_interval_ms"
                ]
            )
        ),

        output=OutputConfig(
            directory=str(
                raw["output"]["directory"]
            ),

            analysis_interval_s=float(
                raw["output"][
                    "analysis_interval_s"
                ]
            ),

            checkpoint_interval_s=float(
                raw["output"][
                    "checkpoint_interval_s"
                ]
            ),

            checkpoint_compression=bool(
                raw["output"][
                    "checkpoint_compression"
                ]
            ),

            analysis_fields=tuple(
                raw["output"].get(
                    "analysis_fields",
                    [],
                )
            ),

            analysis_particles=tuple(
                raw["output"].get(
                    "analysis_particles",
                    [],
                )
            ),
        ),
    )