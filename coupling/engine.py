import math

from compute.multirate import MultiRateScheduler

from simio.checkpoint import CheckpointManager
from simio.output import OutputManager

from spatial.config import SimulationConfig
from spatial.field_diffusion import FieldDiffusion
from spatial.fields import FieldState
from spatial.grid import SpatialGrid
from spatial.particle_diffusion import ParticleDiffusion
from spatial.particles import ParticleState
from spatial.reactions import (
    ReactionEngine,
    ReactionRegistry,
)
from spatial.species import SpeciesRegistry


class SimulationEngine:

    def __init__(
        self,
        config: SimulationConfig,
    ):

        self.config = config


        # ====================================================
        # Biological definitions
        # ====================================================

        self.species = SpeciesRegistry.from_yaml(
            config.biology.species_file
        )


        # ====================================================
        # Spatial world
        # ====================================================

        self.grid = SpatialGrid.from_config(
            config.spatial
        )


        # ====================================================
        # Diffusion timing
        # ====================================================

        self.fastest_diffusion_um2_s = (
            self.species.fastest_diffusion_um2_s
        )


        if (
            self.fastest_diffusion_um2_s
            <= 0
        ):

            raise ValueError(
                "At least one diffusing species is required."
            )


        particle_rates = [
            species.diffusion_um2_s
            for species
            in self.species.particle_species
            if species.diffusion_um2_s > 0
        ]


        field_rates = [
            species.diffusion_um2_s
            for species
            in self.species.field_species
            if species.diffusion_um2_s > 0
        ]


        self.fastest_particle_diffusion_um2_s = max(
            particle_rates,
            default=0.0,
        )


        self.fastest_field_diffusion_um2_s = max(
            field_rates,
            default=0.0,
        )


        self.base_dt_s = (
            config.spatial.diffusion_timestep_s(
                self.fastest_diffusion_um2_s
            )
        )


        # ====================================================
        # Particle diffusion interval
        # ====================================================

        self.particle_dt_s = None


        if (
            self.fastest_particle_diffusion_um2_s
            > 0
        ):

            safe_dt = (
                config.spatial.diffusion_timestep_s(
                    self.fastest_particle_diffusion_um2_s
                )
            )


            ticks = max(
                1,
                math.floor(
                    safe_dt
                    / self.base_dt_s
                    + 1e-12
                ),
            )


            self.particle_dt_s = (
                ticks
                * self.base_dt_s
            )


        # ====================================================
        # Field diffusion interval
        # ====================================================

        self.field_dt_s = None


        if (
            self.fastest_field_diffusion_um2_s
            > 0
        ):

            safe_dt = (
                config.spatial.diffusion_timestep_s(
                    self.fastest_field_diffusion_um2_s
                )
            )


            ticks = max(
                1,
                math.floor(
                    safe_dt
                    / self.base_dt_s
                    + 1e-12
                ),
            )


            self.field_dt_s = (
                ticks
                * self.base_dt_s
            )


        # ====================================================
        # Molecular state
        # ====================================================

        self.particle_state = ParticleState(
            grid=self.grid,
            registry=self.species,
            config=config.particles,
            seed=config.seed,
        )


        self.particle_state.validate(
            expected_count=(
                self.particle_state.initial_particle_count
            )
        )


        self.field_state = FieldState(
            grid=self.grid,
            registry=self.species,
            config=config.fields,
        )

        self.field_state.validate()


        # ====================================================
        # Particle diffusion
        # ====================================================

        self.particle_diffusion = None


        if self.particle_dt_s is not None:

            self.particle_diffusion = ParticleDiffusion(
                particle_state=self.particle_state,
                spatial_config=config.spatial,
                base_dt_s=self.particle_dt_s,
                seed=config.seed,
            )


        # ====================================================
        # Field diffusion
        # ====================================================

        self.field_diffusion = None


        if self.field_dt_s is not None:

            self.field_diffusion = FieldDiffusion(
                field_state=self.field_state,
                spatial_config=config.spatial,
                timestep_s=self.field_dt_s,
            )


        # ====================================================
        # Reactions
        # ====================================================

        self.reactions = ReactionRegistry.from_yaml(
            path=config.biology.reactions_file,

            species_registry=self.species,

            voxel_volume_liters=(
                config.spatial.voxel_volume_liters
            ),
        )


        self.reaction_engine = ReactionEngine(
            particle_state=self.particle_state,
            field_state=self.field_state,
            registry=self.reactions,
            seed=config.seed,
        )


        # ====================================================
        # Output managers
        # ====================================================

        self.output_manager = OutputManager(
            output_dir=(
                config.output.directory
            ),

            selected_fields=(
                config.output.analysis_fields
            ),

            selected_particles=(
                config.output.analysis_particles
            ),
        )


        self.checkpoint_manager = CheckpointManager(
            output_dir=(
                config.output.directory
            ),

            compression=(
                config.output.checkpoint_compression
            ),
        )


        # ====================================================
        # Scheduler
        # ====================================================

        self.scheduler = MultiRateScheduler(
            base_dt_s=self.base_dt_s
        )


        self._register_tasks()


    # ========================================================
    # Scheduler registration
    # ========================================================

    def _register_tasks(self):

        self.particle_task = None


        if (
            self.particle_diffusion
            is not None
        ):

            self.particle_task = (
                self.scheduler.add_task(
                    name="particle_diffusion",

                    interval_s=self.particle_dt_s,

                    callback=(
                        self._particle_diffusion_callback
                    ),
                )
            )


        self.field_task = None


        if (
            self.field_diffusion
            is not None
        ):

            self.field_task = (
                self.scheduler.add_task(
                    name="field_diffusion",

                    interval_s=self.field_dt_s,

                    callback=(
                        self._field_diffusion_callback
                    ),
                )
            )


        reaction_dt = (
            self.config.scheduler
            .reaction_interval_ms
            / 1000.0
        )


        self.reaction_task = (
            self.scheduler.add_task(
                name="reactions",

                interval_s=reaction_dt,

                callback=(
                    self._reaction_callback
                ),
            )
        )


        self.analysis_task = (
            self.scheduler.add_task(
                name="analysis_output",

                interval_s=(
                    self.config.output
                    .analysis_interval_s
                ),

                callback=(
                    self._analysis_output_callback
                ),
            )
        )


        self.checkpoint_task = (
            self.scheduler.add_task(
                name="checkpoint",

                interval_s=(
                    self.config.output
                    .checkpoint_interval_s
                ),

                callback=(
                    self._checkpoint_callback
                ),
            )
        )


    # ========================================================
    # Scheduler callbacks
    # ========================================================

    def _particle_diffusion_callback(
        self,
        elapsed_dt,
        tick,
    ):

        self.particle_diffusion.step()


    def _field_diffusion_callback(
        self,
        elapsed_dt,
        tick,
    ):

        self.field_diffusion.step()


    def _reaction_callback(
        self,
        elapsed_dt,
        tick,
    ):

        self.reaction_engine.step(
            dt_s=elapsed_dt
        )


    def _analysis_output_callback(
        self,
        elapsed_dt,
        tick,
    ):

        self.output_manager.write_analysis(
            self
        )


    def _checkpoint_callback(
        self,
        elapsed_dt,
        tick,
    ):

        self.checkpoint_manager.save(
            self
        )


    # ========================================================
    # Run
    # ========================================================

    def run(
        self,
        biological_seconds=None,
    ):

        if biological_seconds is None:

            biological_seconds = (
                self.config
                .biological_duration_s
            )


        return self.scheduler.run(
            biological_seconds
        )


    # ========================================================
    # Manual checkpoint convenience
    # ========================================================

    def save_checkpoint(self):

        return (
            self.checkpoint_manager
            .save(
                self
            )
        )


    def load_checkpoint(
        self,
        path,
    ):

        return (
            self.checkpoint_manager
            .load(
                self,
                path,
            )
        )


    # ========================================================
    # Validation
    # ========================================================

    def validate(self):

        self.particle_state.validate()

        self.field_state.validate()

        self.reaction_engine.validate()


        if (
            self.particle_diffusion
            is not None
        ):

            self.particle_diffusion.validate()


        if (
            self.field_diffusion
            is not None
        ):

            self.field_diffusion.validate()


        return True


    # ========================================================
    # Description
    # ========================================================

    def describe(self):

        spatial = self.config.spatial
        particles = self.config.particles

        print()
        print("SPATIAL CELL ENGINE")
        print("--------------------------------------------")

        print(
            f"Voxel size:                 "
            f"{spatial.voxel_size_nm:.1f} nm"
        )

        print(
            f"Bounding grid:              "
            f"{self.grid.shape}"
        )

        print(
            f"Bounding voxels:            "
            f"{self.grid.bounding_voxels:,}"
        )

        print(
            f"Active cell voxels:         "
            f"{self.grid.cell_voxels:,}"
        )

        print(
            f"Cell volume:                "
            f"{self.grid.cell_volume_um3:.3f} um^3"
        )

        print(
            f"Voxel volume:               "
            f"{spatial.voxel_volume_um3:.6f} um^3"
        )

        print(
            f"Fastest overall D:          "
            f"{self.fastest_diffusion_um2_s:.3f} um^2/s"
        )

        print(
            f"Fastest particle D:         "
            f"{self.fastest_particle_diffusion_um2_s:.3f} um^2/s"
        )

        print(
            f"Fastest field D:            "
            f"{self.fastest_field_diffusion_um2_s:.3f} um^2/s"
        )

        print(
            f"Base timestep:              "
            f"{self.base_dt_s * 1e6:.2f} us"
        )

        print(
            f"Base ticks / bio second:    "
            f"{1.0 / self.base_dt_s:,.0f}"
        )

        print(
            f"Primary particle slots:     "
            f"{particles.primary_slots_per_voxel}"
        )

        print(
            f"Spill particle slots:       "
            f"{particles.spill_slots_per_voxel}"
        )

        print(
            f"Total particle capacity:    "
            f"{particles.total_slots_per_voxel}"
        )


        print()
        print("MULTIRATE SCHEDULE")
        print("--------------------------------------------")


        for task in self.scheduler.tasks:

            print(
                f"{task.name:20s} "
                f"{task.interval_ticks:6d} ticks   "
                f"{task.actual_interval_s*1000:9.4f} ms"
            )


        print(
            f"Current simulation time:    "
            f"{self.scheduler.current_time_s:.6f} s"
        )


        self.species.describe()
        self.reactions.describe()
        self.particle_state.describe()
        self.field_state.describe()


        if (
            self.particle_diffusion
            is not None
        ):

            self.particle_diffusion.describe()


        if (
            self.field_diffusion
            is not None
        ):

            self.field_diffusion.describe()


        self.reaction_engine.describe()