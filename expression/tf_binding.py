from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import numpy as np
import yaml

from expression.promoters import (
    PromoterRegistry,
)
from spatial.particles import (
    ParticleState,
)
from spatial.species import (
    SpeciesRegistry,
)


AVOGADRO = 6.02214076e23


@dataclass(frozen=True)
class TFBindingSite:
    """
    One regulatory binding site associated with one promoter.
    """

    site_id: str

    promoter_locus_tag: str

    promoter_voxel_id: int

    tf_name: str
    tf_particle_id: int

    association_rate_per_tf_s: float
    dissociation_rate_s: float


class TFBindingRegistry:
    """
    Static definitions of transcription-factor binding sites.
    """

    def __init__(
        self,
        sites,
        cell_voxels,
    ):

        self.sites = tuple(
            sites
        )

        self.cell_voxels = int(
            cell_voxels
        )

        self._compile()


    @classmethod
    def from_yaml(
        cls,
        path,
        promoters: PromoterRegistry,
        species_registry: SpeciesRegistry,
        voxel_volume_liters: float,
        cell_voxels: int,
    ):

        path = Path(
            path
        )


        with path.open(
            "r"
        ) as handle:

            raw = yaml.safe_load(
                handle
            )


        sites = []


        for item in raw.get(
            "binding_sites",
            [],
        ):

            site_id = str(
                item["site_id"]
            )


            locus_tag = str(
                item["promoter_locus_tag"]
            )


            promoter = (
                promoters.get_by_locus_tag(
                    locus_tag
                )
            )


            tf_name = str(
                item["tf"]
            )


            tf_species = (
                species_registry.get(
                    tf_name
                )
            )


            if tf_species.particle_id is None:

                raise ValueError(
                    f"TF {tf_name} must be represented "
                    "as a particle species."
                )


            macroscopic_rate = float(
                item[
                    "association_rate_M_inv_s"
                ]
            )


            # =================================================
            # Convert:
            #
            # M^-1 s^-1
            #
            # into:
            #
            # per individual TF molecule per second
            # inside one voxel.
            # =================================================

            association_rate = (
                macroscopic_rate
                /
                (
                    AVOGADRO
                    * voxel_volume_liters
                )
            )


            dissociation_rate = float(
                item[
                    "dissociation_rate_s"
                ]
            )


            if association_rate < 0:

                raise ValueError(
                    "Association rate cannot be negative."
                )


            if dissociation_rate < 0:

                raise ValueError(
                    "Dissociation rate cannot be negative."
                )


            sites.append(
                TFBindingSite(
                    site_id=site_id,

                    promoter_locus_tag=(
                        locus_tag
                    ),

                    promoter_voxel_id=(
                        promoter.voxel_id
                    ),

                    tf_name=tf_name,

                    tf_particle_id=(
                        tf_species.particle_id
                    ),

                    association_rate_per_tf_s=(
                        association_rate
                    ),

                    dissociation_rate_s=(
                        dissociation_rate
                    ),
                )
            )


        return cls(
            sites=sites,
            cell_voxels=cell_voxels,
        )


    def _compile(self):

        nsites = len(
            self.sites
        )


        self.tf_species = np.zeros(
            nsites,
            dtype=np.uint16,
        )


        self.association_rates = np.zeros(
            nsites,
            dtype=np.float32,
        )


        self.dissociation_rates = np.zeros(
            nsites,
            dtype=np.float32,
        )


        sites_by_voxel = [
            []
            for _ in range(
                self.cell_voxels
            )
        ]


        for index, site in enumerate(
            self.sites
        ):

            voxel = (
                site.promoter_voxel_id
            )


            if not (
                0
                <= voxel
                < self.cell_voxels
            ):

                raise ValueError(
                    f"Binding site {site.site_id} "
                    f"has invalid voxel {voxel}."
                )


            self.tf_species[
                index
            ] = site.tf_particle_id


            self.association_rates[
                index
            ] = (
                site.association_rate_per_tf_s
            )


            self.dissociation_rates[
                index
            ] = (
                site.dissociation_rate_s
            )


            sites_by_voxel[
                voxel
            ].append(
                index
            )


        self.voxel_offsets = np.zeros(
            self.cell_voxels + 1,
            dtype=np.int32,
        )


        ordered_sites = []


        for voxel in range(
            self.cell_voxels
        ):

            self.voxel_offsets[
                voxel
            ] = len(
                ordered_sites
            )


            ordered_sites.extend(
                sites_by_voxel[
                    voxel
                ]
            )


        self.voxel_offsets[
            self.cell_voxels
        ] = len(
            ordered_sites
        )


        self.site_indices = np.asarray(
            ordered_sites,
            dtype=np.int32,
        )


    @property
    def site_count(self):

        return len(
            self.sites
        )


    def describe(self):

        print()
        print("TF BINDING REGISTRY")
        print("--------------------------------------------")


        print(
            f"Binding sites:              "
            f"{self.site_count:,}"
        )


        unique_voxels = len(
            {
                site.promoter_voxel_id
                for site
                in self.sites
            }
        )


        print(
            f"Occupied promoter voxels:   "
            f"{unique_voxels:,}"
        )


        for site in self.sites:

            print(
                f"{site.site_id:28s} "
                f"{site.tf_name:16s} "
                f"{site.promoter_locus_tag:10s} "
                f"voxel={site.promoter_voxel_id}"
            )


class TFPromoterBindingEngine:
    """
    GPU engine for reversible TF/promoter binding.

    Free TF:
        stored in ParticleState and can diffuse.

    Bound TF:
        removed from ParticleState and represented by
        bound[site] = 1.

    Unbinding:
        returns the TF particle to the promoter's voxel.
    """

    THREADS = 256


    def __init__(
        self,
        particle_state: ParticleState,
        registry: TFBindingRegistry,
        seed: int,
    ):

        self.particle_state = (
            particle_state
        )

        self.registry = registry

        self.seed = int(
            seed
        )

        self.step_count = 0


        # ====================================================
        # Dynamic promoter occupancy
        # ====================================================

        self.bound = cp.zeros(
            registry.site_count,
            dtype=cp.uint8,
        )


        # [binding events, unbinding events]

        self.event_counts = cp.zeros(
            2,
            dtype=cp.uint64,
        )


        # ====================================================
        # GPU registry arrays
        # ====================================================

        self.voxel_offsets = cp.asarray(
            registry.voxel_offsets
        )

        self.site_indices = cp.asarray(
            registry.site_indices
        )

        self.tf_species = cp.asarray(
            registry.tf_species
        )

        self.association_rates = cp.asarray(
            registry.association_rates
        )

        self.dissociation_rates = cp.asarray(
            registry.dissociation_rates
        )


        # ====================================================
        # Compile kernel
        # ====================================================

        kernel_path = (
            Path(__file__).parents[1]
            / "spatial"
            / "kernels"
            / "promoter_binding.cu"
        )


        source = (
            kernel_path.read_text()
        )


        module = cp.RawModule(
            code=source,

            options=(
                "--std=c++11",
            ),
        )


        self._binding_step = (
            module.get_function(
                "promoter_binding_step"
            )
        )


    @classmethod
    def _blocks(
        cls,
        total,
    ):

        return (
            int(
                (
                    total
                    + cls.THREADS
                    - 1
                )
                // cls.THREADS
            ),
        )


    def step(
        self,
        dt_s: float,
    ):

        if dt_s <= 0:

            raise ValueError(
                "Binding timestep must be > 0."
            )


        state = (
            self.particle_state
        )


        self._binding_step(
            self._blocks(
                state.cell_voxels
            ),

            (
                self.THREADS,
            ),

            (
                state.primary,
                state.primary_occupancy,

                state.spill,
                state.spill_occupancy,

                self.voxel_offsets,
                self.site_indices,

                self.tf_species,
                self.association_rates,
                self.dissociation_rates,

                self.bound,

                self.event_counts,

                np.int32(
                    state.primary_slots
                ),

                np.int32(
                    state.spill_slots
                ),

                np.int64(
                    state.cell_voxels
                ),

                np.float32(
                    dt_s
                ),

                np.uint64(
                    self.seed
                ),

                np.uint64(
                    self.step_count
                ),
            ),
        )


        self.step_count += 1


    def run_steps(
        self,
        steps: int,
        dt_s: float,
    ):

        if steps < 0:

            raise ValueError(
                "steps must be >= 0"
            )


        for _ in range(
            steps
        ):

            self.step(
                dt_s
            )


    @property
    def binding_events(self):

        return int(
            self.event_counts[
                0
            ].item()
        )


    @property
    def unbinding_events(self):

        return int(
            self.event_counts[
                1
            ].item()
        )


    @property
    def bound_count(self):

        return int(
            cp.sum(
                self.bound
            ).item()
        )


    def validate(self):

        self.particle_state.validate()


        if bool(
            cp.any(
                self.bound > 1
            ).item()
        ):

            raise RuntimeError(
                "Invalid promoter binding state."
            )


        return True


    def describe(self):

        print()
        print("TF / PROMOTER BINDING")
        print("--------------------------------------------")


        print(
            f"Binding sites:              "
            f"{self.registry.site_count:,}"
        )


        print(
            f"Currently bound:            "
            f"{self.bound_count:,}"
        )


        print(
            f"Binding events:             "
            f"{self.binding_events:,}"
        )


        print(
            f"Unbinding events:           "
            f"{self.unbinding_events:,}"
        )


        print(
            f"Binding steps:              "
            f"{self.step_count:,}"
        )
