"""FaIR paso a paso: el bucle de ``fair.FAIR.run`` partido en ``prepare`` + ``step(i)``.

Derivado de ``fair/fair.py`` (FaIR v2.2.4, https://github.com/OMS-NetZero/FAIR), Copyright (c) Chris
Smith, Nicholas Leach y colaboradores, licencia Apache-2.0 (ver NOTICE). Cambios respecto del original:

    * el bucle temporal se parte en ``prepare()`` (todo lo que ``run`` hace antes del bucle) y
      ``step(i)`` (el cuerpo del bucle para el punto temporal ``i``), para que las emisiones del año
      ``i`` puedan escribirse DESPUÉS de conocer el estado del año anterior (acoplamiento con el L1);
    * como las emisiones cambian durante la corrida, ``step(i)`` recalcula antes de avanzar las
      emisiones de CO2 (FFI + AFOLU) y las acumuladas del punto ``i + 1`` (en ``run`` se calculan de
      una vez para toda la serie: el resultado es el mismo si las emisiones no cambian);
    * no se escriben las salidas a xarray: los arrays numpy quedan en el objeto.

Con las mismas emisiones, ``step`` reproduce ``FAIR.run`` (prueba de paridad en tests).
"""
from __future__ import annotations

import warnings

import numpy as np

from fair.constants import SPECIES_AXIS
from fair.energy_balance_model import step_temperature
from fair.forcing.aerosol.erfaci import logsum
from fair.forcing.aerosol.erfari import calculate_erfari_forcing
from fair.forcing.ghg import meinshausen2020
from fair.forcing.minor import calculate_linear_forcing
from fair.forcing.ozone import thornhill2021
from fair.gas_cycle import calculate_alpha
from fair.gas_cycle.ch4_lifetime import calculate_alpha_ch4
from fair.gas_cycle.eesc import calculate_eesc
from fair.gas_cycle.forward import step_concentration


class Stepper:
    """Estado numpy de un ``FAIR`` ya configurado (sin ``run``), que avanza un punto temporal por llamada."""

    def __init__(self, f):
        if f.ghg_method != "meinshausen2020" or f.ch4_method != "thornhill2021":
            raise ValueError("Stepper only implements meinshausen2020 + thornhill2021 (the calibrated setup)")
        self.f = f
        self.prepare()
        if f._routine_flags.get("contrails"):
            raise ValueError("contrails are not part of the calibrated ensemble")

    # ------------------------------------------------------------------ antes del bucle
    def prepare(self):
        f = self.f
        f._check_properties()
        f._make_indices()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats._multivariate")
            f._make_ebms()
        self.has_co2_split = (f._co2_indices.sum() + f._co2_ffi_indices.sum() + f._co2_afolu_indices.sum()) == 3
        if self.has_co2_split:
            f.emissions[..., f._co2_indices] = (f.emissions[..., f._co2_ffi_indices].data
                                                + f.emissions[..., f._co2_afolu_indices].data)
        f.cumulative_emissions[1:, ...] = (f.emissions.cumsum(dim="timepoints", skipna=False) * f.timestep
                                           + f.cumulative_emissions[0, ...]).data
        sc = f.species_configs
        self.alpha_lifetime = f.alpha_lifetime.data
        self.airborne = f.airborne_emissions.data
        self.baseline_conc = sc["baseline_concentration"].data
        self.baseline_emis = sc["baseline_emissions"].data
        self.br_atoms = sc["br_atoms"].data
        self.ch4_sens = sc["ch4_lifetime_chemical_sensitivity"].data
        self.lifetime_temp_sens = sc["lifetime_temperature_sensitivity"].data
        self.cl_atoms = sc["cl_atoms"].data
        self.conc = f.concentration.data
        self.conc_per_emis = sc["concentration_per_emission"].data
        self.state = np.full((f._n_timebounds, f._n_scenarios, f._n_configs, f._n_layers + 1), np.nan)
        self.cum = f.cumulative_emissions.data
        self.emis = f.emissions.data
        self.erfari_eff = sc["erfari_radiative_efficiency"].data
        self.aci_scale = sc["aci_scale"].data
        self.aci_shape = sc["aci_shape"].data
        self.forcing = f.forcing.data
        self.forcing_scale = sc["forcing_scale"].data * (1 + sc["tropospheric_adjustment"].data)
        self.efficacy = sc["forcing_efficacy"].data
        self.efficacy_sum = np.full((f._n_timebounds, f._n_scenarios, f._n_configs), np.nan)
        self.forcing_ref_conc = sc["forcing_reference_concentration"].data
        self.forcing_sum = f.forcing_sum.data
        self.forcing_temp_fb = sc["forcing_temperature_feedback"].data
        self.frac_release = sc["fractional_release"].data
        self.g0 = sc["g0"].data
        self.g1 = sc["g1"].data
        self.partitions = f.gas_partitions.data
        self.ghg_eff = sc["greenhouse_gas_radiative_efficiency"].data
        self.h2o_factor = sc["h2o_stratospheric_factor"].data
        self.iirf_0 = sc["iirf_0"].data
        self.iirf_airborne = sc["iirf_airborne"].data
        self.iirf_temperature = sc["iirf_temperature"].data
        self.iirf_uptake = sc["iirf_uptake"].data
        self.lu_cum_to_forcing = sc["land_use_cumulative_emissions_to_forcing"].data
        self.lapsi_eff = sc["lapsi_radiative_efficiency"].data
        self.ozone_eff = sc["ozone_radiative_efficiency"].data
        self.partition_fraction = sc["partition_fraction"].data
        self.unperturbed_lifetime = sc["unperturbed_lifetime"].data
        self.eb_matrix_d = f.ebms["eb_matrix_d"].data
        self.forcing_vector_d = f.ebms["forcing_vector_d"].data
        self.stochastic_d = f.ebms["stochastic_d"].data
        self.forcing_sum[0:1, ...] = np.nansum(self.forcing[0:1, ...], axis=SPECIES_AXIS)
        self.state[0, ..., 0] = self.forcing_sum[0, ...]
        self.state[..., 1:] = f.temperature.data
        if np.sum(np.isnan(self.forcing_ref_conc[:, f._ghg_indices])) > 0:
            raise ValueError("NaN forcing_reference_concentration: cannot compute GHG forcing")
        if not hasattr(f, "ghg_forcing_offset"):
            f.ghg_forcing_offset = meinshausen2020(
                self.baseline_conc[None, None, ...], self.forcing_ref_conc[None, None, ...],
                self.forcing_scale[None, None, ...], self.ghg_eff[None, None, ...],
                f._co2_indices, f._ch4_indices, f._n2o_indices, f._minor_ghg_indices)
        self.i = 0

    # ------------------------------------------------------------------ un punto temporal
    def step(self, i: int | None = None):
        """Avanza el punto temporal ``i`` (por defecto el siguiente): con las emisiones de ``i`` produce
        concentraciones, forzamiento y temperatura en el límite temporal ``i + 1``."""
        f = self.f
        i = self.i if i is None else i
        if self.has_co2_split:
            self.emis[i, ..., f._co2_indices] = (self.emis[i, ..., f._co2_ffi_indices]
                                                 + self.emis[i, ..., f._co2_afolu_indices])
        self.cum[i + 1, ...] = self.cum[i, ...] + self.emis[i, ...] * f.timestep
        st = self.state
        gi = f._ghg_indices
        if f._routine_flags["ghg"]:
            # 1. alpha
            self.alpha_lifetime[i:i + 1, ..., gi] = calculate_alpha(
                self.airborne[i:i + 1, ..., gi], self.cum[i:i + 1, ..., gi],
                self.g0[None, None, ..., gi], self.g1[None, None, ..., gi], self.iirf_0[None, None, ..., gi],
                self.iirf_airborne[None, None, ..., gi], self.iirf_temperature[None, None, ..., gi],
                self.iirf_uptake[None, None, ..., gi], st[i:i + 1, ..., 1:2], f.iirf_max)
            # 2. vida del metano (Thornhill 2021)
            self.alpha_lifetime[i:i + 1, ..., f._ch4_indices] = calculate_alpha_ch4(
                self.emis[i:i + 1, ...], self.conc[i:i + 1, ...], st[i:i + 1, ..., 1:2],
                self.baseline_emis[None, None, ...], self.baseline_conc[None, None, ...],
                self.ch4_sens[None, None, ...], self.lifetime_temp_sens[None, None, :, None],
                f._aerosol_chemistry_from_emissions_indices, f._aerosol_chemistry_from_concentration_indices)
            # 3. emisiones -> concentraciones
            fi = f._ghg_forward_indices
            (self.conc[i + 1:i + 2, ..., fi], self.partitions[..., fi, :],
             self.airborne[i + 1:i + 2, ..., fi]) = step_concentration(
                self.emis[i:i + 1, ..., fi, None], self.partitions[..., fi, :],
                self.airborne[i + 1:i + 2, ..., fi, None], self.alpha_lifetime[i:i + 1, ..., fi, None],
                self.baseline_conc[None, None, ..., fi], self.baseline_emis[None, None, ..., fi, None],
                self.conc_per_emis[None, None, ..., fi], self.unperturbed_lifetime[None, None, ..., fi, :],
                self.partition_fraction[None, None, ..., fi, :], f.timestep)
            if np.any(f._ghg_inverse_indices):
                raise ValueError("concentration-driven GHGs are not supported by the stepper")
            # 5. forzamiento de los GEI (Meinshausen 2020)
            self.forcing[i + 1:i + 2, ..., gi] = meinshausen2020(
                self.conc[i + 1:i + 2, ...],
                self.forcing_ref_conc[None, None, ...] * np.ones((1, f._n_scenarios, f._n_configs, f._n_species)),
                self.forcing_scale[None, None, ...], self.ghg_eff[None, None, ...],
                f._co2_indices, f._ch4_indices, f._n2o_indices, f._minor_ghg_indices)[0:1, ..., gi]
            self.forcing[i + 1:i + 2, ..., gi] = (self.forcing[i + 1:i + 2, ..., gi]
                                                  - f.ghg_forcing_offset[..., gi])
        em, cc = f._aerosol_chemistry_from_emissions_indices, f._aerosol_chemistry_from_concentration_indices
        # 6. aerosoles, interacción con la radiación
        if f._routine_flags["ari"]:
            self.forcing[i + 1:i + 2, ..., f._ari_indices] = calculate_erfari_forcing(
                self.emis[i:i + 1, ...], self.conc[i + 1:i + 2, ...], self.baseline_emis[None, None, ...],
                self.baseline_conc[None, None, ...], self.forcing_scale[None, None, ..., f._ari_indices],
                self.erfari_eff[None, None, ...], em, cc)
        # 7. aerosoles, interacción con las nubes
        if f._routine_flags["aci"]:
            self.forcing[i + 1:i + 2, ..., f._aci_indices] = logsum(
                self.emis[i:i + 1, ...], self.conc[i + 1:i + 2, ...], self.baseline_emis[None, None, ...],
                self.baseline_conc[None, None, ...], self.forcing_scale[None, None, ..., f._aci_indices],
                self.aci_scale[None, None, :], self.aci_shape[None, None, ...], em, cc)
        # 8. EESC
        if f._routine_flags["eesc"]:
            self.conc[i + 1:i + 2, ..., f._eesc_indices] = calculate_eesc(
                self.conc[i + 1:i + 2, ...], self.frac_release[None, None, ...], self.cl_atoms[None, None, ...],
                self.br_atoms[None, None, ...], f._cfc11_indices, f._halogen_indices, f.br_cl_ods_potential)
        # 9. ozono
        if f._routine_flags["ozone"]:
            self.forcing[i + 1:i + 2, ..., f._ozone_indices] = thornhill2021(
                self.emis[i:i + 1, ...], self.conc[i + 1:i + 2, ...], self.baseline_emis[None, None, ...],
                self.baseline_conc[None, None, ...], self.forcing_scale[None, None, ..., f._ozone_indices],
                self.ozone_eff[None, None, ...], em, cc)
        # 11. partículas absorbentes sobre nieve y hielo
        if f._routine_flags["lapsi"]:
            self.forcing[i + 1:i + 2, ..., f._lapsi_indices] = calculate_linear_forcing(
                self.emis[i:i + 1, ...], self.baseline_emis[None, None, ...],
                self.forcing_scale[None, None, ..., f._lapsi_indices], self.lapsi_eff[None, None, ...])
        # 12. vapor de agua estratosférico
        if f._routine_flags["h2o stratospheric"]:
            self.forcing[i + 1:i + 2, ..., f._h2ostrat_indices] = calculate_linear_forcing(
                self.conc[i + 1:i + 2, ...], self.baseline_conc[None, None, ...],
                self.forcing_scale[None, None, ..., f._h2ostrat_indices], self.h2o_factor[None, None, ...])
        # 13. uso del suelo desde las emisiones acumuladas de CO2 AFOLU
        if f._routine_flags["land use"]:
            self.forcing[i + 1:i + 2, ..., f._landuse_indices] = calculate_linear_forcing(
                self.cum[i + 1:i + 2, ...], 0, self.forcing_scale[None, None, ..., f._landuse_indices],
                self.lu_cum_to_forcing[None, None, ...])
        # 14. realimentación temperatura -> forzamiento
        self.forcing[i + 1:i + 2, ...] = (self.forcing[i + 1:i + 2, ...]
                                          + st[i:i + 1, ..., 1:2] * self.forcing_temp_fb[None, None, ...])
        # 15. suma
        self.forcing_sum[i + 1:i + 2, ...] = np.nansum(self.forcing[i + 1:i + 2, ...], axis=SPECIES_AXIS)
        self.efficacy_sum[i + 1:i + 2, ...] = np.nansum(self.forcing[i + 1:i + 2, ...]
                                                        * self.efficacy[None, None, ...], axis=SPECIES_AXIS)
        # 16. forzamiento -> temperatura
        if f._routine_flags["temperature"]:
            st[i + 1:i + 2, ...] = step_temperature(
                st[i:i + 1, ...], self.eb_matrix_d[None, None, ...], self.forcing_vector_d[None, None, ...],
                self.stochastic_d[i + 1:i + 2, None, ...], self.efficacy_sum[i + 1:i + 2, ..., None])
        self.i = i + 1

    def run_to(self, i_end: int):
        """Avanza hasta dejar calculado el límite temporal ``i_end``."""
        while self.i < i_end:
            self.step()

    def temperature(self) -> np.ndarray:
        """Temperatura de superficie (capa 0) en los límites temporales (T+1, escenarios, configs)."""
        return self.state[..., 1]
