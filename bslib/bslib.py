"""This module contains functions to simulate battery storage systems"""
import os
from enum import Enum, auto
from math import sqrt
from typing import NamedTuple, Optional
import pandas as pd  # type: ignore
import numpy as np


class BatteryState(Enum):
    CHARGING = auto()
    DISCHARGING = auto()


def load_parameters(system_id: str) -> dict:
    """Loads model specific parameters from the database.

    :param system_id: ID of the Model in the database, defaults to None
    :type system_id: str, optional
    :return: Parameter of the specified model
    :rtype: dict
    """

    database = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                        "bslib_database.csv")))

    if system_id in database.ID.values:
        database = database.loc[database['ID'] == system_id]
    else:
        raise ValueError(f"Parameters for model {system_id} could not be found in database.")

    database.columns = database.columns.str.replace(r"\[W]", "", regex=True)
    database.columns = database.columns.str.replace(r"\[V]", "", regex=True)
    database.columns = database.columns.str.replace(r"\[s]", "", regex=True)
    database.columns = database.columns.str.replace(r"\[kWh]", "", regex=True)
    database.columns = database.columns.str.replace(r"\[-coupled]", "", regex=True)
    # Remove trailing and leading whitespaces
    database.columns = database.columns.str.strip()

    return database.to_dict(orient='records')[0]


def load_database():
    """This function returns the entire database as a pandas DataFrame."""
    return pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..",
                                                    "src",
                                                    "bslib_database.csv")))


class ACBatMod:
    def __init__(self, system_id: str, *, p_inv_custom: float = None, e_bat_custom: float = None) -> None:
        r"""Performance simulation class for AC-coupled battery systems

        :param system_id: Parameters of the battery system
        :type system_id: dict
        :param p_inv_custom: Custom values for the PV inverter when using a generic system
        :type p_inv_custom: float
        :param e_bat_custom: Custom value for the battery capacity when using a generic system
        :type e_bat_custom: float
        """
        # Load system parameter according to the given system id
        self.__parameter: dict = load_parameters(system_id)

        if self.__parameter['Manufacturer (PE)'] == 'Generic':
            if p_inv_custom is None:
                raise TypeError('Custom value for the inverter power is not specified.')
            if e_bat_custom is None:
                raise TypeError('Custom value for the battery capacity is not specified.')

        # System sizing
        self._E_BAT: float = self.__parameter['E_BAT'] * 1000  # Mean usable battery capacity in Wh
        self._P_AC2BAT_IN = self.__parameter['P_AC2BAT_in']  # Nominal AC charging power in kW
        self._P_BAT2AC_OUT = self.__parameter['P_BAT2AC_out']  # Nominal AC discharging power in kW

        # Conversion losses
        self._ETA_BAT = self.__parameter['eta_BAT'] / 100  # Mean battery efficiency from 0 to 1 representing 0 to 100%
        # Polynomial curve fitting parameters of the power loss functions in W
        self._AC2BAT_A_IN = self.__parameter['AC2BAT_a_in']
        self._AC2BAT_B_IN = self.__parameter['AC2BAT_b_in']
        self._AC2BAT_C_IN = self.__parameter['AC2BAT_c_in']
        self._BAT2AC_A_OUT = self.__parameter['BAT2AC_a_out']
        self._BAT2AC_B_OUT = self.__parameter['BAT2AC_b_out']
        self._BAT2AC_C_OUT = self.__parameter['BAT2AC_c_out']

        # Control losses
        # Threshold to adjust the hysteresis to avoid
        # alternation between charging and discharging
        self._SOC_THRESHOLD = self.__parameter['SOC_h']  # Hysteresis threshold for the recharging of the battery
        self._threshold = False
        self._P_AC2BAT_DEV = self.__parameter['P_AC2BAT_DEV']  # Mean stationary deviation of the charging power in W
        self._P_BAT2AC_DEV = self.__parameter['P_BAT2AC_DEV']  # Mean stationary deviation of the discharging power in W

        # Standby losses
        self._P_SYS_SOC0_DC = self.__parameter['P_SYS_SOC0_DC']  # Standby DC power consumption in discharged state in W
        self._P_SYS_SOC0_AC = self.__parameter['P_SYS_SOC0_AC']  # Standby AC power consumption in discharged state in W
        self._P_SYS_SOC1_DC = self.__parameter['P_SYS_SOC1_DC']  # Standby DC power consumption in charged state in W
        self._P_SYS_SOC1_AC = self.__parameter['P_SYS_SOC1_AC']  # Standby AC power consumption in charged state in W
        self._P_PERI_AC = self.__parameter['P_PERI_AC']  # AC power consumption of other system components in W

        if self.__parameter['Manufacturer (PE)'] == 'Generic':
            self._P_AC2BAT_IN = p_inv_custom  # Custom inverter power in kW
            self._P_BAT2AC_OUT = p_inv_custom  # Custom inverter power in kW
            # Custom battery capacity Wh
            self._E_BAT = e_bat_custom * 1000  # type: ignore
            self._P_SYS_SOC1_DC = self._P_SYS_SOC1_DC * self._E_BAT
            self._P_SYS_SOC1_AC = self._P_SYS_SOC1_AC * self._P_BAT2AC_OUT

        # Minimum AC charging power
        self._P_AC2BAT_MIN = self._AC2BAT_C_IN
        # Minimum AC discharging power
        self._P_BAT2AC_MIN = self._BAT2AC_C_OUT

        # Correction factor to avoid overcharging of the battery
        self._CORR_FACTOR = 0.1

        # Stores the AC/DC power and the state of charge of the battery system
        self._Results = NamedTuple("Results", [("p_bs", float), ("p_bat", float), ("soc", float)])  # type: ignore

        # Stores the state of the battery
        self.state = BatteryState

    def avoid_overcharging(self, p_bs: float, e_b0: float, dt: int) -> float:
        """This method avoids overcharging the battery in both ways."""
        # Estimated amount of energy in Wh that is supplied to or discharged from the storage unit.
        e_bs_est = p_bs * dt / 3600
        # Reduce the power on the AC side to avoid over charging of the battery
        if e_bs_est > 0 and e_bs_est > (self._E_BAT - e_b0):
            p_bs = (self._E_BAT - e_b0) * 3600 / dt
        # When discharging take the correction factor into account
        elif e_bs_est < 0 and abs(e_bs_est) > e_b0:
            p_bs = -((e_b0 * 3600 / dt) * (1 - self._CORR_FACTOR))

        return p_bs

    def adjust_to_stationary_deviations(self, p_bs: float) -> float:
        """This method adjusts the AC power of the battery to the stationary deviations."""
        if p_bs > self._P_AC2BAT_MIN:
            return max(self._P_AC2BAT_MIN, p_bs + self._P_AC2BAT_DEV)

        elif p_bs < -self._P_BAT2AC_MIN:
            return min(-self._P_BAT2AC_MIN, p_bs - self._P_BAT2AC_DEV)

        else:
            return 0.0

    def get_battery_state(self, p_bs: float, soc: float) -> Optional[BatteryState]:
        """This method determines the condition of the battery."""
        if p_bs > 0 and soc < 1 - self._threshold * (1 - self._SOC_THRESHOLD):
            # The last term th*(1-SOC_h) avoids the alternation between
            # charging and standby mode due to the DC power consumption of the
            # battery converter when the battery is fully charged. The battery
            # will not be recharged until the SOC falls below the SOC-threshold
            # (SOC_h) for recharging from PV.
            return self.state.CHARGING
        elif p_bs < 0 and soc > 0:
            return self.state.DISCHARGING
        else:
            return None

    def charge_battery(self, p_bs: float) -> float:
        """This method determines with how much energy the battery will be charged."""
        # Normalized AC power of the battery system
        p_bs_norm = p_bs / self._P_AC2BAT_IN / 1000

        # DC power of the battery affected by the AC2BAT conversion losses
        # of the battery converter in W
        return max(0, p_bs - (self._AC2BAT_A_IN * p_bs_norm * p_bs_norm
                              + self._AC2BAT_B_IN * p_bs_norm
                              + self._AC2BAT_C_IN))

    def discharge_battery(self, p_bs: float) -> float:
        """This method determines with how much energy the battery will be discharged."""
        # Normalized AC power of the battery system
        p_bs_norm = abs(p_bs / self._P_BAT2AC_OUT / 1000)

        # DC power of the battery affected by the BAT2AC conversion losses
        # of the battery converter in W
        return p_bs - (self._BAT2AC_A_OUT * p_bs_norm * p_bs_norm
                       + self._BAT2AC_B_OUT * p_bs_norm
                       + self._BAT2AC_C_OUT)

    def standby_mode(self, soc: float) -> tuple[float, float]:
        """This method determines which standby mode the battery system is in."""
        if soc <= 0:  # Standby mode in discharged state
            # AC and DC power consumption of the battery converter in W
            p_bat = -max(0, self._P_SYS_SOC0_DC)
            p_bs = self._P_SYS_SOC0_AC

        else:  # Standby mode in fully charged state
            # AC and DC consumption of the battery converter in W
            p_bat = -max(0, self._P_SYS_SOC1_DC)
            p_bs = self._P_SYS_SOC1_AC

        return p_bs, p_bat

    def change_battery_content(self, p_bat: float, e_b0: float, dt: int) -> float:
        """This method determines the energy content of the battery."""
        # Change the energy content of the battery and conversion from Ws to Wh
        if p_bat > 0:  # Charging
            return e_b0 + p_bat * sqrt(self._ETA_BAT) * dt / 3600

        elif p_bat < 0:  # Discharging
            return e_b0 + p_bat / sqrt(self._ETA_BAT) * dt / 3600

        else:
            return e_b0

    def simulate(self, *, p_load: float, soc: float, dt: int) -> NamedTuple:
        r"""Simulation function for AC-coupled battery systems.

        :param p_load: Set point for the power on the AC side of the battery system in W.
                      A positive value represents an excess power.
                      A negative value represents a power demand.
        :type p_load: float

        :param soc: State of charge of the battery. Range 0 to 1 represents 0 to 100%.
        :type soc: float

        :param dt: Time increment in seconds.
        :type dt: integer

        :return: Power on the AC side (py:attribute::p_bs) and DC side (py:attribute::p_bat) of the battery system.
                 As well as the state of charge (py:attribute::soc) of the battery.
        :rtype: namedtuple[p_bs, p_bat, soc]

        """

        # AC power of the battery with respect to additional power consumption
        # of other peripheral system components in W
        p_bs = p_load - self._P_PERI_AC

        # Energy content of the battery before calculations in Ws
        e_b0 = soc * self._E_BAT

        # Check if the battery holds enough unused capacity for charging or discharging
        p_bs = self.avoid_overcharging(p_bs, e_b0, dt)

        # Adjust the AC power of the battery system due to the stationary
        # deviations, taking the minimum charging and discharging power into
        # account
        p_bs = self.adjust_to_stationary_deviations(p_bs)

        # Limit the AC power of the battery system to the rated power of the
        # battery converter
        p_bs = max(-self._P_BAT2AC_OUT * 1000,
                   min(self._P_AC2BAT_IN * 1000, p_bs))

        battery_state = self.get_battery_state(p_bs, soc)

        if battery_state == self.state.CHARGING:
            p_bat = self.charge_battery(p_bs)

        elif battery_state == self.state.DISCHARGING:
            p_bat = self.discharge_battery(p_bs)

        else:  # Neither charging nor discharging of the battery
            p_bat = 0.0

        # Check if the battery system is in standby mode
        if p_bat == 0:
            p_bs, p_bat = self.standby_mode(soc)

        # Change the energy content of the battery and conversion from Ws to Wh
        e_b = self.change_battery_content(p_bat, e_b0, dt)

        # Calculate the resulting state of charge of the battery
        soc = e_b / self._E_BAT

        # Adjust the hysteresis threshold to avoid alternation
        # between charging and standby mode due to the DC power
        # consumption of the battery converter.
        self._threshold = bool(self._threshold and soc > self._SOC_THRESHOLD or soc > 1)

        # Outputs
        return self._Results(p_bs=p_bs, p_bat=p_bat, soc=soc)


class DCBatMod:
    """Performance Simulation Class for DC-coupled PV-Battery systems

    :param parameter: PV battery system parameters
    :type parameter: dict

    :param d: array containing parameters
    :type d: numpy array

    :param ppv: normalized DC power output of the PV generator
    :type ppv: numpy array

    :param pl: AC load power
    :type pl: numpy array

    :param Pr: Residual power for battery charging
    :type Pr: numpy array

    :param Prpv: AC residual power
    :type Pr: numpy array

    :param Ppv: DC power output of the PV generator
    :type Ppv: numpy array

    :param ppv2ac: Normalized AC output power of the PV2AC conversion pathway to cover the AC power demand
    :type ppv2ac: numpy array

    :param Ppv2ac_out: Target AC output power of the PV2AC conversion pathway
    :type Ppv2ac_out: numpy array

    :param dt: time step width in seconds
    :type dt: integer
    """

    def __init__(self, system_id: str, *, p_inv_custom=None, e_bat_custom=None) -> None:
        """Constructor method
        """
        # Load system parameter according to the given system id
        self.__parameter: dict = load_parameters(system_id)

        # System sizing
        self._E_BAT = self.__parameter["E_BAT"] * 1000  # Mean battery capacity in Wh
        self._P_PV2AC_IN = self.__parameter["P_PV2AC_in"]  # Rated PV input power (DC) in kW
        self._P_PV2AC_OUT = self.__parameter["P_PV2AC_out"]  # Rated PV output power (AC) in kW
        self._P_PV2BAT_IN = self.__parameter["P_PV2BAT_in"]  # Nominal input charging power (DC) in kW
        self._P_BAT2AC_OUT = self.__parameter["P_BAT2AC_out"]  # Nominal discharging power (AC) in kW

        # Conversion losses
        self._eta_BAT = self.__parameter["eta_BAT"] / 100  # Mean battery efficiency in %
        # Polynomial curve fitting parameters of the power loss functions in W
        self._PV2AC_A_IN = self.__parameter["PV2AC_a_in"]
        self._PV2AC_B_IN = self.__parameter["PV2AC_b_in"]
        self._PV2AC_C_IN = self.__parameter["PV2AC_c_in"]
        self._PV2AC_A_OUT = self.__parameter["PV2AC_a_out"]
        self._PV2AC_B_OUT = self.__parameter["PV2AC_b_out"]
        self._PV2AC_C_OUT = self.__parameter["PV2AC_c_out"]
        self._PV2BAT_A_IN = self.__parameter["PV2BAT_a_in"]
        self._PV2BAT_B_IN = self.__parameter["PV2BAT_b_in"]
        self._BAT2AC_A_OUT = self.__parameter["BAT2AC_a_out"]
        self._BAT2AC_B_OUT = self.__parameter["BAT2AC_b_out"]
        self._BAT2AC_C_OUT = self.__parameter["BAT2AC_c_out"]
        self._P_PV2AC_MIN = self._PV2AC_C_IN  # Minimum input power of the PV2AC conversion pathway

        # Control losses
        # Threshold to adjust the hysteresis to avoid alternation between charging and discharging
        self._SOC_THRESHOLD = self.__parameter["SOC_h"]  # Hysteresis threshold for the recharging of the battery
        self._threshold = False
        self._P_PV2BAT_DEV = self.__parameter["P_PV2BAT_DEV"]  # Mean stationary deviation of the charging power in W
        self._P_BAT2AC_DEV = self.__parameter["P_BAT2AC_DEV"]  # Mean stationary deviation of the discharging power in W

        # Standby losses
        self._P_SYS_SOC1_AC = self.__parameter["P_SYS_SOC1_AC"]  # AC standby power consumption in charged state in W
        self._P_SYS_SOC0_AC = self.__parameter["P_SYS_SOC0_AC"]  # DC standby power consumption in discharged state in W
        self._P_SYS_SOC1_DC = self.__parameter["P_SYS_SOC1_DC"]  # DC standby power consumption in charged state in W
        self._P_SYS_SOC0_DC = self.__parameter["P_SYS_SOC0_DC"]  # AC standby power consumption in discharged state in W
        self._P_PERI_AC = self.__parameter["P_PERI_AC"]  # AC power consumption of other system components in W

        if self.__parameter['Manufacturer (PE)'] == 'Generic':
            self._P_PV2AC_IN = p_inv_custom  # in kW
            self._P_PV2AC_OUT = p_inv_custom  # in kW
            self._P_PV2BAT_IN = p_inv_custom  # in kW
            self._P_BAT2AC_OUT = p_inv_custom  # in kW
            self._E_BAT = e_bat_custom * 1000  # Custom battery capacity in Wh
            self._P_SYS_SOC1_DC = self._P_SYS_SOC1_DC * self._E_BAT
            self._P_SYS_SOC1_AC = self._P_SYS_SOC1_AC * self._P_BAT2AC_OUT

        # Correction factor to avoid overcharge and discharge the battery
        self._CORR_FACTOR = 0.1

        self._state = BatteryState

        # Stores the AC power and the state of charge of the battery system
        self._Results = NamedTuple("Results", [("p_pvbs", float), ("p_bat", float), ("soc", float)])  # type: ignore

    def __get_residual_power(self, p_load: float, p_pv: float) -> tuple[float, float, float, float]:
        """This method determines the residual power."""
        # Output of the PV generator limited to the maximum DC input power of the PV2AC conversion pathway
        p_pv_limited = min(p_pv, self._P_PV2AC_IN * 1000)

        # Power demand on the AC side of the battery system respecting the power consumption of other system components
        p_ac = p_load + self._P_PERI_AC

        # Normalized AC output power of the PV2AC conversion pathway to cover the AC power demand
        ppv2ac_ac_out_norm = min(p_ac, self._P_PV2AC_OUT * 1000) / self._P_PV2AC_OUT / 1000

        # Target DC input power of the PV2AC conversion pathway
        p_pv2ac_dc_in = min(p_ac, self._P_PV2AC_OUT * 1000) + (self._PV2AC_A_OUT * ppv2ac_ac_out_norm * ppv2ac_ac_out_norm
                                                               + self._PV2AC_B_OUT * ppv2ac_ac_out_norm
                                                               + self._PV2AC_C_OUT)

        # Normalized DC input power of the PV2AC conversion pathway
        ppv2ac_dc_in_norm = p_pv_limited / self._P_PV2AC_IN / 1000

        # Target AC output power of the PV2AC conversion pathway
        p_pv2ac_ac_out = max(0, p_pv_limited - (self._PV2AC_A_IN * ppv2ac_dc_in_norm * ppv2ac_dc_in_norm
                                                + self._PV2AC_B_IN * ppv2ac_dc_in_norm
                                                + self._PV2AC_C_IN))

        # Residual power for battery charging
        p_rpv = p_pv_limited - p_pv2ac_dc_in

        # Residual power for battery discharging
        p_r = p_pv2ac_ac_out - p_ac

        return p_r, p_rpv, p_pv_limited, p_pv2ac_ac_out

    def __avoid_overcharging(self, p_r: float, p_rpv: float, e_b0: float, dt: int) -> tuple[float, float]:
        """This method avoids overcharging the battery in both ways."""
        # Check if the battery holds enough unused capacity for charging or discharging
        # Estimated amount of energy that is supplied to or discharged from the storage unit.
        e_bs_rpv = p_rpv * dt / 3600
        e_bs_r = p_r * dt / 3600

        # Reduce P_bs to avoid over charging of the battery
        if e_bs_rpv > 0 and e_bs_rpv > (self._E_BAT - e_b0):
            p_rpv = (self._E_BAT - e_b0) * 3600 / dt
        # When discharging take the correction factor into account
        elif e_bs_r < 0 and np.abs(e_bs_r) > e_b0:
            p_r = (e_b0 * 3600 / dt) * (1 - self._CORR_FACTOR)

        return p_r, p_rpv

    def __get_battery_state(self, p_rpv: float, soc: float) -> Optional[BatteryState]:
        """This method determines the condition of the battery."""
        if p_rpv > 0 and soc < 1 - self._threshold * (1 - self._SOC_THRESHOLD):
            '''The last term th*(1-SOC_h) avoids the alternation between
            charging and standby mode due to the DC power consumption of the
            battery converter when the battery is fully charged. The battery
            will not be recharged until the SOC falls below the SOC-threshold
            (SOC_h) for recharging from PV.'''
            return self._state.CHARGING
        elif p_rpv < 0 and soc > 0:
            return self._state.DISCHARGING
        else:
            return None

    def __charge_battery(self, p_rpv: float, p_pv_limited: float) -> tuple[float, float]:
        """This method determines with how much energy the battery will be charged."""
        # Charging power
        p_pv2bat_in = p_rpv

        # Adjust the charging power due to the stationary deviations
        p_pv2bat_in = max(0, p_pv2bat_in + self._P_PV2BAT_DEV)

        # Limit the charging power to the maximum charging power
        p_pv2bat_in = min(p_pv2bat_in, self._P_PV2BAT_IN * 1000)

        # Limit the charging power to the current power output of the PV generator
        p_pv2bat_in = min(p_pv2bat_in, p_pv_limited)

        # Normalized charging power
        p_pv2bat_norm = p_pv2bat_in / self._P_PV2BAT_IN / 1000

        # DC power of the battery affected by the PV2BAT conversion losses
        # (the idle losses of the PV2BAT conversion pathway are not taken
        # into account)
        p_bat = max(0, p_pv2bat_in - (self._PV2BAT_A_IN * p_pv2bat_norm * p_pv2bat_norm
                                      + self._PV2BAT_B_IN * p_pv2bat_norm))

        # Realized DC input power of the PV2AC conversion pathway
        p_pv2ac_dc_in = p_pv_limited - p_pv2bat_in

        # Normalized DC input power of the PV2AC conversion pathway
        p_pv2ac_dc_in_norm = p_pv2ac_dc_in / self._P_PV2AC_IN / 1000

        # Realized AC power of the PV-battery system
        p_pvbs = max(0, p_pv2ac_dc_in - (self._PV2AC_A_IN * p_pv2ac_dc_in_norm * p_pv2ac_dc_in_norm
                                         + self._PV2AC_B_IN * p_pv2ac_dc_in_norm
                                         + self._PV2AC_C_IN))
        return p_pvbs, p_bat

    def __discharge_battery(self, p_r: float, p_pv_limited: float, p_pv2ac_ac_out: float) -> tuple[float, float]:
        """This method determines with how much energy the battery will be charged."""
        # Discharging power
        p_bat2ac_out = p_r * - 1

        # Adjust the discharging power due to the stationary deviations
        p_bat2ac_out = max(0, p_bat2ac_out + self._P_BAT2AC_DEV)

        # Adjust the discharging power to the maximum discharging power
        p_bat2ac_out = min(p_bat2ac_out, self._P_BAT2AC_OUT * 1000)

        # Limit the discharging power to the maximum AC power output of the PV-battery system
        p_bat2ac_out = min(self._P_PV2AC_OUT * 1000 - p_pv2ac_ac_out, p_bat2ac_out)

        # Normalized discharging power
        p_pv2bat_norm = p_bat2ac_out / self._P_BAT2AC_OUT / 1000

        # DC power of the battery affected by the BAT2AC conversion losses
        # (if the idle losses of the PV2AC conversion pathway are covered by
        # the PV generator, the idle losses of the BAT2AC conversion pathway
        # are not taken into account)
        if p_pv_limited > self._P_PV2AC_MIN:
            p_bat = -1 * (p_bat2ac_out + (self._BAT2AC_A_OUT * p_pv2bat_norm * p_pv2bat_norm
                                          + self._BAT2AC_B_OUT * p_pv2bat_norm))
        else:
            p_bat = -1 * (p_bat2ac_out + (self._BAT2AC_A_OUT * p_pv2bat_norm * p_pv2bat_norm
                                          + self._BAT2AC_B_OUT * p_pv2bat_norm
                                          + self._BAT2AC_C_OUT)) + p_pv_limited

        # Realized AC power of the PV-battery system
        p_pvbs = p_pv2ac_ac_out + p_bat2ac_out

        return p_pvbs, p_bat

    def __standby_mode(self, p_pvbs: float, p_bat: float, soc: float) -> tuple[float, float]:
        """This method determines which standby mode the battery system is in."""
        if p_pvbs == 0 and soc <= 0:  # Standby mode in discharged state
            # DC and AC power consumption of the PV-battery inverter
            p_bat = -1 * max(0, self._P_SYS_SOC0_DC)
            p_pvbs = -1 * self._P_SYS_SOC0_AC

        elif p_pvbs > 0 and soc > 0:  # Standby mode in fully charged state
            # DC power consumption of the PV-battery inverter
            p_bat = -1 * max(0, self._P_SYS_SOC1_DC)

        return p_pvbs, p_bat

    def __change_energy_content(self, p_bat: float, e_b0: float, dt: int) -> float:
        """This method determines the energy content of the battery."""
        # Change the energy content of the battery Wx to Wh conversion
        if p_bat > 0:
            return e_b0 + p_bat * sqrt(self._eta_BAT) * dt / 3600
        elif p_bat < 0:
            return e_b0 + p_bat / sqrt(self._eta_BAT) * dt / 3600
        else:
            return e_b0

    def simulation(self, *, p_load: float, p_pv: float, soc: float, dt: int) -> NamedTuple:
        """Performance simulation function for DC-coupled battery systems

        :param p_load: AC set point for the AC power on the AC side of the battery system
        :type p_load: float
        :param p_pv: set point for the PV power on the DC side of the battery system
        :type p_pv: float
        :param soc: state of charge of the battery in 0-1 (e.g. 0%-100%)
        :type soc: float
        :param dt: time step width
        :type dt: integer
        """

        # Inputs
        p_r, p_rpv, p_pv_limited, p_pv2ac_ac_out = self.__get_residual_power(p_load, p_pv)

        # Current energy content of the battery
        e_b0 = soc * self._E_BAT

        # Check if the battery holds enough unused capacity for charging or discharging
        # Estimated amount of energy that is supplied to or discharged from the storage unit.
        p_r, p_rpv = self.__avoid_overcharging(p_r, p_rpv, e_b0, dt)

        battery_state = self.__get_battery_state(p_rpv, soc)

        # Realized AC power of the PV-battery system
        if battery_state == self._state.CHARGING:
            p_pvbs, p_bat = self.__charge_battery(p_rpv, p_pv_limited)

        elif battery_state == self._state.DISCHARGING:
            p_pvbs, p_bat = self.__discharge_battery(p_r, p_pv_limited, p_pv2ac_ac_out)

        else:  # Neither charging nor discharging of the battery
            # Realized AC power of the PV-battery system
            p_bat = 0.0
            p_pvbs = p_pv2ac_ac_out

        # Decision if the standby mode is active
        if p_bat == 0:
            p_pvbs, p_bat = self.__standby_mode(p_pvbs, p_bat, soc)

        # Change the energy content of the battery Wx to Wh conversion
        e_b = self.__change_energy_content(p_bat, e_b0, dt)

        # Calculate the state of charge of the battery
        soc = e_b / self._E_BAT

        # Adjust the hysteresis threshold to avoid alternation between charging
        # and standby mode due to the DC power consumption of the
        # PV-battery inverter
        self._threshold = bool(self._threshold and soc > self._SOC_THRESHOLD or soc > 1)

        return self._Results(p_pvbs=p_pvbs, p_bat=p_bat, soc=soc)


#if __name__ == "__main__":
#    ac_battery = ACBatMod("S2")
#    result = ac_battery.simulate(p_load=500, soc=0.0, dt=1)
#    dc_battery = DCBatMod("S3")
#    result = dc_battery.simulation(p_load=500, p_pv=350, soc=0, dt=1)
