"""This module contains functions to simulate battery storage systems"""
import os
from math import sqrt
from collections import namedtuple
from typing import Dict, Tuple, NamedTuple
import pandas as pd
import numpy as np


def load_parameters(system_id: str) -> Dict:
    """Loads model specific parameters from the database.

    :param system_id: ID of the Model in the database, defaults to None
    :type system_id: str, optional
    :return: Parameter of the specified model
    :rtype: dict
    """

    database = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                        "..",
                                                        "src",
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
    return pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..",
                                                    "src",
                                                    "bslib_database.csv")))


# Not used anymore included for legacy reasons
def transform_dict_to_array(parameter):
    """Function for transforming a dict to an numpy array

    :param parameter: dict of system parameters
    :type parameter: dict
    :return: array of system parameters
    :rtype: numpy array
    """
    # Der Standby-SOC1_AC
    if parameter['Type'] == 'AC':
        d = np.array(parameter['E_BAT'])  # 0 multi mit dem gewünschten Kapazität
        d = np.append(d, parameter['eta_BAT'])  # 1
        d = np.append(d, parameter['t_CONSTANT'])  # 2
        d = np.append(d, parameter['P_SYS_SOC0_DC'])  # 3
        d = np.append(d, parameter['P_SYS_SOC0_AC'])  # 4
        d = np.append(d, parameter['P_SYS_SOC1_DC'])  # 5 multi mit Kapazität in kWh
        d = np.append(d, parameter['P_SYS_SOC1_AC'])  # 6 multi mit WR-Leistung in W / 1000
        d = np.append(d, parameter['AC2BAT_a_in'])  # 7
        d = np.append(d, parameter['AC2BAT_b_in'])  # 8
        d = np.append(d, parameter['AC2BAT_c_in'])  # 9
        d = np.append(d, parameter['BAT2AC_a_out'])  # 10
        d = np.append(d, parameter['BAT2AC_b_out'])  # 11
        d = np.append(d, parameter['BAT2AC_c_out'])  # 12
        d = np.append(d, parameter['P_AC2BAT_DEV'])  # 13
        d = np.append(d, parameter['P_BAT2AC_DEV'])  # 14
        d = np.append(d, parameter['P_BAT2AC_out'])  # 15 multi mit gewünschten WR-Leistung in W / 1000
        d = np.append(d, parameter['P_AC2BAT_in'])  # 16 multi mit gewünschten WR-Leistung in W / 1000
        d = np.append(d, parameter['t_DEAD'])  # 17
        d = np.append(d, parameter['SOC_h'])  # 18

    if parameter['Type'] == 'DC':
        d = np.array(parameter['E_BAT'])  # 1
        d = np.append(d, parameter['P_PV2AC_in'])  # 2
        d = np.append(d, parameter['P_PV2AC_out'])  # 3
        d = np.append(d, parameter['P_PV2BAT_in'])  # 4
        d = np.append(d, parameter['P_BAT2AC_out'])  # 5
        d = np.append(d, parameter['PV2AC_a_in'])  # 6
        d = np.append(d, parameter['PV2AC_b_in'])  # 7
        d = np.append(d, parameter['PV2AC_c_in'])  # 8
        d = np.append(d, parameter['PV2BAT_a_in'])  # 9
        d = np.append(d, parameter['PV2BAT_b_in'])  # 10
        d = np.append(d, parameter['BAT2AC_a_out'])  # 11
        d = np.append(d, parameter['BAT2AC_b_out'])  # 12
        d = np.append(d, parameter['BAT2AC_c_out'])  # 13
        d = np.append(d, parameter['eta_BAT'])  # 14
        d = np.append(d, parameter['SOC_h'])  # 15
        d = np.append(d, parameter['P_PV2BAT_DEV'])  # 16
        d = np.append(d, parameter['P_BAT2AC_DEV'])  # 17
        d = np.append(d, parameter['t_DEAD'])  # 18
        d = np.append(d, parameter['t_CONSTANT'])  # 19
        d = np.append(d, parameter['P_SYS_SOC1_DC'])  # 20
        d = np.append(d, parameter['P_SYS_SOC0_AC'])  # 21
        d = np.append(d, parameter['P_SYS_SOC0_DC'])  # 22

    if parameter['Type'] == 'PV':
        d = np.array(parameter['E_BAT'])
        d = np.append(d, parameter['P_PV2AC_in'])
        d = np.append(d, parameter['P_PV2AC_out'])
        d = np.append(d, parameter['P_PV2BAT_in'])
        d = np.append(d, parameter['P_BAT2PV_out'])
        d = np.append(d, parameter['PV2AC_a_in'])
        d = np.append(d, parameter['PV2AC_b_in'])
        d = np.append(d, parameter['PV2AC_c_in'])
        d = np.append(d, parameter['PV2BAT_a_in'])
        d = np.append(d, parameter['PV2BAT_b_in'])
        d = np.append(d, parameter['PV2BAT_c_in'])
        d = np.append(d, parameter['PV2AC_a_out'])
        d = np.append(d, parameter['PV2AC_b_out'])
        d = np.append(d, parameter['PV2AC_c_out'])
        d = np.append(d, parameter['BAT2PV_a_out'])
        d = np.append(d, parameter['BAT2PV_b_out'])
        d = np.append(d, parameter['BAT2PV_c_out'])
        d = np.append(d, parameter['eta_BAT'])
        d = np.append(d, parameter['SOC_h'])
        d = np.append(d, parameter['P_PV2BAT_DEV'])
        d = np.append(d, parameter['P_BAT2AC_DEV'])
        d = np.append(d, parameter['P_SYS_SOC1_DC'])
        d = np.append(d, parameter['P_SYS_SOC0_AC'])
        d = np.append(d, parameter['P_SYS_SOC0_DC'])
        d = np.append(d, parameter['t_DEAD'])
        d = np.append(d, parameter['t_CONSTANT'])

    return d


# Einige Parameter bei generic angeben
# Parameter wie Standby Verlust sind abhängig von anderen Parametern und werden nachträglich brechnet.
# Generic Topologie angeben. Checken ob generic ausgewählt wurde.
# Kapazität und Leistung vom WR reteed Power AC-Seite angeben.


class Battery:
    def __init__(self,
                 sys_id: str = None,
                 *,
                 p_pv: float = None,  # rated power of the PV inverter in kWp ToDo Ist diese Variable noch notwendig?
                 p_inv_custom: float = None,  # power of the inverter in kW
                 e_bat_custom: float = None  # capacity of the battery in kWh
                 ):
        self.parameter = load_parameters(sys_id)
        if self.parameter['Type'] == 'DC' and p_pv is None:
            raise TypeError('Rated power of the PV generator is not specified.')

        if self.parameter['Manufacturer (PE)'] == 'Generic':
            if p_inv_custom is None:
                raise TypeError('Custom value for the inverter power is not specified.')
            if e_bat_custom is None:
                raise TypeError('Custom value for the battery capacity is not specified.')

        self.model = self.__load_model(p_inv_custom, e_bat_custom)

    def __load_model(self, p_inv_custom, e_bat_custom):
        if self.parameter['Type'] == 'AC':
            return ACBatMod(self.parameter, p_inv_custom=p_inv_custom, e_bat_custom=e_bat_custom)
        if self.parameter['Type'] == 'DC':
            pass
            # return DCBatMod(self.parameter, p_pv, p_inv_custom, e_bat_custom)

    def simulate(self, **kwargs):
        return self.model.simulate(**kwargs)


class ACBatMod:
    def __init__(self, system_id: str, *, p_inv_custom: float = None, e_bat_custom: float = None):
        r"""Performance simulation class for AC-coupled battery systems

        :param system_id: Parameters of the battery system
        :type system_id: dict
        :param p_inv_custom: Custom values for the PV inverter when using a generic system
        :type p_inv_custom: float
        :param e_bat_custom: Custom value for the battery capacity when using a generic system
        :type e_bat_custom: float
        """
        # Load system parameter according to the given system id
        self.__parameter = load_parameters(system_id)

        if self.__parameter['Manufacturer (PE)'] == 'Generic':
            if p_inv_custom is None:
                raise TypeError('Custom value for the inverter power is not specified.')
            if e_bat_custom is None:
                raise TypeError('Custom value for the battery capacity is not specified.')

        # System sizing
        self._E_BAT = self.__parameter['E_BAT'] * 1000  # Mean usable battery capacity in Wh
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
        self._P_PERI = self.__parameter['P_PERI_AC']  # AC power consumption of other system components in W

        if self.__parameter['Manufacturer (PE)'] == 'Generic':
            self._P_AC2BAT_IN = p_inv_custom  # Custom inverter power in kW
            self._P_BAT2AC_OUT = p_inv_custom  # Custom inverter power in kW
            self._E_BAT = e_bat_custom * 1000  # Custom battery capacity Wh
            self._P_SYS_SOC1_DC = self._P_SYS_SOC1_DC * self._E_BAT
            self._P_SYS_SOC1_AC = self._P_SYS_SOC1_AC * self._P_BAT2AC_OUT

        # Minimum AC charging power
        self._P_AC2BAT_MIN = self._AC2BAT_C_IN
        # Minimum AC discharging power
        self._P_BAT2AC_MIN = self._BAT2AC_C_OUT

        # Correction factor to avoid overcharging of the battery
        self._CORR_FACTOR = 0.1

        # Stores the AC power and the state of charge of the battery system
        self._Results = namedtuple("Results", ["p_bs", "p_bat", "soc"])

    def simulate(self, *, p_set: float, soc: float, dt: int) -> NamedTuple[float, float, float]:
        r"""Simulation function for AC-coupled battery systems.

        :param p_set: Set point for the power on the AC side of the battery system in W.
                      A positive value represents an excess power.
                      A negative value represents a power demand.
        :type p_set: float

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
        p_bs = p_set - self._P_PERI

        # Energy content of the battery before calculations in Ws
        e_b0 = soc * self._E_BAT

        # Check if the battery holds enough unused capacity for charging or discharging
        # Estimated amount of energy in Wh that is supplied to or discharged from the storage unit.
        e_bs_est = p_bs * dt / 3600
        # Reduce the power on the AC side to avoid over charging of the battery
        if e_bs_est > 0 and e_bs_est > (self._E_BAT - e_b0):
            p_bs = (self._E_BAT - e_b0) * 3600 / dt
        # When discharging take the correction factor into account
        elif e_bs_est < 0 and abs(e_bs_est) > e_b0:
            p_bs = -((e_b0 * 3600 / dt) * (1 - self._CORR_FACTOR))

        # Adjust the AC power of the battery system due to the stationary
        # deviations, taking the minimum charging and discharging power into
        # account
        if p_bs > self._P_AC2BAT_MIN:
            p_bs = max(self._P_AC2BAT_MIN, p_bs + self._P_AC2BAT_DEV)

        elif p_bs < -self._P_BAT2AC_MIN:
            p_bs = min(-self._P_BAT2AC_MIN, p_bs - self._P_BAT2AC_DEV)

        else:
            p_bs = 0.0

        # Limit the AC power of the battery system to the rated power of the
        # battery converter
        p_bs = max(-self._P_BAT2AC_OUT * 1000,
                   min(self._P_AC2BAT_IN * 1000, p_bs))

        # Check whether the battery will be charged or discharged
        if p_bs > 0 and soc < 1 - self._threshold * (1 - self._SOC_THRESHOLD):  # Charging
            # The second operand avoids the alternation between charging and standby mode
            # due to the DC power consumption of the battery converter when the battery is fully charged.
            # The battery will not be recharged until the SOC falls below the SOC-threshold
            # for recharging from PV.

            # Normalized AC power of the battery system
            p_bs_norm = p_bs / self._P_AC2BAT_IN / 1000

            # DC power of the battery affected by the AC2BAT conversion losses
            # of the battery converter in W
            p_bat = max(0, p_bs - (self._AC2BAT_A_IN * p_bs_norm * p_bs_norm
                                   + self._AC2BAT_B_IN * p_bs_norm
                                   + self._AC2BAT_C_IN))

        elif p_bs < 0 and soc > 0:  # Discharging
            # Normalized AC power of the battery system
            p_bs_norm = abs(p_bs / self._P_BAT2AC_OUT / 1000)

            # DC power of the battery affected by the BAT2AC conversion losses
            # of the battery converter in W
            p_bat = p_bs - (self._BAT2AC_A_OUT * p_bs_norm * p_bs_norm
                            + self._BAT2AC_B_OUT * p_bs_norm
                            + self._BAT2AC_C_OUT)

        else:  # Neither charging nor discharging of the battery
            # Set the DC power of the battery to 0W
            p_bat = 0.0

        # Check if the battery system is in standby mode
        if p_bat == 0:
            if soc <= 0:  # Standby mode in discharged state
                # DC power consumption of the battery converter in W
                p_bat = -max(0, self._P_SYS_SOC0_DC)
                # AC power consumption of the battery converter in W
                p_bs = self._P_SYS_SOC0_AC

            else:  # Standby mode in fully charged state
                # DC consumption of the battery converter in W
                p_bat = -max(0, self._P_SYS_SOC1_DC)
                # AC power consumption of the battery converter in W
                p_bs = self._P_SYS_SOC1_AC

        # Change the energy content of the battery and conversion from Ws to Wh
        if p_bat > 0:  # Charging
            e_b = e_b0 + p_bat * sqrt(self._ETA_BAT) * dt / 3600

        elif p_bat < 0: # Discharging
            e_b = e_b0 + p_bat / sqrt(self._ETA_BAT) * dt / 3600

        else:
            e_b = e_b0

        # Calculate the resulting state of charge of the battery
        soc = e_b / self._E_BAT

        # Adjust the hysteresis threshold to avoid alternation
        # between charging and standby mode due to the DC power
        # consumption of the battery converter.
        self._threshold = bool(self._threshold and soc > self._SOC_THRESHOLD or soc > 1)

        # Outputs
        return self._Results(p_bs=p_bs, p_bat=p_bat, soc=soc)


# Check in DCBATMOD if when discharging the value has to be negative
"""
# When discharging take the correction factor into account
        elif E_bs_r < 0 and np.abs(E_bs_r) > E_b0:
            P_r = !!!-!!!(E_b0 * 3600 / _dt) * (1 - corr)
"""


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

    def __init__(self, system_id: str, *, p_inv_custom=None, e_bat_custom=None):
        """Constructor method
        """
        # Load system parameter according to the given system id
        self.__parameter = load_parameters(system_id)

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
        self._P_SYS_SOC1_DC = self.__parameter["P_SYS_SOC1_DC"]  # DC standby power consumption in charged state in W
        self._P_SYS_SOC0_AC = self.__parameter["P_SYS_SOC0_AC"]  # AC standby power consumption in discharged state in W
        self._P_SYS_SOC0_DC = self.__parameter["P_SYS_SOC0_DC"]  # DC standby power consumption in discharged state in W
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

    def __get_residual_power(self, p_load, p_pv_limited):

        # Power demand on the AC side of the battery system respecting the power consumption of other system components
        p_ac = p_load + self._P_PERI_AC

        # Normalized AC output power of the PV2AC conversion pathway to cover the AC power demand
        ppv2ac_ac_norm = min(p_ac, self._P_PV2AC_OUT * 1000) / self._P_PV2AC_OUT / 1000

        # Target DC input power of the PV2AC conversion pathway
        p_pv2ac_dc_in = min(p_ac, self._P_PV2AC_OUT * 1000)+(self._PV2AC_A_OUT * ppv2ac_ac_norm * ppv2ac_ac_norm
                                                             + self._PV2AC_B_OUT * ppv2ac_ac_norm
                                                             + self._PV2AC_C_OUT)

        # Normalized DC input power of the PV2AC conversion pathway
        ppv2ac_dc_norm = p_pv_limited / self._P_PV2AC_IN / 1000

        # Target AC output power of the PV2AC conversion pathway
        p_pv2ac_ac_out = max(0, p_pv_limited-(self._PV2AC_A_IN * ppv2ac_dc_norm * ppv2ac_dc_norm
                                              + self._PV2AC_B_IN * ppv2ac_dc_norm
                                              + self._PV2AC_C_IN))

        # Residual power for battery charging
        p_rpv = p_pv_limited - p_pv2ac_dc_in

        # Residual power for battery discharging
        p_r = p_pv2ac_ac_out - p_ac

        return p_r, p_rpv

    def simulation(self, *, p_load, p_pv, soc, dt):
        """Performance simulation function for DC-coupled battery systems

        :param p_ac: AC set point for the AC power on the AC side of the battery system
        :type p_ac: float
        :param p_pv: set point for the PV power on the DC side of the battery system
        :type p_pv: float
        :param soc: state of charge of the battery in 0-1 (e.g. 0%-100%)
        :type soc: float
        :param dt: time step width
        :type dt: integer
        """

        # Inputs
        # Output of the PV generator limited to the maximum DC input power of the PV2AC conversion pathway
        p_pv_limited = min(p_pv, self._P_PV2AC_IN * 1000)
        p_r, p_rpv = self.__get_residual_power(p_load, p_pv_limited)

        # Energy content of the battery in the previous time step
        e_b0 = soc * self._E_BAT

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

        # Decision if the battery should be charged or discharged
        if p_rpv > 0 and soc < 1 - self._threshold * (1 - self._SOC_THRESHOLD):  # Charging the battery
            '''
            The last term th*(1-SOC_h) avoids the alternation between
            charging and standby mode due to the DC power consumption of the
            battery converter when the battery is fully charged. The battery
            will not be recharged until the SOC falls below the SOC-threshold
            (SOC_h) for recharging from PV.
            '''
            # Charging power
            p_pv2bat = p_rpv

            # Adjust the charging power due to the stationary deviations
            p_pv2bat = max(0, p_pv2bat + self._P_PV2BAT_DEV)

            # Limit the charging power to the maximum charging power
            p_pv2bat = min(p_pv2bat, self._P_PV2BAT_IN * 1000)

            # Limit the charging power to the current power output of the PV generator
            p_pv2bat = min(p_pv2bat, p_pv_limited)

            # Normalized charging power
            p_pv2bat_norm = p_pv2bat / self._P_PV2BAT_IN / 1000

            # DC power of the battery affected by the PV2BAT conversion losses
            # (the idle losses of the PV2BAT conversion pathway are not taken
            # into account)
            p_bat = max(0, p_pv2bat - (self._PV2BAT_A_IN * p_pv2bat_norm * p_pv2bat_norm
                                       + self._PV2BAT_B_IN * p_pv2bat_norm))

            # Realized DC input power of the PV2AC conversion pathway
            p_pv2ac_dc_in = p_pv_limited - p_pv2bat

            # Normalized DC input power of the PV2AC conversion pathway
            p_pv2ac_dc_in_norm = p_pv2ac_dc_in / self._P_PV2AC_IN / 1000

            # Realized AC power of the PV-battery system
            p_pv2ac_ac_out = max(0, p_pv2ac_dc_in - (self._PV2AC_A_IN * p_pv2ac_dc_in_norm * p_pv2ac_dc_in_norm
                                              + self._PV2AC_B_IN * p_pv2ac_dc_in_norm
                                              + self._PV2AC_C_IN))
            p_pvbs = p_pv2ac_ac_out

        elif p_rpv < 0 and soc > 0:  # Discharging the battery

            # Discharging power
            p_bat2ac = p_r * - 1

            # Normalized DC input power of the PV2AC conversion pathway
            p_pv2ac_dc_in_norm = p_pv_limited / self._P_PV2AC_IN / 1000

            # Target AC output power of the PV2AC conversion pathway
            p_pv2ac_ac_out = max(0, p_pv_limited - (self._PV2AC_A_IN * p_pv2ac_dc_in_norm * p_pv2ac_dc_in_norm
                                                    + self._PV2AC_B_IN * p_pv2ac_dc_in_norm
                                                    + self._PV2AC_C_IN))

            # Adjust the discharging power due to the stationary deviations
            p_bat2ac = max(0, p_bat2ac + self._P_BAT2AC_DEV)

            # Adjust the discharging power to the maximum discharging power
            p_bat2ac = min(p_bat2ac, self._P_BAT2AC_OUT * 1000)

            # Limit the discharging power to the maximum AC power output of the PV-battery system
            p_bat2ac = min(self._P_PV2AC_OUT * 1000 - p_pv2ac_ac_out, p_bat2ac)

            # Normalized discharging power
            p_pv2bat_norm = p_bat2ac / self._P_BAT2AC_OUT / 1000

            # DC power of the battery affected by the BAT2AC conversion losses
            # (if the idle losses of the PV2AC conversion pathway are covered by
            # the PV generator, the idle losses of the BAT2AC conversion pathway
            # are not taken into account)
            if p_pv_limited > self._P_PV2AC_MIN:
                p_bat = -1 * (p_bat2ac + (self._BAT2AC_A_OUT * p_pv2bat_norm * p_pv2bat_norm
                                          + self._BAT2AC_B_OUT * p_pv2bat_norm))
            else:
                p_bat = -1 * (p_bat2ac + (self._BAT2AC_A_OUT * p_pv2bat_norm * p_pv2bat_norm
                                          + self._BAT2AC_B_OUT * p_pv2bat_norm
                                          + self._BAT2AC_C_OUT)) + p_pv_limited

            # Realized AC power of the PV-battery system
            p_pvbs = p_pv2ac_ac_out + p_bat2ac

        else:  # Neither charging nor discharging of the battery
            # Normalized DC input power of the PV2AC conversion pathway
            p_pv2ac_dc_in_norm = p_pv_limited / self._P_PV2AC_IN / 1000

            # Target AC output power of the PV2AC conversion pathway
            p_pv2ac_ac_out = max(0, p_pv_limited - (self._PV2AC_A_IN * p_pv2ac_dc_in_norm * p_pv2ac_dc_in_norm
                                                    + self._PV2AC_B_IN * p_pv2ac_dc_in_norm
                                                    + self._PV2AC_C_IN))
            # Set the DC power of the battery to zero
            p_bat = 0

            # Realized AC power of the PV-battery system
            p_pvbs = p_pv2ac_ac_out

        # Decision if the standby mode is active
        if p_bat == 0 and p_pvbs == 0 and soc <= 0:  # Standby mode in discharged state

            # DC and AC power consumption of the PV-battery inverter
            p_bat = -np.maximum(0, self._P_SYS_SOC0_DC)
            p_pvbs = -self._P_SYS_SOC0_AC

        elif p_bat == 0 and p_pvbs > 0 and soc > 0:  # Standby mode in fully charged state

            # DC power consumption of the PV-battery inverter
            p_bat = -np.maximum(0, self._P_SYS_SOC1_DC)

        # Transfer the realized AC power of the PV-battery system and the DC power of the battery
        _Ppvbs = p_pvbs
        _Pbat = p_bat

        # Change the energy content of the battery Wx to Wh conversion
        if p_bat > 0:
            E_b = e_b0 + p_bat * np.sqrt(self._eta_BAT) * dt / 3600
        elif p_bat < 0:
            E_b = e_b0 + p_bat / np.sqrt(self._eta_BAT) * dt / 3600
        else:
            E_b = e_b0

        # Calculate the state of charge of the battery
        soc = E_b / self._E_BAT

        # Adjust the hysteresis threshold to avoid alternation between charging
        # and standby mode due to the DC power consumption of the
        # PV-battery inverter
        if self._threshold and soc > self._SOC_THRESHOLD or soc > 1:
            self._threshold = True
        else:
            self._threshold = False

        return _Ppv2ac_out, _Ppvbs, _Pbat, soc


if __name__ == "__main__":
    
    # pv_inv = 1  # kW
    # e_bat = 5  # kWh
    #
    # battery = Battery(sys_id='S2', p_inv_custom=pv_inv, e_bat_custom=e_bat)
    #
    # dt = 1  # delta-t
    # soc = 0.0  # state of charge at 0%
    # p_set = 123
    #
    # results = battery.simulate(p_set=p_set, soc=soc, dt=dt)

    # # Array for testing with a timestep width of 15 minutes
    # test_values = np.empty(int(525600 / 15), )
    # test_values[:int(525600 / 15 / 2)] = 3000
    # test_values[int(525600 / 15 / 2):] = -3830
    # p_bs_list = []
    # soc_list = []
    """ From step 22 P_bs changes to 14.9.
    # In the block for charging 
    P_bat = np.maximum(0, P_bs - (self._AC2BAT_a_in * p_bs * p_bs
                                  + self._AC2BAT_b_in * p_bs
                                  + self._AC2BAT_c_in))
    P_bat is calculated as 0.0
    """
    # for value in test_values:
    #     p_bs, soc = battery.simulate(P_setpoint=value, soc=soc, dt=dt)
    #     p_bs_list.append(p_bs)
    #     soc_list.append(soc)

    # d = {'Test_values': test_values, 'P_BS': p_bs_list, 'SOC': soc_list}
    # df = pd.DataFrame(d)
    # df['SOC'] = df['SOC'] * 100
    # print(max(soc_list))
