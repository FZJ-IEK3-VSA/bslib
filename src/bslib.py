"""This module contains functions to simulate battery storage systems"""
import os
from typing import Dict
import pandas as pd
import numpy as np


# Load paramterS
def load_parameter(model_name: str = None) -> Dict:
    """Loads model specific parameters from the database.

    :param model_name: Model Name, defaults to None
    :type model_name: str, optional
    :return: Parameter of the specified model
    :rtype: dict
    """

    df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  "..",
                                                  "input",
                                                  "bslib_database.csv")))

    df = df.loc[df['ID'] == model_name]

    df.columns = df.columns.str.rstrip('[W]')
    df.columns = df.columns.str.rstrip('[V]')
    df.columns = df.columns.str.rstrip('[s]')
    df.columns = df.columns.str.rstrip('[-coupled]')
    df.columns = df.columns.str.strip()

    parameter: dict = df.to_dict(orient='records')[0]

    return parameter


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
    def __init__(self, sys_id: str = None, p_inv_custom: float = None, e_bat_custom: float = None):
        self.parameter = load_parameter(sys_id)
        self.model = self._get_model(self.parameter, p_inv_custom, e_bat_custom)

    @staticmethod
    def _get_model(parameter):
        if parameter['Type'] == 'AC':
            return ACBatMod

    def simulate(self):
        self.model.simulate()


class ACBatMod:
    def __init__(self, parameter: Dict, dt, *args):
        """Performance Simulation function for AC-coupled battery systems

        :param d: array containing parameters
        :type d: numpy array
        :param dt: time step width
        :type dt: integer
        """

        # Loading of particular variables
        self._dt = dt
        self._E_BAT = parameter['E_BAT']
        self._eta_BAT = parameter['eta_BAT']
        self._t_CONSTANT = parameter['t_CONSTANT']
        self._P_SYS_SOC0_DC = parameter['P_SYS_SOC0_DC']
        self._P_SYS_SOC0_AC = parameter['P_SYS_SOC0_AC']
        self._P_SYS_SOC1_DC = parameter['P_SYS_SOC1_DC']
        self._P_SYS_SOC1_AC = parameter['P_SYS_SOC1_AC']
        self._AC2BAT_a_in = parameter['AC2BAT_a_in']
        self._AC2BAT_b_in = parameter['AC2BAT_b_in']
        self._AC2BAT_c_in = parameter['AC2BAT_c_in']
        self._BAT2AC_a_out = parameter['BAT2AC_a_out']
        self._BAT2AC_b_out = parameter['BAT2AC_b_out']
        self._BAT2AC_c_out = parameter['BAT2AC_c_out']
        self._P_AC2BAT_DEV = parameter['P_AC2BAT_DEV']
        self._P_BAT2AC_DEV = parameter['P_BAT2AC_DEV']
        self._P_BAT2AC_out = parameter['P_BAT2AC_out']
        self._P_AC2BAT_in = parameter['P_AC2BAT_in']
        self._t_DEAD = int(round(parameter['t_DEAD']))
        self._SOC_h = parameter['SOC_h']

        if parameter['Manufacturer (PE)'] == 'Generic':
            self._PV_inv = args[0]  # Custom inverter power
            self._E_BAT = self._E_BAT * args[1]  # Custom battery capacity
            self._P_SYS_SOC1_DC = self._P_SYS_SOC1_DC * self._E_BAT  # Multi mit Kapazität in kWh
            self._P_SYS_SOC1_AC = self._P_SYS_SOC1_AC * self._PV_inv / 1000  # Multi mit WR-Leistung in W / 1000
            self._P_BAT2AC_out = self._P_BAT2AC_out * self._PV_inv / 1000  # Multi mit WR-Leistung in W / 1000
            self._P_AC2BAT_in = self._P_AC2BAT_in * self._PV_inv / 1000  # Multi mit WR-Leistung in W / 1000

        self._th = False

        self._P_AC2BAT_min = self._AC2BAT_c_in
        self._P_BAT2AC_min = self._BAT2AC_c_out

        # Correction factor to avoid overcharge and discharge the battery
        self.corr = 0.1

        # Initialization of particular variables
        # Binary variable to activate the first-order time delay element
        self._tde = self._t_CONSTANT > 0
        # Factor of the first-order time delay element
        self._ftde = 1 - np.exp(-self._dt / self._t_CONSTANT)
        # Capacity of the battery, conversion from kWh to Wh
        self._E_BAT *= 1000
        # Efficiency of the battery in percent
        self._eta_BAT /= 100

    def simulate(self, Pr: float, soc: float):

        # Inputs
        P_bs = Pr

        # Calculation
        # Energy content of the battery in the previous time step
        E_b0 = soc * self._E_BAT

        # Calculate the AC power of the battery system from the residual power

        # Check if the battery holds enough unused capacity for charging or discharging
        # Estimated amount of energy in Wh that is supplied to or discharged from the storage unit.
        E_bs_est = P_bs * self._dt / 3600

        # Reduce P_bs to avoid over charging of the battery
        if E_bs_est > 0 and E_bs_est > (self._E_BAT - E_b0):
            P_bs = (self._E_BAT - E_b0) * 3600 / self._dt
        # When discharging take the correction factor into account
        elif E_bs_est < 0 and np.abs(E_bs_est) > (E_b0):
            P_bs = (E_b0 * 3600 / self._dt) * (1 - self.corr)

        # Adjust the AC power of the battery system due to the stationary
        # deviations taking the minimum charging and discharging power into
        # account
        if P_bs > self._P_AC2BAT_min:
            P_bs = np.maximum(self._P_AC2BAT_min, P_bs + self._P_AC2BAT_DEV)

        elif P_bs < -self._P_BAT2AC_min:
            P_bs = np.minimum(-self._P_BAT2AC_min, P_bs - self._P_BAT2AC_DEV)

        else:
            P_bs = 0

        # Limit the AC power of the battery system to the rated power of the
        # battery converter
        P_bs = np.maximum(-self._P_BAT2AC_out * 1000,
                          np.minimum(self._P_AC2BAT_in * 1000, P_bs))

        # Decision if the battery should be charged or discharged
        # Charging
        if P_bs > 0 and soc < 1 - self._th * (1 - self._SOC_h):
            # The last term th*(1-SOC_h) avoids the alternation between
            # charging and standby mode due to the DC power consumption of the
            # battery converter when the battery is fully charged. The battery
            # will not be recharged until the SOC falls below the SOC-threshold
            # (SOC_h) for recharging from PV.

            # Normalized AC power of the battery system
            p_bs = P_bs / self._P_AC2BAT_in / 1000

            # DC power of the battery affected by the AC2BAT conversion losses
            # of the battery converter
            P_bat = np.maximum(0, P_bs - (self._AC2BAT_a_in * p_bs * p_bs
                                          + self._AC2BAT_b_in * p_bs
                                          + self._AC2BAT_c_in))
        # Discharging
        elif P_bs < 0 and soc > 0:

            # Normalized AC power of the battery system
            p_bs = np.abs(P_bs / self._P_BAT2AC_out / 1000)

            # DC power of the battery affected by the BAT2AC conversion losses
            # of the battery converter
            P_bat = P_bs - (self._BAT2AC_a_out * p_bs * p_bs
                            + self._BAT2AC_b_out * p_bs
                            + self._BAT2AC_c_out)

        # Neither charging nor discharging of the battery
        else:
            # Set the DC power of the battery to zero
            P_bat = 0

        # Decision if the standby mode is active
        if P_bat == 0 and soc <= 0:  # Standby mode in discharged state

            # DC and AC power consumption of the battery converter
            P_bat = -np.maximum(0, self._P_SYS_SOC0_DC)
            P_bs = self._P_SYS_SOC0_AC

        elif P_bat == 0 and soc > 0:  # Standby mode in fully charged state

            # DC and AC power consumption of the battery converter
            P_bat = -np.maximum(0, self._P_SYS_SOC1_DC)
            P_bs = self._P_SYS_SOC1_AC

        # Change the energy content of the battery from Ws to Wh conversion
        if P_bat > 0:
            E_b = E_b0 + P_bat * np.sqrt(self._eta_BAT) * self._dt / 3600

        elif P_bat < 0:
            E_b = E_b0 + P_bat / np.sqrt(self._eta_BAT) * self._dt / 3600

        else:
            E_b = E_b0

        # Calculate the state of charge of the battery
        soc = E_b / self._E_BAT

        # Adjust the hysteresis threshold to avoid alternation
        # between charging and standby mode due to the DC power
        # consumption of the battery converter.
        if self._th and soc > self._SOC_h or soc > 1:
            self._th = True
        else:
            self._th = False

        # Outputs
        return P_bs, soc


if __name__ == "__main__":
    print()
