"""This files contains function to load and edit the data base for bslib"""
import os

import pandas as pd
import numpy as np


def read_excel_to_df():
    """This functions reads an Microsoft Excel file and returns it as an pandas data frame"""
    df = pd.read_excel(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  "..",
                                                  "input",
                                                  "PerModPAR.xlsx")), sheet_name='Data')

    # Drop all columns after End marker
    end = df.loc[:, (df == 'End').any()].columns
    end_name = end.values[0]
    df = df.loc[:, :end_name]
    df = df.iloc[:, :-1]

    # Drop first 4 columns
    df = df.iloc[:, 4:]

    df = df.drop('Unit', axis=1)

    df = df.dropna(how="all", axis=1)

    return df


def assign_specific_values(parameter):
    # Assign specific parameters
    # parameter['P_PV2AC_out_PVINV'] = parameter[col_name][15].value
    parameter['P_PV2AC_out_PVINV'] = parameter['P_PV2AC_out']
    # parameter['P_PV2AC_out'] = ws[col_name][24].value
    parameter['P_PV2AC_out'] = parameter['P_PV2AC_out [W].1']
    # parameter['P_AC2BAT_in_DCC'] = ws[col_name][25].value
    parameter['P_AC2BAT_in_DCC'] = parameter['P_AC2BAT_in']
    # parameter['P_AC2BAT_in'] = ws[col_name][26].value
    parameter['P_AC2BAT_in'] = parameter['P_AC2BAT_in [W].1']
    # parameter['P_BAT2AC_out'] = ws[col_name][27].value
    parameter['P_BAT2AC_out'] = parameter['P_BAT2AC_out']
    # parameter['P_BAT2AC_out_DCC'] = ws[col_name][28].value
    parameter['P_BAT2AC_out_DCC'] = parameter['P_BAT2AC_out [W].1']

    # Specific parameters of DC-coupled systems
    if parameter['Type'] == 'DC':
        parameter['P_AC2BAT_in'] = parameter['P_AC2BAT_in_DCC']  # Nominal charging power (AC) in kW
        parameter['P_BAT2AC_out'] = parameter['P_BAT2AC_out_DCC']

    # Specific parameters of PV inverters and AC-coupled systems
    if parameter['Type'] == 'PVINV' or parameter['Type'] == 'AC' and \
            parameter['P_PV2AC_out_PVINV'] is not None:
        parameter['P_PV2AC_out'] = parameter['P_PV2AC_out_PVINV']

    # Specific parameters of PV-coupled systems
    if parameter['Type'] == 'PV':
        parameter['P_BAT2PV_in'] = parameter['P_BAT2AC_in']
        parameter['P_BAT2AC_out'] = parameter['P_BAT2AC_out_DCC']

    # Convert to kW
    convert_to_kw = ['P_PV2AC_in',
                     'P_PV2AC_out_PVINV',
                     'P_PV2AC_out',
                     'P_AC2BAT_in_DCC',
                     'P_AC2BAT_in',
                     'P_BAT2AC_out',
                     'P_BAT2AC_out_DCC',
                     'P_PV2BAT_in',
                     'P_BAT2PV_out',
                     'P_PV2BAT_out',
                     'P_BAT2AC_in'
                     ]

    for par in convert_to_kw:
        if parameter[par] is not None:
            parameter[par] = parameter[par] / 1000

    return parameter


def transpose_df(df):
    """This functions transposes the DataFrame

    :param df: DataFrame
    :type df: pandas DataFrame
    :return: DataFrame
    :rtype: pandas DataFrame
    """

    df = df.T
    header_row = 0
    df.columns = df.iloc[header_row]
    df = df.iloc[1:, :]
    df = df.iloc[:, 1:]

    # Drop columns which are empty
    df = df.loc[:, df.columns.notnull()]

    # Fill emtpy cells with nan
    df = df.replace(r'^\s+$', np.nan, regex=True)

    return df


def drop_columns(df):
    """This function drops certain columns from the DataFrame

    :param df: DataFrame
    :type df: pandas DataFrame
    :return: DataFrame
    :rtype: pandas DataFrame
    """

    drop_cols = [
        "Type",
        "p_PV2AC_5",
        "p_PV2AC_10",
        "p_PV2AC_20",
        "p_PV2AC_25",
        "p_PV2AC_30",
        "p_PV2AC_50",
        "p_PV2AC_75",
        "p_PV2AC_100",
        'eta_PV2AC_5',
        'eta_PV2AC_10',
        'eta_PV2AC_20',
        'eta_PV2AC_25',
        'eta_PV2AC_30',
        'eta_PV2AC_50',
        'eta_PV2AC_75',
        'eta_PV2AC_100',
        'p_PV2BAT_5',
        'p_PV2BAT_10',
        'p_PV2BAT_20',
        'p_PV2BAT_25',
        'p_PV2BAT_30',
        'p_PV2BAT_50',
        'p_PV2BAT_75',
        'p_PV2BAT_100',
        'eta_PV2BAT_5',
        'eta_PV2BAT_10',
        'eta_PV2BAT_20',
        'eta_PV2BAT_25',
        'eta_PV2BAT_30',
        'eta_PV2BAT_50',
        'eta_PV2BAT_75',
        'eta_PV2BAT_100',
        'p_AC2BAT_5',
        'p_AC2BAT_10',
        'p_AC2BAT_20',
        'p_AC2BAT_25',
        'p_AC2BAT_30',
        'p_AC2BAT_50',
        'p_AC2BAT_75',
        'p_AC2BAT_100',
        'eta_AC2BAT_5',
        'eta_AC2BAT_10',
        'eta_AC2BAT_20',
        'eta_AC2BAT_25',
        'eta_AC2BAT_30',
        'eta_AC2BAT_50',
        'eta_AC2BAT_75',
        'eta_AC2BAT_100',
        'p_BAT2AC_5',
        'p_BAT2AC_10',
        'p_BAT2AC_20',
        'p_BAT2AC_25',
        'p_BAT2AC_30',
        'p_BAT2AC_50',
        'p_BAT2AC_75',
        'p_BAT2AC_100',
        'eta_BAT2AC_5',
        'eta_BAT2AC_10',
        'eta_BAT2AC_20',
        'eta_BAT2AC_25',
        'eta_BAT2AC_30',
        'eta_BAT2AC_50',
        'eta_BAT2AC_75',
        'eta_BAT2AC_100',
        'p_BAT2PV_5',
        'p_BAT2PV_10',
        'p_BAT2PV_20',
        'p_BAT2PV_25',
        'p_BAT2PV_30',
        'p_BAT2PV_50',
        'p_BAT2PV_75',
        'p_BAT2PV_100',
        'eta_BAT2PV_5',
        'eta_BAT2PV_10',
        'eta_BAT2PV_20',
        'eta_BAT2PV_25',
        'eta_BAT2PV_30',
        'eta_BAT2PV_50',
        'eta_BAT2PV_75',
        'eta_BAT2PV_100',
        'ref_1',
        'ref_2',
        'Man3',
        'Pro3',
        'Info',
        'Cat'
    ]

    df = df.drop(drop_cols, axis=1)

    return df


def rename_columns(df):
    """This function renames certain columns of the DataFrame

    :param df: DataFrame
    :type df: pandas DataFrame
    :return: DataFrame
    :rtype: pandas DataFrame
    """
    renamed_cols = {"Man1": "Manufacturer (PE)",
                    "Pro1": "Model (PE)",
                    "Man2": "Manufacturer (BAT)",
                    "Pro2": "Model (BAT)",
                    "Top": "Type [-coupled]",
                    'P_PV2AC_in': 'P_PV2AC_in [W]',
                    'P_PV2AC_out': 'P_PV2AC_out [W]',
                    'U_PV_min': 'U_PV_min [V]',
                    'U_PV_nom': 'U_PV_nom [V]',
                    'U_PV_max': 'U_PV_max [V]',
                    'U_MPP_min': 'U_MPP_min [V]',
                    'U_MPP_max': 'U_MPP_max [V]',
                    'P_AC2BAT_in': 'P_AC2BAT_in [W]',
                    'P_BAT2AC_out': 'P_BAT2AC_out [W]',
                    'P_PV2BAT_in': 'P_PV2BAT_in [W]',
                    'P_BAT2PV_out': 'P_BAT2PV_out [W]',
                    'P_PV2BAT_out': 'P_PV2BAT_out [W]',
                    'P_BAT2AC_in': 'P_BAT2AC_in [W]',
                    'U_BAT_min': 'U_BAT_min [V]',
                    'U_BAT_nom': 'U_BAT_nom [V]',
                    'U_BAT_max': 'U_BAT_max [V]',
                    'E_BAT_100': 'E_BAT_100 [kWh]',
                    'E_BAT_50': 'E_BAT_50 [kWh]',
                    'E_BAT_25': 'E_BAT_25 [kWh]',
                    'E_BAT_usable': 'E_BAT_usable [kWh]',
                    'eta_BAT_100': 'eta_BAT_100',
                    'eta_BAT_50': 'eta_BAT_50',
                    'eta_BAT_25': 'eta_BAT_25',
                    'eta_BAT': 'eta_BAT',
                    'P_SYS_SOC1_AC': 'P_SYS_SOC1_AC [W]',
                    'P_SYS_SOC1_DC': 'P_SYS_SOC1_DC [W]',
                    'P_SYS_SOC0_AC': 'P_SYS_SOC0_AC [W]',
                    'P_SYS_SOC0_DC': 'P_SYS_SOC0_DC [W]',
                    'P_PVINV_AC': 'P_PVINV_AC [W]',
                    'P_PERI_AC': 'P_PERI_AC [W]',
                    'P_PV2BAT_DEV_IMPORT': 'P_PV2BAT_DEV_IMPORT [W]',
                    'P_PV2BAT_DEV_EXPORT': 'P_PV2BAT_DEV_EXPORT [W]',
                    'P_BAT2AC_DEV_IMPORT': 'P_BAT2AC_DEV_IMPORT [W]',
                    'P_BAT2AC_DEV_EXPORT': 'P_BAT2AC_DEV_EXPORT [W]',
                    't_DEAD': 't_DEAD [s]',
                    't_SETTLING': 't_SETTLING [s]'
                    }

    return df.rename(columns=renamed_cols)


def export_to_csv(df):
    """This function exports a DataFrame to a CSV file.

    :param df: DataFrame
    :type df: pandas DataFrame
    """
    df.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                  "..",
                                                  "input",
                                                  "bslib_database.csv")), index=False)


def convert_to_nan(df):
    """This function converts certain values to numpy nan values.

    :param df: DataFrame
    :type df: pandas DataFrame
    :return: DataFrame
    :rtype: pandas DataFrame
    """

    df[df == 'ns'] = np.nan
    df[df == 'o'] = np.nan
    df[df == 'c'] = np.nan
    df[df == ' '] = np.nan
    df[df == 'nan'] = np.nan
    return df


def eta2abc(parameter):
    """Function to calculate the parameters of the power loss functions (quadratic equations) from the path efficiencies

    :param parameter: Holds parameters of the system
    :type parameter: dict
    :return: Dictionary holding parameters from the Excel sheet
    :rtype: dict
    """
    # PV2AC conversion pathway TODO
    if parameter['Top'] == 'DC' or parameter['Top'] == 'PVINV' or parameter['Top'] == 'PV' and parameter[
        'P_PV2AC_out'] is not None or parameter['Top'] == 'AC' and parameter['P_PV2AC_out'] is not None:
        # Create variables for the sampling points and corresponding efficiencies TODO
        p_pv2ac = np.fromiter((value for key, value in parameter.items() if 'p_PV2AC_' in key and value is not None),
                              float)
        eta_pv2ac = np.fromiter(
            (value / 100 for key, value in parameter.items() if 'eta_PV2AC_' in key and value is not None), float)

        # Absolute input and output power in W
        p_pv2ac_out = parameter['P_PV2AC_out'] * p_pv2ac * 1000
        p_pv2ac_in = p_pv2ac_out / eta_pv2ac

        # Absolute power loss in W
        P_l_pv2ac_in = (1 - eta_pv2ac) * p_pv2ac_in
        P_l_pv2ac_out = (1 / eta_pv2ac - 1) * p_pv2ac_out

        # Polynomial curve fitting parameters of the power loss functions in W

        # Based on input power
        p = np.polyfit(p_pv2ac_in / parameter['P_PV2AC_in'] / 1000, P_l_pv2ac_in, 2)
        parameter['PV2AC_a_in'] = p[0]
        parameter['PV2AC_b_in'] = p[1]
        parameter['PV2AC_c_in'] = p[2]

        # Based on output power
        p = np.polyfit(p_pv2ac, P_l_pv2ac_out, 2)
        parameter['PV2AC_a_out'] = p[0]
        parameter['PV2AC_b_out'] = p[1]
        parameter['PV2AC_c_out'] = p[2]

    # PV2BAT conversion pathway
    if parameter['Top'] == 'DC' or parameter['Top'] == 'PV':

        # Create variables for the sampling points and corresponding efficiencies
        p_pv2bat = np.array([value for key, value in parameter.items() if 'p_PV2BAT_' in key])
        eta_pv2bat = np.array([value / 100 for key, value in parameter.items() if 'eta_PV2BAT_' in key])

        # Create missing variables

        # Nominal input power of the PV2BAT conversion pathway of DC-coupled systems
        if parameter['P_PV2BAT_in'] is None:
            parameter['P_PV2BAT_in'] = parameter['P_PV2BAT_out'] / (parameter['eta_PV2BAT_100'] / 100)

        # Absolute input and output power in W
        p_pv2bat_out = parameter['P_PV2BAT_out'] * p_pv2bat * 1000
        p_pv2bat_in = p_pv2bat_out / eta_pv2bat

        # Absolute power loss in W
        P_l_pv2bat_in = (1 - eta_pv2bat) * p_pv2bat_in
        P_l_pv2bat_out = (1 / eta_pv2bat - 1) * p_pv2bat_out

        # Polynomial curve fitting parameters of the power loss functions in W

        # Based on input power
        p = np.polyfit(p_pv2bat_in / parameter['P_PV2BAT_in'] / 1000, P_l_pv2bat_in, 2)
        parameter['PV2BAT_a_in'] = p[0]
        parameter['PV2BAT_b_in'] = p[1]
        parameter['PV2BAT_c_in'] = p[2]

        # Based on output power
        p = np.polyfit(p_pv2bat, P_l_pv2bat_out, 2)
        parameter['PV2BAT_a_out'] = p[0]
        parameter['PV2BAT_b_out'] = p[1]
        parameter['PV2BAT_c_out'] = p[2]

    # AC2BAT conversion pathway
    if parameter['Top'] == 'AC' or parameter['Top'] == 'DC' and parameter['P_AC2BAT_in'] is not None:
        # Create variables for the sampling points and corresponding efficiencies TODO
        p_ac2bat = np.fromiter((value for key, value in parameter.items() if 'p_AC2BAT_' in key), float)
        eta_ac2bat = np.fromiter((value / 100 for key, value in parameter.items() if 'eta_AC2BAT_' in key), float)

        # Absolute input and output power in W
        p_ac2bat_out = parameter['P_PV2BAT_out'] * p_ac2bat * 1000
        p_ac2bat_in = p_ac2bat_out / eta_ac2bat

        # Absolute power loss in W
        P_l_ac2bat_in = (1 - eta_ac2bat) * p_ac2bat_in
        P_l_ac2bat_out = (1 / eta_ac2bat - 1) * p_ac2bat_out

        # Polynomial curve fitting parameters of the power loss functions in W

        # Based on input power
        p = np.polyfit(p_ac2bat_in / parameter['P_AC2BAT_in'] / 1000, P_l_ac2bat_in, 2)
        parameter['AC2BAT_a_in'] = p[0]
        parameter['AC2BAT_b_in'] = p[1]
        parameter['AC2BAT_c_in'] = p[2]

        # Based on output power
        p = np.polyfit(p_ac2bat, P_l_ac2bat_out, 2)
        parameter['AC2BAT_a_out'] = p[0]
        parameter['AC2BAT_b_out'] = p[1]
        parameter['AC2BAT_c_out'] = p[2]

    # BAT2AC conversion pathway
    if parameter['Top'] == 'AC' or parameter['Top'] == 'DC' or parameter['Top'] == 'PV' and parameter[
        'P_BAT2AC_out'] is not None:
        # Create variables for the sampling points and corresponding efficiencies TODO
        p_bat2ac = np.fromiter((value for key, value in parameter.items() if 'p_BAT2AC_' in key), float)
        eta_bat2ac = np.fromiter((value / 100 for key, value in parameter.items() if 'eta_BAT2AC_' in key), float)

        # Absolute input and output power in W
        p_bat2ac_out = parameter['P_BAT2AC_out'] * p_bat2ac * 1000
        p_bat2ac_in = p_bat2ac_out / eta_bat2ac

        # Absolute power loss in W
        P_l_bat2ac_in = (1 - eta_bat2ac) * p_bat2ac_in
        P_l_bat2ac_out = (1 / eta_bat2ac - 1) * p_bat2ac_out

        # Polynomial curve fitting parameters of the power loss functions in W

        # Based on input power
        p = np.polyfit(p_bat2ac_in / parameter['P_BAT2AC_in'] / 1000, P_l_bat2ac_in, 2)
        parameter['BAT2AC_a_in'] = p[0]
        parameter['BAT2AC_b_in'] = p[1]
        parameter['BAT2AC_c_in'] = p[2]

        # Based on output power
        p = np.polyfit(p_bat2ac, P_l_bat2ac_out, 2)
        parameter['BAT2AC_a_out'] = p[0]
        parameter['BAT2AC_b_out'] = p[1]
        parameter['BAT2AC_c_out'] = p[2]

    # BAT2PV conversion pathway
    if parameter['Top'] == 'PV':
        # Create variables for the sampling points and corresponding efficiencies TODO
        p_bat2pv = np.fromiter((value for key, value in parameter.items() if 'p_BAT2PV_' in key), float)
        eta_bat2pv = np.fromiter((value / 100 for key, value in parameter.items() if 'eta_BAT2PV_' in key), float)

        # Absolute input and output power in W
        p_bat2pv_out = parameter['P_BAT2PV_out'] * p_bat2pv * 1000
        p_bat2pv_in = p_bat2pv_out / eta_bat2pv

        # Absolute power loss in W
        P_l_bat2pv_in = (1 - eta_bat2pv) * p_bat2pv_in
        P_l_bat2pv_out = (1 / eta_bat2pv - 1) * p_bat2pv_out

        # Polynomial curve fitting parameters of the power loss functions in W

        # Based on input power TODO
        p = np.polyfit(p_bat2pv_in / parameter['P_BAT2AC_in'] / 1000, P_l_bat2pv_in, 2)
        parameter['BAT2PV_a_in'] = p[0]
        parameter['BAT2PV_b_in'] = p[1]
        parameter['BAT2PV_c_in'] = p[2]

        # Based on output power
        p = np.polyfit(p_bat2pv, P_l_bat2pv_out, 2)
        parameter['BAT2PV_a_out'] = p[0]
        parameter['BAT2PV_b_out'] = p[1]
        parameter['BAT2PV_c_out'] = p[2]

    # Additional parameters

    # Mean battery capacity in kWh
    try:
        parameter['E_BAT'] = (parameter['E_BAT_usable'] / parameter['eta_BAT'] * 100 + parameter['E_BAT_usable']) / 2
    except:
        parameter['E_BAT'] = None

    # Mean stationary deviation of the charging power in W
    try:
        parameter['P_PV2BAT_DEV'] = parameter['P_PV2BAT_DEV_IMPORT'] - parameter['P_PV2BAT_DEV_EXPORT']
    except:
        parameter['P_PV2BAT_DEV'] = None

    if parameter['Top'] == 'AC':
        parameter['P_AC2BAT_DEV'] = parameter['P_PV2BAT_DEV']

        # Mean stationary deviation of the discharging power in W
    try:
        parameter['P_BAT2AC_DEV'] = parameter['P_BAT2AC_DEV_EXPORT'] - parameter['P_BAT2AC_DEV_IMPORT']
    except:
        parameter['P_BAT2AC_DEV'] = None

    # Time constant for the first-order time delay element in s
    try:
        parameter['t_CONSTANT'] = (parameter['t_SETTLING'] - round(parameter['t_DEAD'])) / 3
    except:
        parameter['t_CONSTANT'] = None

    # Hysteresis threshold for the recharging of the battery
    parameter['SOC_h'] = 0.98

    # Feed-in power limit in kW/kWp
    parameter['p_ac2g_max'] = 0.7

    return parameter


def main():
    """
    This is the main function which gets called when this module gets called
    when this module is executed as a script.
    """

    df = read_excel_to_df()
    df = convert_to_nan(df)
    df = transpose_df(df)

    df = df.reset_index(drop=True)
    indexes = df.index.tolist()

    for index in indexes:
        row_dict = df.iloc[index, :].to_dict()
        parameter = assign_specific_values(row_dict)
        parameter = eta2abc(parameter)
        for key in parameter.keys():
            df.loc[index, key] = parameter.get(key)

    df = drop_columns(df)
    df = rename_columns(df)
    df = df.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

    export_to_csv(df)


if __name__ == "__main__":
    main()
