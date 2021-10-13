"""This module contains functions to simulate battery storage systems"""
import pandas as pd
import numpy as np
from math import isnan


def load_parameter(model_name: str):
    """Loads system parameter from excel file

    :param model_name:
    :param fname: Path to the excel file
    :type fname: string
    :param col_name: Column to read data from
    :type col_name: string
    :return: Dictionary holding parameters from the Excel sheet
    :rtype: dict
    """

    df = pd.read_csv('../input/out.csv')
    df = df.loc[df['ID'] == model_name]

    df.columns = df.columns.str.rstrip('[W]')
    df.columns = df.columns.str.rstrip('[V]')
    df.columns = df.columns.str.rstrip('[s]')
    df.columns = df.columns.str.rstrip('[-coupled]')
    df.columns = df.columns.str.strip()
    print(df.dtypes)
    parameter: dict = df.to_dict(orient='records')[0]

    # Assign specific parameters
    #parameter['P_PV2AC_out_PVINV'] = parameter[col_name][15].value
    parameter['P_PV2AC_out_PVINV'] = parameter['P_PV2AC_out']
    #parameter['P_PV2AC_out'] = ws[col_name][24].value
    parameter['P_PV2AC_out'] = parameter['P_PV2AC_out [W].1']
    #parameter['P_AC2BAT_in_DCC'] = ws[col_name][25].value
    parameter['P_AC2BAT_in_DCC'] = parameter['P_AC2BAT_in']
    #parameter['P_AC2BAT_in'] = ws[col_name][26].value
    parameter['P_AC2BAT_in'] = parameter['P_AC2BAT_in [W].1']
    #parameter['P_BAT2AC_out'] = ws[col_name][27].value
    parameter['P_BAT2AC_out'] = parameter['P_BAT2AC_out']
    #parameter['P_BAT2AC_out_DCC'] = ws[col_name][28].value
    parameter['P_BAT2AC_out_DCC'] = parameter['P_BAT2AC_out [W].1']

    # Specific parameters of DC-coupled systems
    if parameter['Type'] == 'DC':
        parameter['P_AC2BAT_in'] = parameter['P_AC2BAT_in_DCC']  # Nominal charging power (AC) in kW
        parameter['P_BAT2AC_out'] = parameter['P_BAT2AC_out_DCC']

    # Specific parameters of PV inverters and AC-coupled systems
    if parameter['Type'] == 'PVINV' or parameter['Type'] == 'AC' and parameter['P_PV2AC_out_PVINV'] is not None:
        parameter['P_PV2AC_out'] = parameter['P_PV2AC_out_PVINV']

    # Specific parameters of PV-coupled systems
    if parameter['Type'] == 'PV':
        parameter['P_BAT2PV_in'] = parameter['P_BAT2AC_in']
        parameter['P_BAT2AC_out'] = parameter['P_BAT2AC_out_DCC']


    # Convert to kW
    convert_to_kw = ['P_PV2AC_in', 'P_PV2AC_out_PVINV', 'P_PV2AC_out', 'P_AC2BAT_in_DCC', 'P_AC2BAT_in', 'P_BAT2AC_out',
                     'P_BAT2AC_out_DCC', 'P_PV2BAT_in', 'P_BAT2PV_out', 'P_PV2BAT_out', 'P_BAT2AC_in']

    for par in convert_to_kw:
        if parameter[par] is not None:
            parameter[par] = float(parameter[par]) / 1000

    clean_dict = {k: parameter[k] for k in parameter if not isnan(k)}

    return parameter

if __name__ == "__main__":
    parameter = load_parameter('S2')
    print()
