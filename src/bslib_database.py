"""This files contains function to load and edit the data base for bslib"""
import pandas as pd
import numpy as np


def read_excel_to_df():
    """This functions reads an Microsoft Excel file and returns it as an pandas data frame"""
    df = pd.read_excel('../input/PerModPAR.xlsx', sheet_name='Data')

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


def transpose_df(df):
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
    df.to_csv('../input/out.csv', index=False)


def main():
    df = read_excel_to_df()
    df = transpose_df(df)
    df = drop_columns(df)
    df = rename_columns(df)
    export_to_csv(df)


if __name__ == "__main__":
    main()
