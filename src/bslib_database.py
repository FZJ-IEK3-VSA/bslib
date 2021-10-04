"""This files contains function to load and edit the data base for bslib"""
import pandas as pd
import numpy as np


def read_excel_to_df():
    """This functions reads an Microsoft Excel file and returns it as an pandas data frame"""
    df = pd.read_excel('../input/PerModPAR.xlsx', sheet_name='Data')

    end = df.loc[:, (df == 'End').any()].columns

    end_name = end.values[0]

    df = df.loc[:, :end_name]

    df = df.iloc[:, :-1]

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

    df = df.loc[:, df.columns.notnull()]

    df = df.replace(r'^\s+$', np.nan, regex=True)

    return df

def edit_dataframe(df):

    drop_cols = [
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

    df['Top'] = df['Type']

    return df


def main():
    df = read_excel_to_df()
    df = transpose_df(df)
    df = edit_dataframe(df)
    export_to_csv(df)


def export_to_csv(df):
    df.to_csv('../input/out.csv', index=False, header=None)


if __name__ == "__main__":
    main()
