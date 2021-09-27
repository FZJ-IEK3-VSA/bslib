"""This files contains function to load and edit the data base for bslib"""
import pandas as pd
import numpy as np


def read_excel_to_df():
    """This functions reads an Microsoft Excel file and returns it as an pandas data frame"""
    df = pd.read_excel('../input/PerModPAR.xlsx', sheet_name='Data', usecols=[4,5,7,8,9,10,11,12,18], na_values='^\s+$')
    df.fillna('', inplace=True)

    return df


def transpose_df(df):
    df = df.T
    return df


def main():
    df = read_excel_to_df()
    df = transpose_df(df)
    export_to_csv(df)


def export_to_csv(df):
    df.to_csv('../input/out.csv', index=False, header=None)


if __name__ == "__main__":
    main()
