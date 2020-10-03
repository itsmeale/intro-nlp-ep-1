# -*- coding: utf-8 -*-

import pandas as pd


def load_dataset(filepath: str) -> pd.DataFrame:
    return pd.read_parquet(filepath)


def write_dataset(dataframe: pd.DataFrame, filepath: str):
    return dataframe.to_parquet(filepath, index=False)


def load_raw_dataset(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, encoding="cp1252", sep=";", na_filter=True)
