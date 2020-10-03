# -*- coding: utf-8 -*-
import re
import pandas as pd
from unidecode import unidecode


def remove_punctuation(df: pd.DataFrame, document_col):
    new_df = df.copy()
    removal = lambda s: re.sub("[^A-Za-z0-9\s]", "", s)
    new_df[document_col] = new_df[document_col].apply(removal)
    return new_df


def remove_accents(df: pd.DataFrame, document_col):
    new_df = df.copy()
    new_df[document_col] = new_df[document_col].apply(unidecode)
    return new_df


def to_lowercase(df: pd.DataFrame, document_col):
    new_df = df.copy()
    new_df[document_col] = new_df[document_col].str.lower()
    return new_df


def collapse_spaces(df: pd.DataFrame, document_col):
    new_df = df.copy()
    collapse = lambda s: re.sub(" +", " ", s)
    new_df[document_col] = new_df[document_col].apply(collapse)
    new_df[document_col] = new_df[document_col].str.strip()
    return new_df
