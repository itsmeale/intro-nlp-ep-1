# -*- coding: utf-8 -*-
import re
import pandas as pd
from unidecode import unidecode


def preprocess_on_copy(function):
    def decorate_function(df, document_col):
        new_df = df.copy()
        del df
        return function(new_df, document_col)

    return decorate_function


@preprocess_on_copy
def remove_punctuation(df: pd.DataFrame, document_col):
    removal = lambda s: re.sub("[^A-Za-z0-9\s]", "", s)
    df[document_col] = df[document_col].apply(removal)
    return df


@preprocess_on_copy
def remove_accents(df: pd.DataFrame, document_col):
    df[document_col] = df[document_col].apply(unidecode)
    return df


@preprocess_on_copy
def to_lowercase(df: pd.DataFrame, document_col):
    df[document_col] = df[document_col].str.lower()
    return df


@preprocess_on_copy
def collapse_spaces(df: pd.DataFrame, document_col):
    collapse = lambda s: re.sub(" +", " ", s)
    df[document_col] = df[document_col].apply(collapse)
    df[document_col] = df[document_col].str.strip()
    return df
