# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nlp.toolbox.datasets import write_dataset, load_raw_dataset
from nlp.toolbox.preprocessing import (
    remove_accents,
    remove_punctuation,
    collapse_spaces,
    to_lowercase,
)

RAW_ABORTION_FILE = Path("data", "raw", "ep1_abortion_train.csv")
RAW_GUN_FILE = Path("data", "raw", "ep1_gun-control_train.csv")
BOW_FILEPATH = Path("data", "interim", "bow.parquet")
TFIDF_FILEPATH = Path("data", "interim", "tfidf.parquet")

DATASET_TOPIC: Dict = {
    RAW_ABORTION_FILE: "abortion",
    RAW_GUN_FILE: "gun-control",
}


def load_datasets():
    datasets: List = list()
    for dataset_path, topic in DATASET_TOPIC.items():
        temp_df = load_raw_dataset(dataset_path)
        temp_df["topic"] = topic
        temp_df["opinion"] = temp_df[topic]
        temp_df.drop(columns=[topic], inplace=True)
        datasets.append(temp_df)
    return pd.concat(datasets)


def preprocess_dataframe(
    dataframe: pd.DataFrame, document_col: str = "text"
) -> pd.DataFrame:
    return (
        dataframe.pipe(to_lowercase, document_col)
        .pipe(remove_accents, document_col)
        .pipe(remove_punctuation, document_col)
        .pipe(collapse_spaces, document_col)
        .reset_index()
        .drop(columns=["index"])
    )


def vectorize_text_data(
    df: pd.DataFrame, document_col: str, Vectorizer, ngram_range: Tuple = (1, 1)
) -> pd.DataFrame:
    vectorizer = Vectorizer(ngram_range=ngram_range)
    features = vectorizer.fit_transform(df[document_col].values)
    features_df = pd.DataFrame(
        features.todense(), columns=vectorizer.get_feature_names()
    )
    return pd.concat([features_df, df], axis=1)


def main():
    documents_col: str = "text"
    df: pd.DataFrame = load_datasets()
    preprocessed_df: pd.DataFrame = preprocess_dataframe(df, documents_col)

    bag_of_words_df: pd.DataFrame = vectorize_text_data(
        df=preprocessed_df,
        document_col=documents_col,
        Vectorizer=CountVectorizer,
    )

    tfidf_df: pd.DataFrame = vectorize_text_data(
        df=preprocessed_df,
        document_col=documents_col,
        Vectorizer=TfidfVectorizer,
    )

    write_dataset(bag_of_words_df, BOW_FILEPATH)
    write_dataset(tfidf_df, TFIDF_FILEPATH)


if __name__ == "__main__":
    main()
