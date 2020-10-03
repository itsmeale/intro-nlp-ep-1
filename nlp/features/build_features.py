# -*- coding: utf-8 -*-
from typing import Dict, List

import pandas as pd
from pathlib import Path

from nlp.toolbox.datasets import write_dataset, load_raw_dataset

RAW_ABORTION_FILE = Path("data", "raw", "ep1_abortion_train.csv")
RAW_GUN_FILE = Path("data", "raw", "ep1_gun-control_train.csv")
BOW_FILEPATH = Path("data", "interim", "bow.parquet")
TFIDF_FILEPATH = Path("data", "interim", "tfidf.parquet")

DATASET_TOPIC: Dict = {
    RAW_ABORTION_FILE: "abortion",
    RAW_GUN_FILE: "gun",
}


def load_datasets():
    datasets: List = list()

    for dataset_path, topic in DATASET_TOPIC.items():
        temp_df = load_raw_dataset(dataset_path)
        temp_df["topic"] = topic
        datasets.append(temp_df)

    return pd.concat(datasets)


def main():
    df: pd.DataFrame = load_datasets()    

    # TODO: implement this
    # preprocessed_df: pd.DataFrame = preprocess_dataframe(df)

    # TODO: implement this
    # bag_of_words_df: pd.DataFrame = generate_bow(preprocessed_df)

    # TODO: implement this
    # tfidf_df: pd.DataFrame = generate_tfidf(preprocessed_df)

    # write_dataset(bag_of_words_df, BOW_FILEPATH)
    # write_dataset(tfidf_df, TFIDF_FILEPATH)


if __name__ == "__main__":
    main()
