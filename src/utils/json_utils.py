import os.path

import orjson
import pandas as pd


def json_parser(filepath: str):
    with open(filepath, "r", encoding="utf8") as f:
        data = orjson.loads(f.read())

    # print("mal_id:", data[0]["mal_id"])
    parsed_info = []

    for record in data:
        title = record.get("title", "")
        genres = record.get("genres", [])
        synopsis = record.get("synopsis", "")
        score = record.get("score", [])
        parsed_info.append(
            (
                title,
                [g["name"] for g in genres],
                synopsis,
                score,
            )
        )
    DIR_BASE = os.path.dirname(os.path.abspath(__file__))
    df = pd.DataFrame(parsed_info, columns=["title", "genres", "synopsis", "score"])
    df.to_parquet(
        os.path.join(DIR_BASE, "../../data/processed/parsed_anime_data.parquet"),
        index=False,
        compression="gzip",
    )
