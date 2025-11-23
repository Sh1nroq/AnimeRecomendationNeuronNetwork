from src.utils.utils import preprocessing_data, get_augmentation
from src.db.db_utils import get_info_from_bd
from src.utils.utils import save_embedding_of_all_anime
from src.utils.json_utils import json_parser
from src.utils.utils import get_anime
import pandas as pd

def main():

    # json_parser("../data/raw/anime.json")
    # preprocessing_data("../data/processed/parsed_anime_data.parquet")
    # save_embedding_of_all_anime()
    # get_augmentation("../data/processed/anime_pairs.parquet"

    a = get_anime(filepath= "../data/processed/parsed_anime_data.parquet", name ='Cowboy Bebop', key= 'genres')

    print(type(a))

if __name__ == "__main__":
    main()
