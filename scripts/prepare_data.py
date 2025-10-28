from src.utils.utils import preprocessing_data, get_anime_search_table
from src.db.db_utils import get_info_from_bd
from src.utils.utils import save_embedding_of_all_anime
from src.utils.json_utils import json_parser


def main():
    # preprocessing_data(titles, genres, synopsis)
    # get_anime_search_table(titles, synopsis)

    json_parser("../data/raw/anime.json")
    save_embedding_of_all_anime()


if __name__ == "__main__":
    main()
