import logging
import shutil
from pathlib import Path
from random import random
from typing import Tuple

from datasets import get_dataset_split_names, load_dataset

LENGTH = 10000

INCORRECT_PERCENTAGE = 5

INTERMEDIATE = Path(__file__).parent / "intermediate"

logging.basicConfig(filename="dog_and_cat_dataset.log", level=logging.DEBUG)
logging.getLogger("PIL").setLevel(logging.CRITICAL + 1)


def flip_id(id):
    return int(not bool(id))


def create_data_set(
    length: int = LENGTH, incorrect_percentage: int = INCORRECT_PERCENTAGE
) -> Tuple[int, int, int, int]:
    splits = get_dataset_split_names("cats_vs_dogs")

    train = load_dataset("cats_vs_dogs", split=splits[0])
    iterable_train = train.to_iterable_dataset()
    CAT_ID = 0
    DOG_ID = 1
    ID_KEY = "labels"
    IMAGE_KEY = "image"
    threshold: float = float(incorrect_percentage) / 100.0
    if INTERMEDIATE.is_dir():
        try:
            shutil.rmtree(INTERMEDIATE.as_posix())
        except OSError:
            logging.info(f"Cannot remove {INTERMEDIATE.as_posix}")
    INTERMEDIATE.mkdir(parents=True, exist_ok=False)
    cats_folder = INTERMEDIATE / "cat"
    cats_folder.mkdir(parents=True, exist_ok=False)
    dogs_folder = INTERMEDIATE / "dog"
    dogs_folder.mkdir(parents=True, exist_ok=False)
    count_real_cats: int = 0
    count_real_dogs: int = 0
    count_mislabeled_cats: int = 0
    count_mislabeled_dogs: int = 0
    total = 0
    for idx, sample in enumerate(list(iterable_train)):
        if total >= length:
            break
        id = sample[ID_KEY]
        if count_real_cats > length // 2 and id == CAT_ID:
            continue
        if count_real_dogs > length // 2 and id == DOG_ID:
            continue
        if id == CAT_ID:
            count_real_cats += 1
        else:
            count_real_dogs += 1
        if random() < threshold:  # nosec
            if id == DOG_ID:
                count_mislabeled_dogs += 1
            if id == CAT_ID:
                count_mislabeled_cats += 1
            id = flip_id(id)
        image = sample[IMAGE_KEY]
        if id == CAT_ID:
            image.save((cats_folder / f"{idx}.jpg").as_posix())
        else:
            image.save((dogs_folder / f"{idx}.jpg").as_posix())
        total += 1
    return (
        count_real_cats,
        count_real_dogs,
        count_mislabeled_cats,
        count_mislabeled_dogs,
    )


if __name__ == "__main__":
    (
        count_real_cats,
        count_real_dogs,
        count_mislabeled_cats,
        count_mislabeled_dogs,
    ) = create_data_set()
    logging.info(f"count_real_cats: {count_real_cats}")
    logging.info(f"count_real_dogs: {count_real_dogs}")
    logging.info(f"count_mislabeled_cats: {count_mislabeled_cats}")
    logging.info(f"count_mislabeled_dogs: {count_mislabeled_dogs}")
