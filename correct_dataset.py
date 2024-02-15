import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        default=0,
        type=float,
        help="score threshold; only apply corrections above this score",
    )
    parser.add_argument("corrections_file", help="path to corrections json file")
    parser.add_argument("source_dataset", help="path to source dataset folder")
    parser.add_argument("destination_dataset", help="path to corrected dataset folder")
    args = parser.parse_args()
    src_path = Path(args.source_dataset)
    if not src_path.exists():
        print(f"No source dataset found at {src_path}.")
        exit()

    dest_path = Path(args.destination_dataset)
    if not dest_path.exists():
        dest_path.mkdir(exist_ok=True, parents=True)

    for subdir in src_path.iterdir():
        if subdir.is_dir():
            (dest_path / subdir.name).mkdir()

    corrections_path = Path(args.corrections_file)
    with corrections_path.open() as f:
        corrections = json.load(f)

    corrections_count = 0
    for img_name, original, corrected, score in zip(
        corrections["corrected_paths"],
        corrections["original_labels"],
        corrections["corrected_labels"],
        corrections["corrected_scores"],
    ):
        src_file = src_path / original / img_name
        if original == corrected or score < args.threshold:
            shutil.copyfile(src_file, dest_path / original / img_name)
        else:
            corrections_count += 1
            dest_file = dest_path / corrected / img_name
            if dest_file.exists():
                dest_file = dest_path / corrected / f"{original}_{img_name}"
            shutil.copyfile(src_file, dest_file)

    print(f"{corrections_count} corrections applied to {dest_path}")


if __name__ == "__main__":
    main()
