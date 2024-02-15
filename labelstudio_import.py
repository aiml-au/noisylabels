import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from label_studio_sdk import Client
from tqdm import tqdm

from dataset_cats_and_dogs.cats_and_dogs_noisy import INTERMEDIATE, create_data_set

NO_API_KEY_VALUE = "MISSING"
LABEL_STUDIO_ENV_VARIABLE = "LABEL_STUDIO_API_KEY"

LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST", "http://localhost:8080")
API_KEY = os.environ.get(LABEL_STUDIO_ENV_VARIABLE, NO_API_KEY_VALUE)

if API_KEY == NO_API_KEY_VALUE:
    raise ValueError(
        "Please use the environment variable"
        + f" {LABEL_STUDIO_ENV_VARIABLE}"
        + " for the label-studio API key"
    )

SCORE = 0.87


def get_task(label: str, filepath: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    result["data"] = {"image": f"file://{filepath.as_posix()}"}
    result["predictions"] = [
        {
            "result": [
                {
                    "from_name": "image_class",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": [f"{label}"]},
                }
            ],
            "score": SCORE,
        }
    ]
    return result


def parse_dataset(folder: Path) -> Tuple[Dict[str, Any], List[str]]:
    classnames = list(sorted(f.name for f in folder.iterdir() if f.is_dir()))
    result = {}
    for idx, classname in enumerate(classnames):
        images = (folder / classname).glob("*.jpg")
        for image in images:
            result[image.name] = (image, idx)
    return result, classnames


def upload_image(image_path: Path, project_id: int):
    """Upload image file to project using Label Studio API"""
    upload_url = f'{LABEL_STUDIO_HOST.rstrip("/")}/api/projects/{project_id}/import'
    response = requests.post(
        upload_url,
        params={"commit_to_project": "true"},
        headers={"Authorization": f"Token {API_KEY}"},
        files={str(image_path): (str(image_path), image_path.open("rb"))},
        timeout=100,
    )
    if response.status_code != 201:
        raise Exception(
            f"Can't upload task data using {upload_url}, "
            f"response status_code = {response.status_code}"
        )


def create_annotation(task_id, label_config, label_index, user_id):
    annotation_url = f"{LABEL_STUDIO_HOST.rstrip('/')}/api/tasks/{task_id}/annotations/"

    annotation = {
        "task": task_id,
        "completed_by": user_id,
        "ground_truth": False,
        "was_cancelled": False,
        "lead_time": 0,
        "result": [
            {
                "value": {"choices": [label_config["choice"]["labels"][label_index]]},
                "from_name": "choice",
                "to_name": label_config["choice"]["to_name"][0],
                "type": "choices",
            },
        ],
    }

    response = requests.post(
        annotation_url,
        headers={"Authorization": f"Token {API_KEY}"},
        timeout=10,
        json=annotation,
    )
    if response.status_code != 201:
        raise Exception(
            f"Can't post annotation data using {annotation_url}, "
            f"response status_code = {response.status_code}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="",
        help="folder containing image classification dataset, one subfolder per class",
    )
    parser.add_argument(
        "--title",
        default="Project created from SDK",
        help="title for the Label Studio project",
    )
    args = parser.parse_args()

    if args.dataset == "":
        create_data_set()
        dataset_path = INTERMEDIATE
    else:
        dataset_path = Path(args.dataset)
    tasks_to_import, classnames = parse_dataset(dataset_path)

    ls = Client(url=LABEL_STUDIO_HOST, api_key=API_KEY)
    users = ls.get_users()
    user_id = users[0].id
    choices = "\n".join([f'<Choice value="{classname}"/>' for classname in classnames])
    project = ls.start_project(
        title=args.title,
        label_config=f"""
    <View>
    <Image name="image" value="$image"/>
    <Choices name="choice" toName="image">
        {choices}
    </Choices>
    </View>
    """,
    )

    for image_name, (image_path, label_index) in tqdm(
        tasks_to_import.items(), desc="Importing images"
    ):
        upload_image(image_path, project.params["id"])
    for task in tqdm(project.get_tasks(), desc="Importing labels"):
        image_name = task["data"]["image"].split("-", 1)[1]
        label_index = tasks_to_import[image_name][1]
        create_annotation(task["id"], project.parsed_label_config, label_index, user_id)


if __name__ == "__main__":
    main()
