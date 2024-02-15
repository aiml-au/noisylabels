import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import boto3
import numpy as np
import requests
import torch
from botocore.exceptions import ClientError
from filelock import FileLock, Timeout
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME, get_single_tag_keys
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from instanceGM import main
from PreResNet import ResNet18

LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "localhost:8080")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")

logging.getLogger("PIL").setLevel(logging.WARNING)

inference_lock = FileLock("inference.lock")

train_lock = FileLock("train.lock")

CHECKPOINT_PATH = Path("checkpoints")


class InstanceGMModel(LabelStudioMLBase):
    def __init__(self, project_id: Optional[str] = None, **kwargs):
        super(InstanceGMModel, self).__init__(project_id, **kwargs)

        self.hostname = LABEL_STUDIO_HOST
        self.access_token = LABEL_STUDIO_API_KEY

        self.endpoint_url = kwargs.get("endpoint_url")
        self.model = self.appref.config["MODEL"]

        # FIXME use transforms per
        # https://pytorch.org/vision/master/models.html#using-the-pre-trained-models
        self.transform_test = transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor()]
        )

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        """Inference

        Parameters
        ----------
        tasks : list[dict]
            Label Studio tasks in JSON format
            (https://labelstud.io/guide/task_format.html)
        context : dict
            Label Studio context in JSON format
            (https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)

        Returns
        -------
        list[dict]
            Predictions array in JSON format
        """
        predictions = []
        self.from_name, self.to_name, self.value, self.labels = get_single_tag_keys(
            self.parsed_label_config, "Choices", "Image"
        )
        logging.info(
            f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}"""
        )

        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)

            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform_test(image).cuda()
            with inference_lock:
                model_version = self.get("model_version")
                if self.model is None:
                    self.model = ResNet18(len(self.labels)).cuda()
                    self.model.eval()
                    self.model.version = "INITIAL"
                    self.appref.config["MODEL"] = self.model

                if self.model.version != model_version:
                    state_dicts = torch.load(CHECKPOINT_PATH / f"{model_version}.tar")
                    self.model.load_state_dict(state_dicts["net1_state_dict"])
                    self.model.version = model_version
                logging.info(f"Running prediction with model version {model_version}")
                outputs = self.model(image_tensor.unsqueeze(dim=0))
            score, pred = torch.max(outputs, 1)

            predictions.append(
                {
                    "model_version": model_version,
                    "result": [
                        {
                            "from_name": self.from_name,
                            "to_name": self.to_name,
                            "type": "choices",
                            "score": score.item(),
                            "value": {"choices": [self.labels[pred.item()]]},
                        }
                    ],
                }
            )

        return predictions

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated.

        Parameters
        ----------
        event : str
            event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        data : dict
            the payload received from the event (check Webhook event reference
            (https://labelstud.io/guide/webhook_reference.html))
        """

        try:
            train_lock.acquire(timeout=1)
        except Timeout:
            # Don't start another training process if one is already active
            return

        try:
            self.from_name, self.to_name, self.value, self.labels = get_single_tag_keys(
                self.parsed_label_config, "Choices", "Image"
            )

            image_paths, image_classes = [], []
            logging.info("Collecting annotations...")

            project_id = data["project"]["id"]
            try:
                tasks = self._get_annotated_dataset_snapshot(project_id)
            except Exception as e:
                logging.info(f"Exception: {e}")
                raise

            logging.info(
                f"Getting images with annotations from list of {len(tasks)} tasks"
            )
            for task in tasks:
                if not task.get("annotations"):
                    continue
                annotation = task["annotations"][0]
                # get input text from task data
                if annotation.get("skipped") or annotation.get("was_cancelled"):
                    continue

                image_paths.append(self.get_local_path(self._get_image_url(task)))
                image_classes.append(annotation["result"][0]["value"]["choices"][0])

            dataloader_factory = ImageClassifierDataloaderFactory(
                image_paths, image_classes, self.labels
            )
            logging.info(
                f"Using {len(image_paths)} training images"
                f" with {len(image_classes)} labels"
            )

            logging.info("Train model...")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            model_version = f"{data['project']['title']}_{timestamp}"
            checkpoint_file = CHECKPOINT_PATH / f"{model_version}.tar"
            main(
                dataloader_factory,
                checkpoint_file,
                len(self.labels),
                data["project"]["title"],
            )
            with inference_lock:
                # store new data to the cache
                self.set("model_version", model_version)

            logging.info(f'New model version: {self.get("model_version")}')

            logging.info("fit() completed successfully.")
        finally:
            train_lock.release()

    def _get_image_url(self, task):
        """Get image URL from task object.

        Will auth and generate a presigned s3 URL if appropriate.

        Parameters
        ----------
        task : dict
            The task object.

        Returns
        -------
        str
            The image URL.
        """
        image_url = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        if image_url.startswith("s3://"):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip("/")
            client = boto3.client("s3", endpoint_url=self.endpoint_url)
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": bucket_name, "Key": key},
                )
            except ClientError as exc:
                logging.error(
                    f"Can't generate presigned URL for {image_url}. Reason: {exc}"
                )
        return image_url

    def _get_annotated_dataset(self, project_id):
        """Retrieve annotated data from Label Studio API"""
        download_url = (
            f'{LABEL_STUDIO_HOST.rstrip("/")}/api/projects/{project_id}/export'
        )
        response = requests.get(
            download_url,
            headers={"Authorization": f"Token {LABEL_STUDIO_API_KEY}"},
            timeout=10,
        )
        if response.status_code != 200:
            raise Exception(
                f"Can't load task data using {download_url}, "
                f"response status_code = {response.status_code}"
            )
        return json.loads(response.content)

    def _get_annotated_dataset_snapshot(self, project_id):
        """Retrieve annotated data from Label Studio API"""
        snapshot_url = (
            f'{LABEL_STUDIO_HOST.rstrip("/")}/api/projects/{project_id}/exports'
        )
        logging.info("Creating snapshot")
        create_response = requests.post(
            snapshot_url,
            headers={"Authorization": f"Token {LABEL_STUDIO_API_KEY}"},
            timeout=100,
        )
        if create_response.status_code != 201:
            raise Exception(
                f"Can't create task data snapshot using {snapshot_url}, "
                f"response status_code = {create_response.status_code}"
            )
        create_json = create_response.json()
        snapshot_id = create_json["id"]

        completed = False
        while not completed:
            time.sleep(5)
            logging.info("Listing snapshots")
            list_response = requests.get(
                snapshot_url,
                headers={"Authorization": f"Token {LABEL_STUDIO_API_KEY}"},
                timeout=10,
            )
            if list_response.status_code != 200:
                raise Exception(
                    f"Can't list task data snapshots using {snapshot_url}, "
                    f"response status_code = {list_response.status_code}"
                )
            list_json = list_response.json()
            for snapshot in list_json:
                if snapshot["id"] != snapshot_id:
                    continue
                if snapshot["status"] == "completed":
                    completed = True
                    break
            else:
                raise Exception(f"No data snapshot with ID {snapshot_id}")

        download_url = (
            f'{LABEL_STUDIO_HOST.rstrip("/")}/api/projects/{project_id}/'
            f"exports/{snapshot_id}/download"
        )
        logging.info("Grabbing snapshot")
        response = requests.get(
            download_url,
            headers={"Authorization": f"Token {LABEL_STUDIO_API_KEY}"},
            params={"exportType": "JSON"},
            timeout=100,
        )
        if response.status_code != 200:
            raise Exception(
                f"Can't load task data using {download_url}, "
                f"response status_code = {response.status_code}"
            )
        snapshot_url = (
            f'{LABEL_STUDIO_HOST.rstrip("/")}/api/projects/{project_id}/'
            f"exports/{snapshot_id}"
        )
        logging.info("Deleting snapshot")
        requests.delete(
            snapshot_url,
            headers={"Authorization": f"Token {LABEL_STUDIO_API_KEY}"},
            timeout=10,
        )
        return response.json()


class ImageClassifierDataset(Dataset):
    def __init__(
        self,
        image_paths,
        image_classes,
        transform,
        classes,
        mode,
        pred=[],
        probability=[],
    ):
        self.classes = classes
        self.mode = mode
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        self.images, self.labels = [], []

        indexes = range(len(image_paths))
        if self.mode == "labeled":
            indexes = pred.nonzero()[0]
            self.probability = [probability[i] for i in indexes]
        elif self.mode == "unlabeled":
            indexes = (1 - pred).nonzero()[0]

        for image_path, image_class in zip(
            (image_paths[idx] for idx in indexes),
            (image_classes[idx] for idx in indexes),
        ):
            try:
                with open(image_path, mode="rb") as f:
                    image = Image.open(f).convert("RGB")
                    image = transform(image)
            except Exception as exc:
                logging.error(exc)
                continue
            self.images.append(image)
            self.labels.append(self.class_to_label[image_class])

    def __getitem__(self, index):
        if self.mode == "labeled":
            return (
                self.images[index],
                self.images[index],
                self.labels[index],
                self.probability[index],
            )
        elif self.mode == "unlabeled":
            return self.images[index], self.images[index]
        elif self.mode == "all":
            return self.images[index], self.labels[index], index
        elif self.mode == "test":
            return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


class ImageClassifierDataloaderFactory:
    def __init__(self, image_paths, image_classes, classes):
        self.image_paths = image_paths
        self.image_classes = image_classes
        self.classes = classes
        self.batch_size = 64
        self.num_workers = 5
        self.transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        # FIXME use transforms per
        # https://pytorch.org/vision/master/models.html#using-the-pre-trained-models

    def run(self, mode, pred=[], prob=[]):
        if mode == "warmup":
            dataset = ImageClassifierDataset(
                self.image_paths,
                self.image_classes,
                self.transform,
                self.classes,
                "all",
            )
            return DataLoader(
                dataset=dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
            )
        elif mode == "train":
            labeled_dataset = ImageClassifierDataset(
                self.image_paths,
                self.image_classes,
                self.transform,
                self.classes,
                "labeled",
                pred,
                prob,
            )
            labeled_dataloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            unlabeled_dataset = ImageClassifierDataset(
                self.image_paths,
                self.image_classes,
                self.transform,
                self.classes,
                "unlabeled",
                pred,
            )
            if len(unlabeled_dataset) == 0:
                # FIXME hack: cannot pass an empty dataset to DataLoader, so
                # we just treat the first image as unlabeled.
                # This may be a terrible idea.
                unlabeled_dataset = ImageClassifierDataset(
                    [self.image_paths[0]],
                    [self.image_classes[0]],
                    self.transform,
                    self.classes,
                    "unlabeled",
                    np.zeros(1),
                )
            unlabeled_dataloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )

            return labeled_dataloader, unlabeled_dataloader
        elif mode == "test":
            test_dataset = ImageClassifierDataset(
                self.image_paths,
                self.image_classes,
                self.transform,
                self.classes,
                "test",
            )
            return DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
        elif mode == "eval_train":
            eval_dataset = ImageClassifierDataset(
                self.image_paths,
                self.image_classes,
                self.transform,
                self.classes,
                "all",
            )
            return DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
