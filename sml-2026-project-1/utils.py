"""Utility functions for project 1."""
import yaml
import os
import numpy as np
from pathlib import Path
import pandas as pd
from PIL import Image

from sklearn.metrics import mean_absolute_error

IMAGE_SIZE = (300, 300)


def load_config():
    utils_dir = Path(__file__).parent
    config_path = utils_dir / "config.yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = utils_dir / data_dir
    config["data_dir"] = data_dir

    config["color_space"] = config.get("color_space", "RGB").upper()
    if config["color_space"] not in {"RGB", "GRAY", "HSV"}:
        raise ValueError("color_space must be one of RGB, GRAY, or HSV")

    if config.get("downsample_factor") is None:
        raise NotImplementedError("Make sure to set downsample_factor!")

    print(f"[INFO]: Configs are loaded with: \n {config}")
    return config


def _prepare_image(image, config):
    color_space = config["color_space"]
    if color_space == "GRAY":
        image = image.convert("L")
    elif color_space == "HSV":
        image = image.convert("HSV")
    else:
        image = image.convert("RGB")

    image = image.resize(
        (
            IMAGE_SIZE[0] // config["downsample_factor"],
            IMAGE_SIZE[1] // config["downsample_factor"],
        ),
        resample=Image.BILINEAR,
    )
    return np.asarray(image).reshape(-1).astype(np.float32)


def load_dataset(config, split="train"):
    labels = pd.read_csv(
        config["data_dir"] / f"{split}_labels.csv", dtype={"ID": str}
    )

    channels = 1 if config["color_space"] == "GRAY" else 3
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    ) * channels
    images = np.zeros((len(labels), feature_dim), dtype=np.float32)

    for idx, (_, row) in enumerate(labels.iterrows()):
        image = Image.open(config["data_dir"] / f"{split}_images" / f"{row['ID']}.png")
        images[idx] = _prepare_image(image, config)

    distances = labels["distance"].to_numpy(dtype=np.float32)
    return images, distances


def load_test_dataset(config):
    channels = 1 if config["color_space"] == "GRAY" else 3
    feature_dim = (IMAGE_SIZE[0] // config["downsample_factor"]) * (
        IMAGE_SIZE[1] // config["downsample_factor"]
    ) * channels

    images = []
    img_root = os.path.join(config["data_dir"], "test_images")

    for img_file in sorted(os.listdir(img_root)):
        if img_file.endswith(".png"):
            image = Image.open(os.path.join(img_root, img_file))
            images.append(_prepare_image(image, config))

    return np.array(images, dtype=np.float32)


def print_results(gt, pred, label="Result"):
    print(f"{label} MAE: {round(mean_absolute_error(gt, pred)*100, 3)}")


def save_results(pred):
    text = "ID,Distance\n"

    for i, distance in enumerate(pred):
        text += f"{i:03d},{distance}\n"

    with open("predictions.csv", "w") as f:
        f.write(text)
