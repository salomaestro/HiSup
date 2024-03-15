import argparse
import logging
import os
import sys
from pathlib import Path
import cv2

import numpy as np
import torch
from skimage import io

sys.path.append("/storage/experiments/hisup")

from hisup.config import cfg
from hisup.dataset.build import build_transform
from hisup.detector import BuildingDetector
from hisup.utils.checkpoint import DetectronCheckpointer
from hisup.utils.comm import to_single_device
from hisup.utils.visualizer import show_polygons


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument(
        "--config-file",
        default="configs/retinanet.yaml",
        help="path to config file",
        type=Path,
    )

    parser.add_argument(
        "--model",
        type=Path,
        help="path to .pth file of model to use",
        required=True,
    )

    parser.add_argument(
        "--img",
        required=True,
        type=Path,
        help="path to test image",
    )

    args = parser.parse_args()

    return args


def test(cfg, args):
    logger = logging.getLogger("inference")
    device = cfg.MODEL.DEVICE
    if not torch.cuda.is_available():
        device = "cpu"

    image = io.imread(args.img)[:, :, :3]
    H, W = image.shape[:2]
    img_mean, img_std = [], []
    for i in range(image.shape[-1]):
        pixels = image[:, :, i].ravel()
        img_mean.append(np.mean(pixels))
        img_std.append(np.std(pixels))

    # cfg.DATASETS.IMAGE.PIXEL_MEAN = img_mean
    # cfg.DATASETS.IMAGE.PIXEL_STD = img_std

    model = BuildingDetector(cfg, test=True)
    model = model.to(device)

    checkpointer = DetectronCheckpointer(
        cfg,
        model,
        save_dir=cfg.OUTPUT_DIR,
        save_to_disk=True,
        logger=logger,
    )

    checkpointer.load(str(args.model), use_latest=False)

    model = model.eval()

    transform = build_transform(cfg)
    image_tensor = transform(image.astype(float))[None].to(device)
    meta = {
        "height": image.shape[0],
        "width": image.shape[1],
    }

    with torch.no_grad():
        output, _ = model(image_tensor, [meta])
        output = to_single_device(output, "cpu")

    if len(output["polys_pred"]) > 0:
        polygons = output["polys_pred"][0]
        show_polygons(image, polygons)
    else:
        print("No building polygons.")


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    test(cfg, args)
