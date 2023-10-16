from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
import random
import json
from pathlib import Path
from argparse import ArgumentParser

PREDICTION_COLOR = [0, 1, 0]
GROUND_TRUTH_COLOR = [1, 0, 1]

def display_annotations(annotations, colors, ax):
    polygons = []

    for ann in annotations:
        for seg in ann["segmentation"]:
            poly = np.array(seg).reshape((len(seg)//2, 2))
            polygons.append(Polygon(poly))

    p = PatchCollection(polygons, facecolors=colors, linewidths=0, alpha=0.3)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolors="none", edgecolors=colors, linewidths=2)
    ax.add_collection(p)

def display_image(image, ax):
    ax.set_axis_off()
    ax.imshow(image)

def load_prediction(path):
    with open(path) as f:
        return json.load(f)

def main(gt_annotations_path, pred_annotations_path, image_dir, image_id):
    gt = COCO(gt_annotations_path)
    pred = load_prediction(pred_annotations_path)
    results = gt.loadRes(pred)

    if image_id is None:
        image_id = random.choice(gt.getImgIds())

    image_meta = gt.loadImgs(image_id)[0]
    image_path = image_dir / image_meta["file_name"]
    image = io.imread(image_path)

    gt_annotations_ids = gt.getAnnIds(imgIds=image_meta["id"])
    gt_annotations = gt.loadAnns(gt_annotations_ids)

    pred_annotations_ids = results.getAnnIds(imgIds=image_meta["id"])
    pred_annotations = results.loadAnns(pred_annotations_ids)

    legend_elements = [
        Patch(facecolor=GROUND_TRUTH_COLOR, edgecolor="none", label="Ground Truth"),
        Patch(facecolor=PREDICTION_COLOR, edgecolor="none", label="Prediction")
    ]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1)
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    ax.set_title(image_meta["file_name"])
    ax.set_axis_off()
    ax.imshow(image)

    display_annotations(gt_annotations, GROUND_TRUTH_COLOR, ax)
    display_annotations(pred_annotations, PREDICTION_COLOR, ax)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", "--gt_annotations_path", type=Path, required=True)
    parser.add_argument("-p", "--pred_annotations_path", type=Path, required=True)
    parser.add_argument("-i", "--image_dir", type=Path, required=True)
    parser.add_argument("--image_id", type=int, required=False)

    args = parser.parse_args()

    main(args.gt_annotations_path, args.pred_annotations_path, args.image_dir, args.image_id)
