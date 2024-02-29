import json
import random
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon
from pycocotools.coco import COCO

PREDICTION_COLOR = [0, 1, 0]
GROUND_TRUTH_COLOR = [1, 0, 1]


def generate_annotations_gt(annotations):
    polygons = []

    for ann in annotations:
        seg = ann["segmentation"][0]
        poly = np.array(seg).reshape((len(seg) // 2, 2))
        polygons.append(Polygon(poly))

    return polygons


def generate_annotations_dt(annotations, ax):
    polygons = []

    for ann in annotations:
        seg = ann["segmentation"][0]
        bbox = ann["bbox"]
        score = ann["score"]

        poly = np.array(seg).reshape((len(seg) // 2, 2))
        polygons.append(Polygon(poly))

        bbox_center = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
        ax.text(
            bbox_center[0],
            bbox_center[1],
            f"{score:.3f}",
            color="white",
            size=8,
            # backgroundcolor="black",
            ha="center",
            va="center",
        )

    return polygons


def display_image(image, ax):
    ax.set_axis_off()
    ax.imshow(image)


def load_prediction(path):
    with open(path) as f:
        return json.load(f)


def create_index(gt_annotations_path, pred_annotations_path):
    gt = COCO(gt_annotations_path)
    pred = load_prediction(pred_annotations_path)
    results = gt.loadRes(pred)

    return gt, results


def get_random_image_id(gt):
    return random.choice(gt.getImgIds())


def load_image(image_path):
    print("Reading image:", image_path)
    return io.imread(image_path)


def get_image_ann(gt, results, image_dir, image_id=None):
    # Choose random image if image_id is not provided
    if image_id is None:
        image_id = get_random_image_id(gt)

    # Convert image_dir to Path if it is not already
    if not isinstance(image_dir, Path):
        image_dir = Path(image_dir)

    # Get image meta data
    image_meta = gt.loadImgs(image_id)[0]
    image_path = image_dir / image_meta["file_name"]

    # Get ground truth and prediction annotations
    gt_annotations_ids = gt.getAnnIds(imgIds=image_meta["id"])
    gt_annotations = gt.loadAnns(gt_annotations_ids)

    pred_annotations_ids = results.getAnnIds(imgIds=image_meta["id"])
    pred_annotations = results.loadAnns(pred_annotations_ids)

    return image_path, gt_annotations, pred_annotations


def gen_nonempty_image_id(gt):
    image_ids = gt.getImgIds()
    for image_id in image_ids:
        ann_ids = gt.getAnnIds(imgIds=image_id)
        if ann_ids:
            yield image_id


def get_nonempty_image_id(gt, n=1):
    for image_id in gen_nonempty_image_id(gt):
        if n == 0:
            break
        yield image_id
        n -= 1


def find_all_nonempty_image_ids(gt):
    return list(gen_nonempty_image_id(gt))


def display_annotations(gt, results, image_dir, image_id):
    if image_id is None:
        image_id = random.choice(gt.getImgIds())

    if not isinstance(image_dir, Path):
        image_dir = Path(image_dir)

    image_path, gt_annotations, pred_annotations = get_image_ann(
        gt,
        results,
        image_dir,
        image_id,
    )

    image = load_image(image_path)

    legend_elements = [
        Patch(facecolor=GROUND_TRUTH_COLOR, edgecolor="none", label="Ground Truth"),
        Patch(facecolor=PREDICTION_COLOR, edgecolor="none", label="Prediction"),
    ]

    # plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1)
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    ax.set_title(image_path.name)
    ax.set_axis_off()
    ax.imshow(image)

    gt_polygons = generate_annotations_gt(gt_annotations)
    dt_polygons = generate_annotations_dt(pred_annotations, ax)

    p = PatchCollection(
        gt_polygons, facecolors=GROUND_TRUTH_COLOR, linewidths=0, alpha=0.3
    )
    ax.add_collection(p)
    p = PatchCollection(
        gt_polygons, facecolors="none", edgecolors=GROUND_TRUTH_COLOR, linewidths=2
    )
    ax.add_collection(p)

    p = PatchCollection(
        dt_polygons, facecolors=PREDICTION_COLOR, linewidths=0, alpha=0.3
    )
    ax.add_collection(p)
    p = PatchCollection(
        dt_polygons, facecolors="none", edgecolors=PREDICTION_COLOR, linewidths=2
    )
    ax.add_collection(p)

    plt.tight_layout()
    return fig, ax

def plot_n_annotations(gt, results, image_dir, n, apply_func=lambda x: x):
    image_ids = apply_func(find_all_nonempty_image_ids(gt))
    figs = []
    axs = []
    for i in range(n):
        fig, ax = display_annotations(gt, results, image_dir, image_ids[i])
        figs.append(fig)
        axs.append(ax)
    return figs, axs


def main(gt_annotations_path, pred_annotations_path, image_dir, image_id, num_images):
    gt, results = create_index(gt_annotations_path, pred_annotations_path)

    if image_id is not None:
        display_annotations(gt, results, image_dir, image_id)
        plt.show()
        return

    for _ in range(num_images):
        display_annotations(gt, results, image_dir, None)
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", "--gt_annotations_path", type=Path, required=True)
    parser.add_argument("-p", "--pred_annotations_path", type=Path, required=True)
    parser.add_argument("-i", "--image_dir", type=Path, required=True)
    parser.add_argument("--image_id", type=int, required=False)
    parser.add_argument("--num_images", type=int, required=False, default=1)

    args = parser.parse_args()

    main(
        args.gt_annotations_path,
        args.pred_annotations_path,
        args.image_dir,
        args.image_id,
        args.num_images,
    )
