"""
This is the code from https://github.com/zorzi-s/PolyWorldPretrainedNetwork.
@article{zorzi2021polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2111.15491},
  year={2021}
}
"""

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm
from copy import deepcopy

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou

class cIoUEval:

    def __init__(self, coco_gti, cocodt, logger=print):
        self.coco_gti = coco_gti
        self.coco = deepcopy(coco_gti)
        self.cocodt = cocodt
        self.logger = logger
        self.imgIds = coco_gti.getImgIds(catIds=coco_gti.getCatIds())

    def evaluate(self):
        return compute_IoU_cIoU(None, None, self.imgIds, self.logger, self.coco_gti, self.coco)

def compute_IoU_cIoU(input_json, gti_annotations, image_ids=None, logger=print, coco_gti=None, coco=None):
    # Ground truth annotations
    if not coco_gti:
        coco_gti = COCO(gti_annotations)

    # Predictions annotations
    if not coco:
        submission_file = json.loads(open(input_json).read())
        coco = COCO(gti_annotations)
        coco = coco.loadRes(submission_file)


    if not image_ids:
        image_ids = coco.getImgIds(catIds=coco.getCatIds())
    bar = tqdm(image_ids)

    list_iou = []
    list_ciou = []
    pss = []
    for image_id in bar:

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        N = 0
        for _idx, annotation in enumerate(annotations):
            try:
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            except Exception:
                import ipdb; ipdb.set_trace()
            m = cocomask.decode(rle)
            if _idx == 0:
                mask = m.reshape((img['height'], img['width']))
                N = len(annotation['segmentation'][0]) // 2
            else:
                mask = mask + m.reshape((img['height'], img['width']))
                N = N + len(annotation['segmentation'][0]) // 2

        mask = mask != 0


        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        N_GT = 0
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            if _idx == 0:
                mask_gti = m.reshape((img['height'], img['width']))
                N_GT = len(annotation['segmentation'][0]) // 2
            else:
                mask_gti = mask_gti + m.reshape((img['height'], img['width']))
                N_GT = N_GT + len(annotation['segmentation'][0]) // 2

        mask_gti = mask_gti != 0

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        pss.append(ps)

        bar.set_description("iou: %2.4f, c-iou: %2.4f, ps:%2.4f" % (np.mean(list_iou), np.mean(list_ciou), np.mean(pss)))
        bar.refresh()

    mean_iou = np.mean(list_iou)
    mean_ciou = np.mean(list_ciou)
    logger("Done!")
    logger(f"Mean IoU: {mean_iou}")
    logger(f"Mean C-IoU: {mean_ciou}")

    return mean_iou, mean_ciou



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    compute_IoU_cIoU(input_json=dt_file,
                    gti_annotations=gt_file)
