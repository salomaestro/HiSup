import argparse

from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hisup.utils.metrics.angle_eval import ContourEval
from hisup.utils.metrics.cIoU import compute_IoU_cIoU
from hisup.utils.metrics.polis import PolisEval


def parse_coco(cocoEval):
    return {
        "AP": cocoEval.stats[0],
        "AP50": cocoEval.stats[1],
        "AP75": cocoEval.stats[2],
        "APs": cocoEval.stats[3],
        "APm": cocoEval.stats[4],
        "APl": cocoEval.stats[5],
        "AR1": cocoEval.stats[6],
        "AR10": cocoEval.stats[7],
        "AR100": cocoEval.stats[8],
        "ARs": cocoEval.stats[9],
        "ARm": cocoEval.stats[10],
        "ARl": cocoEval.stats[11],
    }


def verify_index(cocoGt, resFile):
    try:
        cocoDt = cocoGt.loadRes(resFile)
    except IndexError as e:
        print("No detection result found")
        return []

    return cocoDt


def coco_eval(annFile, resFile, cocoGt=None):
    type = 1
    annType = ["bbox", "segm"]
    print("Running demo for *%s* results." % (annType[type]))

    if not cocoGt:
        cocoGt = COCO(annFile)

    cocoDt = verify_index(cocoGt, resFile)

    if not cocoDt:
        return []

    imgIds = cocoGt.getImgIds()
    imgIds = imgIds[:]

    cocoEval = COCOeval(cocoGt, cocoDt, annType[type])
    cocoEval.params.imgIds = imgIds
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return parse_coco(cocoEval)


def boundary_eval(annFile, resFile):
    dilation_ratio = 0.02  # default settings 0.02
    cocoGt = BCOCO(annFile, get_boundary=True, dilation_ratio=dilation_ratio)

    cocoDt = verify_index(cocoGt, resFile)

    if not cocoDt:
        return []

    cocoEval = BCOCOeval(
        cocoGt, cocoDt, iouType="boundary", dilation_ratio=dilation_ratio
    )
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return parse_coco(cocoEval)


def polis_eval(annFile, resFile, gt_coco=None):
    if not gt_coco:
        gt_coco = COCO(annFile)

    dt_coco = verify_index(gt_coco, resFile)

    if not dt_coco:
        return []

    polisEval = PolisEval(gt_coco, dt_coco)
    return polisEval.evaluate()


def max_angle_error_eval(annFile, resFile, gt_coco=None):
    if not gt_coco:
        gt_coco = COCO(annFile)

    dt_coco = verify_index(gt_coco, resFile)

    if not dt_coco:
        return []

    contour_eval = ContourEval(gt_coco, dt_coco)
    pool = Pool(processes=20)
    max_angle_diffs = contour_eval.evaluate(pool=pool).mean()
    print("Mean max tangent angle error(MTA): ", max_angle_diffs)
    return max_angle_diffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    parser.add_argument(
        "--eval-type",
        default="coco_iou",
        choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"],
    )
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    if eval_type == "coco_iou":
        coco_eval(gt_file, dt_file)
    elif eval_type == "boundary_iou":
        boundary_eval(gt_file, dt_file)
    elif eval_type == "polis":
        polis_eval(gt_file, dt_file)
    elif eval_type == "angle":
        max_angle_error_eval(gt_file, dt_file)
    elif eval_type == "ciou":
        compute_IoU_cIoU(dt_file, gt_file)
    else:
        raise RuntimeError(
            'please choose a correct type from \
                            ["coco_iou", "boundary_iou", "polis", "angle", "ciou"]'
        )
