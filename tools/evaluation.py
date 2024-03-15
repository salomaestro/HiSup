import argparse

from boundary_iou.coco_instance_api.coco import COCO as BCOCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval as BCOCOeval
from multiprocess import Pool
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hisup.utils.metrics.angle_eval import ContourEval
from hisup.utils.metrics.cIoU import compute_IoU_cIoU
from hisup.utils.metrics.polis import PolisEval


# def parse_coco(cocoEval):
#     return pd.DataFrame(data=cocoEval.stats, index=["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]).to_markdown()

def verify_index(cocoGt, resFile):
    try:
        cocoDt = cocoGt.loadRes(resFile)
        if (cocoDt.getAnnIds() == []):
            print("No detection result found")
            return []
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
    return cocoEval.stats


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
    return cocoEval.stats


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

def fmt_pct(num):
    return f"{num*100:.1f}"

def fmt(num):
    return f"{num:.3f}"

if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    parser.add_argument(
        "--eval-type",
        default=["coco_iou", "boundary_iou", "polis", "angle", "ciou"],
        choices=["coco_iou", "boundary_iou", "polis", "angle", "ciou"],
        nargs="+",
    )
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    results = dict()
    if "coco_iou" in eval_type:
        res = coco_eval(gt_file, dt_file)
        results["AP"] = fmt_pct(res[0])
        results["AR$_{100}$"] = fmt_pct(res[8])
    if "boundary_iou" in eval_type:
        res = boundary_eval(gt_file, dt_file)
        results["AP$^{boundary}$"] = fmt_pct(res[0])
    if "polis" in eval_type:
        results["PoLiS"] = fmt(polis_eval(gt_file, dt_file))
    if "angle" in eval_type:
        results["MTA"] = fmt(max_angle_error_eval(gt_file, dt_file))
    if "ciou" in eval_type:
        res = compute_IoU_cIoU(dt_file, gt_file)
        results["IoU"] = fmt_pct(res[0])
        results["cIoU"] = fmt_pct(res[1])

    print(pd.DataFrame(results, index=[0]).to_markdown())
