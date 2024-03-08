import argparse
import logging
import os
import sys
from pathlib import Path

import wandb

sys.path.append("/storage/experiments/hisup")

import torch

from hisup.config import cfg
from hisup.config.paths_catalog import DatasetCatalog
from hisup.detector import BuildingDetector
from hisup.utils.checkpoint import DetectronCheckpointer
from hisup.utils.logger import setup_logger
from scripts.display_annotations import create_index, plot_n_annotations
from tools.test_pipelines import TestPipeline

torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    subparser = parser.add_subparsers(dest="command")

    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--eval-type",
        type=str,
        help="evalutation type for the test results",
        default="coco_iou",
        choices=["coco_iou", "boundary_iou", "polis", "all"],
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    full_eval_parser = subparser.add_parser("epochs")
    full_eval_parser.add_argument(
        "-f",
        "--files",
        type=Path,
        help="path to .pth files of model at different epochs",
        nargs="+",
    )
    full_eval_parser.add_argument(
        "-a",
        "--all",
        help="evaluate all models in the output directory",
        action="store_true",
    )

    args = parser.parse_args()

    return args


def test(cfg, args, epoch_file=None, epoch=None):
    logger = logging.getLogger("testing")
    device = cfg.MODEL.DEVICE
    model = BuildingDetector(cfg, test=True)
    model = model.to(device)

    if args.config_file is not None:
        checkpointer = DetectronCheckpointer(
            cfg,
            model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True,
            logger=logger,
        )

        if epoch_file is None:
            _ = checkpointer.load()
        else:
            _ = checkpointer.load(str(epoch_file), use_latest=False)

        model = model.eval()

    test_pipeline = TestPipeline(cfg, args.eval_type, wandb.log, epoch)
    test_pipeline.test(model)


if __name__ == "__main__":
    args = parse_args()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    else:
        cfg.OUTPUT_DIR = "outputs/default"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    logger = setup_logger("testing", output_dir)
    logger.info(args)
    if args.config_file is not None:
        logger.info("Loaded configuration file {}".format(args.config_file))
    else:
        logger.info("Loaded the default configuration for testing")

    wandb.init(
        project="terratec-overtrain",  # might need to change this
        config={
            "model": cfg.MODEL.NAME,
            "dataset": cfg.DATASETS.TEST[0],
        },
    )

    if args.command == "epochs":
        if args.all:
            files = sorted(Path(cfg.OUTPUT_DIR).glob("*.pth"))
        else:
            files = sorted(args.files)

        logger.info(
            "Testing models at epochs: {}".format(
                list(map(lambda x: int(x.stem.replace("model_", "")), files))
            )
        )

        for epoch_file in files:
            # Get the epoch number from the filename
            current_epoch = int(epoch_file.stem.replace("model_", ""))
            logger.info("Testing model at epoch {}".format(current_epoch))

            wandb.log({"test": {"epoch": current_epoch}})

            test(cfg, args, epoch_file.absolute(), current_epoch)

            logger.info("Finished testing model at epoch {}".format(current_epoch))

    else:
        test(cfg, args)
