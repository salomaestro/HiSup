import argparse

import torch

from hisup.config import cfg
from hisup.config.paths_catalog import DatasetCatalog
from hisup.dataset.transforms import Compose, Resize, ToTensor, Normalize
from hisup.dataset import train_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default=None,
                        )

    parser.add_argument("--clean",
                        default=False,
                        action='store_true')

    parser.add_argument("--seed",
                        default=2,
                        type=int)

    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )

    args = parser.parse_args()

    return args


def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset, dargs['factory'])
    args = dargs['args']
    args['transform'] = Compose(
        [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                cfg.DATASETS.IMAGE.WIDTH,
                cfg.DATASETS.TARGET.HEIGHT,
                cfg.DATASETS.TARGET.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255),
         ])
    args['rotate_f'] = cfg.DATASETS.ROTATE_F
    dataset = factory(**args)

    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=train_dataset.collate_fn,
                                          shuffle=True,
                                          num_workers=cfg.DATALOADER.NUM_WORKERS)
    return dataset


if __name__ == "__main__":
    args = parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    train_dataset = build_train_dataset(cfg)
