import os
import os.path as osp


class DatasetCatalog(object):
    # DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
    #             '..','..','data'))
    DATA_DIR = osp.abspath("/storage/datasets")

    DATASETS = {
        "crowdai_train_small": {
            "img_dir": "crowdai/train/images",
            "ann_file": "crowdai/train/annotation-small.json",
        },
        "crowdai_test_small": {
            "img_dir": "crowdai/val/images",
            "ann_file": "crowdai/val/annotation-small.json",
        },
        "crowdai_train": {
            "img_dir": "crowdai/train/images",
            "ann_file": "crowdai/train/annotation.json",
        },
        "crowdai_test": {
            "img_dir": "crowdai/val/images",
            "ann_file": "crowdai/val/annotation.json",
        },
        "inria_train": {
            "img_dir": "inria/train/images",
            "ann_file": "inria/train/annotation.json",
        },
        "inria_test": {
            "img_dir": "coco-Aerial/val/images",
            "ann_file": "coco-Aerial/val/annotation.json",
        },
        "terratec_train_tiny": {
            "img_dir": "terratec/subset-32-1-514-127-20_39o/train/images",
            "ann_file": "terratec/subset-32-1-514-127-20_39o/train/annotation.json",
        },
        "terratec_test_tiny": {
            "img_dir": "terratec/subset-32-1-514-127-20_39o/test/images",
            "ann_file": "terratec/subset-32-1-514-127-20_39o/test/annotation.json",
        },
        "terratec_val_tiny": {
            "img_dir": "terratec/subset-32-1-514-127-20_39o/val/images",
            "ann_file": "terratec/subset-32-1-514-127-20_39o/val/annotation.json",
        },
        "terratec_train_small": {
            "img_dir": "terratec/dataset2018/train/images",
            "ann_file": "terratec/dataset2018/train/annotation-small.json",
        },
        "terratec_test_small": {
            "img_dir": "terratec/dataset2018/test/images",
            "ann_file": "terratec/dataset2018/test/annotation-small.json",
        },
        "terratec_val_small": {
            "img_dir": "terratec/dataset2018/val/images",
            "ann_file": "terratec/dataset2018/val/annotation-small.json",
        },
        "terratec_train": {
            "img_dir": "terratec/dataset2018/train/images",
            "ann_file": "terratec/dataset2018/train/annotation.json",
        },
        "terratec_test": {
            "img_dir": "terratec/dataset2018/test/images",
            "ann_file": "terratec/dataset2018/test/annotation.json",
        },
        "terratec_val": {
            "img_dir": "terratec/dataset2018/val/images",
            "ann_file": "terratec/dataset2018/val/annotation.json",
        },
        "terratec_overtrain": {
            "img_dir": "terratec/dataset2018/overtrain/images",
            "ann_file": "terratec/dataset2018/overtrain/annotation.json",
        },
    }

    @staticmethod
    def get(name, mode):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root=osp.join(data_dir, attrs["img_dir"]),
            ann_file=osp.join(data_dir, attrs["ann_file"]),
        )

        if "train" in mode:
            return dict(factory="TrainDataset", args=args)
        if ("test" in mode or "val" in mode) and "ann_file" in attrs:
            return dict(factory="TestDatasetWithAnnotations", args=args)
        raise NotImplementedError()
