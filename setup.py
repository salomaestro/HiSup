from distutils.core import setup

setup(
    name="hisup",
    version="1.0",
    py_modules=["hisup"],
    install_requires=[
        "yacs",
        "matplotlib",
        "numpy",
        "opencv-python",
        "pillow",
        "pycococreatortools @ git+https://github.com/waspinator/pycococreator.git@114df401e5310c602178b31a48d3bb4cef876258",
        "pycocotools",
        "pyshp",
        "rasterio",
        "scikit-image",
        "scipy",
        "shapely",
        "tqdm",
        "torch",
        "torchvision",
        "descartes",
        # "boundary_iou @ git+https://github.com/bowenc0221/boundary-iou-api.git@37d25586a677b043ed585f10e5c42d4e80176ea9",
        "multiprocess",
    ],
)
