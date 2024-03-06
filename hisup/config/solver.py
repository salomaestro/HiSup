from yacs.config import CfgNode as CN


SOLVER = CN()
SOLVER.LR_SCHEDULER = "MultiStepLR"
SOLVER.IMS_PER_BATCH = 6
SOLVER.MAX_EPOCH = 30
SOLVER.OPTIMIZER = "ADAM"
SOLVER.BASE_LR = 0.01
SOLVER.BACKBONE_LR_FACTOR=1.0
SOLVER.BIAS_LR_FACTOR = 1
SOLVER.CYCLIC_MAX_LR = 1e-3
SOLVER.CYCLIC_STEP_SIZE_UP = 2000

SOLVER.MOMENTUM = 0.9
SOLVER.WEIGHT_DECAY = 0.0002
SOLVER.WEIGHT_DECAY_BIAS = 0
SOLVER.GAMMA = 0.1

SOLVER.STEPS = (25,)
SOLVER.CHECKPOINT_PERIOD = 1
SOLVER.AMSGRAD = False

SOLVER.STATIC_STEP = 25
