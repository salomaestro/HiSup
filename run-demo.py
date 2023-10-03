#!/usr/bin/env python3

from subprocess import run
from time import perf_counter
from pathlib import Path
import logging as log

from init import init, send_notification

log.basicConfig(level=log.INFO)


class Timer:
    def __init__(self, name, logger=None, encapsulation=""):
        self.name = name
        self.logger = print if logger is None else logger

    def __enter__(self):
        self.start = perf_counter()

    def __exit__(self, *args):
        self.logger(
            f'{self.name}: {perf_counter() - self.start:.2f}s')


root = Path("/storage/experiments/hisup")
conda_env = root / "environment.yaml"
init_script = root / "init.sh"
demo_script = root / "scripts/demo.py"

if not conda_env.exists():
    with Timer('Run initial build', log.info):
        run(f'sh {str(conda_env)}', shell=True)

with Timer('Run initialization', log.info):
    init()

with Timer('Run demo', log.info):
    demo = run(
        f'python3 {str(demo_script)} --dataset crowdai --img 000000000027.jpg', shell=True)

if demo.returncode != 0:
    log.error("Demo failed")
    send_notification("Demo failed")
    raise Exception("Demo failed")

send_notification("Demo finished!!!!!")
