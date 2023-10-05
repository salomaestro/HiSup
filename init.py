#!/usr/bin/env python3

from subprocess import run
from pathlib import Path
import logging as log
from requests import post

root = Path("/storage/experiments/hisup").absolute()
conda_env = root / "environment.yaml"
lib = root / "hisup/csrc/lib"
afm = lib / "afm_op"
squeeze = lib / "squeeze"

paths = [root, conda_env, lib, afm, squeeze]


def send_notification(message):
    post("https://ntfy.salodev.no/uit", data=message)


def init():

    for path in paths:
        if not path.exists():
            log.error(f"{path} does not exist")
            raise Exception(f"{path} does not exist")

    send_notification("Starting build...")

    # success = False
    # i = 1
    # while not success:
    #     try:
    #         env_run = run(
    #             f'conda env create --file {str(conda_env)}', shell=True)
    #
    #         if env_run.returncode != 0:
    #             raise Exception("Environment creation failed")
    #
    #         success = True
    #     except Exception as e:
    #         log.error(e)
    #         log.info("Retrying... ({})".format(i))
    #         i += 1
    #         if i > 3:
    #             log.error("Environment creation failed")
    #             send_notification("Environment creation failed")
    #             raise e
    #
    # log.info("Environment created, building afm and squeeze...")
    # send_notification("Environment created, building afm and squeeze...")

    try:
        set_cuda = run("export CUDA_HOME=/opt/conda/envs/hisup", shell=True)

        if set_cuda.returncode != 0:
            log.error("Setting CUDA_HOME failed")
            raise Exception("Setting CUDA_HOME failed")

        log.info("CUDA_HOME set successfully")

        afm_build = run(
            "conda run -n hisup python3 setup.py build_ext --inplace", cwd=afm)
        afm_build_rm = run("rm -rf build", cwd=afm)

        if afm_build.returncode != 0 or afm_build_rm.returncode != 0:
            log.error("AFM build failed, trying to build squeeze only...")

        squeeze_build = run(
            "conda run -n hisup python3 setup.py build_ext --inplace", cwd=squeeze)
        squeeze_build_rm = run("rm -rf build", cwd=squeeze)

        if squeeze_build.returncode != 0 or squeeze_build_rm.returncode != 0:
            log.error("Squeeze build failed")

        if afm_build.returncode != 0 or squeeze_build.returncode != 0:
            raise Exception("Build failed")
    except Exception as e:
        log.error(e)
        send_notification("Build failed")
        raise e

    log.info("Build successful")
    send_notification("Build successful")
