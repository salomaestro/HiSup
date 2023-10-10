#!/bin/bash

cp shared-objects/CUDA.cpython-38-x86_64-linux-gnu.so hisup/csrc/lib/afm_op/
cp shared-objects/squeeze.cpython-38-x86_64-linux-gnu.so hisup/csrc/lib/squeeze/

exec "bash"
