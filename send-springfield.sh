#!/bin/bash

rsync -avzhr --files-from=include.txt . springfield:~/experiments/hisup
