#!/bin/bash

rsync -avzh --files-from=include.txt springfield:~/experiments/hisup
