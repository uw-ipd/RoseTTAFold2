#!/bin/bash

python ./network/predict.py \
    -inputs ./examples/T1165.a3m[1-$1] \
    -prefix T1165 \
    -model ./weights/RF2_apr23.pt \

