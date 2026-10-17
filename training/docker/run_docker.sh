#!/bin/bash
DATADIR=/mnt/data/forvia
UNAME=$(whoami)

docker run -d --rm --runtime=nvidia \
        --name forvia_$UNAME  \
        -v $PWD:/workspace \
        -v $DATADIR:/data \
        -p 1240-1242:1240-1242 \
        -w /workspace \
        --shm-size="256g"\
        forvia:latest \
        tail -f /dev/null

