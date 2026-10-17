#!/bin/bash
UNAME=$(whoami)
XSOCK=/tmp/.X11-unix

docker run -d --rm \
        --name forvia_oms_library_$UNAME  \
        -v $PWD/..:/workspace \
        -w /workspace \
        -e DISPLAY=$DISPLAY \
        -v $XSOCK:$XSOCK \
        forvia_oms_library:latest \
        tail -f /dev/null
