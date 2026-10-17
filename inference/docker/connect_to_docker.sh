#!/bin/bash
UNAME=$(whoami)
xhost +
docker exec -it  forvia_oms_library_$UNAME bash