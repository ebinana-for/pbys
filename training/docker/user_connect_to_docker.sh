#!/bin/bash
UID=$(id -u)
GID=$(id -g)
UNAME=$(whoami)
docker exec --user $UID:$GID -it forvia_$UNAME bash
