#!/bin/bash
UID=$(id -u)
GID=$(id -g)
UNAME=$(whoami)
docker exec -it forvia_$UNAME bash
