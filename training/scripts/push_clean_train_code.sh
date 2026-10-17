#!/bin/bash
set -e

# Checkout clean branch
git checkout temp_branch
git rm -rf .

# Checkout files from main
git checkout master -- .

# Add a revision file
MAIN_HASH=$(git rev-parse --short master)
echo "Snapshot of master @ $MAIN_HASH" > SOURCE_REVISION.txt

# Commit and force push
git add .
git commit -m "Snapshot from master @ $MAIN_HASH"
git push -f new-origin temp_branch