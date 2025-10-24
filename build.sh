#!/usr/bin/env bash
# build.sh

# Exit on error
set -o errexit

# Modify this line to use the desired Python version
pyenv install 3.10.8 -s
pyenv global 3.10.8

pip install --upgrade pip
pip install -r requirements.txt