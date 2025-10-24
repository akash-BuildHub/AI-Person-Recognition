#!/usr/bin/env bash
set -o errexit

apt-get update
apt-get install -y cmake build-essential pkg-config libboost-all-dev libssl-dev libffi-dev libjpeg-dev libpng-dev

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt