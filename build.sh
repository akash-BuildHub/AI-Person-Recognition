#!/usr/bin/env bash
# Exit immediately on error
set -o errexit

# Update and install system-level dependencies
apt-get update
apt-get install -y cmake build-essential pkg-config libboost-all-dev libssl-dev libffi-dev libjpeg-dev libpng-dev

# Upgrade pip and install setuptools + wheel before installing the main dependencies
pip install --upgrade pip setuptools wheel

# Now install your Python dependencies
pip install -r requirements.txt