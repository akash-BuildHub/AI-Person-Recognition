#!/usr/bin/env bash
# Exit immediately if any command fails
set -o errexit

# Update system packages and install build tools
apt-get update
apt-get install -y cmake build-essential pkg-config libboost-all-dev libssl-dev libffi-dev libjpeg-dev libpng-dev

# Upgrade pip and install setuptools + wheel for building packages
pip install --upgrade pip setuptools wheel

# Install Python dependencies
pip install -r requirements.txt