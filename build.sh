#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -o errexit

# Update and install system-level dependencies
apt-get update
apt-get install -y cmake build-essential pkg-config

# Optional (recommended): Install image libraries
apt-get install -y libjpeg-dev libpng-dev libboost-all-dev

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt