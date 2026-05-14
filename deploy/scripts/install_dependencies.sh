#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update -y
sudo apt-get install -y unzip curl

# Install Docker only if missing
if ! command -v docker &> /dev/null; then
  sudo apt-get install -y docker.io
fi

sudo systemctl enable docker
sudo systemctl start docker

# Install AWS CLI only if missing
if ! command -v aws &> /dev/null; then
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    -o "/tmp/awscliv2.zip"
  unzip -o /tmp/awscliv2.zip -d /tmp
  sudo /tmp/aws/install --update
fi

# Allow ubuntu user to run Docker
sudo usermod -aG docker ubuntu

# Cleanup
rm -rf /tmp/aws /tmp/awscliv2.zip

# Ensure app directory exists
mkdir -p /home/ubuntu/app