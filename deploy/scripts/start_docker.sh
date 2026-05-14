
#!/bin/bash
set -e

aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 128529977235.dkr.ecr.eu-north-1.amazonaws.com

docker stop mypulsecore-app || true
docker rm mypulsecore-app || true

docker pull 128529977235.dkr.ecr.eu-north-1.amazonaws.com/pulsecore_dockerimg_repo:latest

docker run -d \
  --name mypulsecore-app \
  --restart unless-stopped \
  --env-file /home/ubuntu/app/.env \
  -p 80:8000 \
  128529977235.dkr.ecr.eu-north-1.amazonaws.com/pulsecore_dockerimg_repo:latest