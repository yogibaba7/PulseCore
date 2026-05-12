aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 128529977235.dkr.ecr.eu-north-1.amazonaws.com
docker pull 128529977235.dkr.ecr.eu-north-1.amazonaws.com/pulsecore_dockerimg_repo:latest
docker stop my-container || true
docker rm my-container || true
docker run -d -p 80:8000 --name pulsecore-app -e PULSECORE_PAT=edd9838e65f7b41e3b8c5c6d13e3b5ee7b4bab42 -e PULSECORE_USERNAME=yogibaba7 -e YOUTUBEAPI_KEY=AIzaSyD1OnVEoYqN1BOZK57zGeJR0VWOaGDuRLc 128529977235.dkr.ecr.eu-north-1.amazonaws.com/pulsecore_dockerimg_repo:latest

