# base model
FROM python:3.11-slim

# workdir
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

# copy
COPY  API/ ./API/

# run
RUN pip install --upgrade pip
RUN pip install -r API/requirements.txt

# expose port
EXPOSE 8000

# COMMAND
CMD ["uvicorn","API.app:app","--host","0.0.0.0","--port","8000"]
