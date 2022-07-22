# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY . .

WORKDIR /app/bee_cam_face

RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
CMD [ "python3", "bee_face_detect.py"]
