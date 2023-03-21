# set base image (host OS)
FROM python:3.8-slim

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# Copy code to the working directory
COPY Implementation/ .

# command to run on container start
ENTRYPOINT ["python", "entrypoint.py"]

# docker build -f Dockerfile -t final_project .
# docker run final_project Amazon 
# docker run final_project MovieLens