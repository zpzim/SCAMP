FROM nvidia/cuda:9.1-runtime

ADD aws/run_job_self_join.sh .
ADD src/SCRIMP-GPU .

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y \
    python \
    python-pip \
    python-virtualenv \
    zip \
    unzip 

RUN pip install awscli
