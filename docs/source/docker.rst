
Using SCAMP's Docker image
==========================
Rather than building from scratch you can run SCAMP via `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_ using the prebuilt `image <https://hub.docker.com/r/zpzim/scamp>`_ on dockerhub.

The docker image uses an ideal environment for SCAMP and builds the project with a new compiler, both CPU/GPU performance should be optimal in the docker container.

The docker image is useful if you want to utilize SCAMP in a distributed environment and start Linux VMs which can run SCAMP as needed.

In order to expose the host GPUs nvidia-docker must be installed correctly. Please follow the directions provided on the nvidia-docker github page. The following example uses docker 19.03 functionality::

  docker pull zpzim/scamp:latest
  docker run --gpus all \
    --volume /path/to/host/input/data/directory:/data \
    --volume /path/to/host/output/directory:/output \
    zpzim/scamp:latest /SCAMP/build/SCAMP \
      --window=<window_size> --input_a_file_name=/data/<filename> \
      --output_a_file_name=/output/<mp_filename> \
      --output_a_index_file_name=/output/<mp_index_filename>
