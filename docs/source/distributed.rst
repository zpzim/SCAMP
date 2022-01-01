Distributed Operation
=====================

SCAMP has a client/server architecture built using grpc. Tested on `GKE <https://cloud.google.com/kubernetes-engine/>`_ but should be possible to get working on `Amazon EKS <https://aws.amazon.com/eks/>`_ as well. To use distributed functionality, build the client and server executables via::

  git submodule update --init --recursive
  mkdir build && cd build
  # requires golang and libz
  cmake -DBUILD_CLIENT_SERVER=1 ..
  make -j8

This will produce three executables in build/src/distributed:
    
  * "SCAMPserver": This is the SCAMP server. It accepts jobs via grpc and handles divying them up among worker clients.
  * "SCAMPclient": Run this on worker nodes, it must be configured with the hostname and port where the SCAMPserver is. This is the workhorse of the computation, it will utilize all gpus or cpus on the host system to compute work handed to it by the server. Each worker node should have only one client executable running at a time. Though not completely necessary, these clients should have high bandwidth to the server for best performance.
  * "SCAMP_distributed": This behaves similarly to the SCAMP executable above, except that it issues jobs to the server via rpc instead of computing them locally. use the --hostname_port="hostname:port" to configure the address of the server. Currently does not support any kind of authentication, so it will need to be run inside any firewalls that would block internet traffic to the compute cluster.
 
The server/clients can be set up to run under kubernetes pods using the Dockerfile in this repo. The docker image zpzim/scamp will contain the latest version of the code ready to deploy to kubernetes.

src/distributed/config contains a sample script which will create a GKE cluster using preemptible GPUs and autoscaling as well as sample configuration files for the scamp grpc service, client, and server pods. You should edit these scripts/configuration files to suit your application.

You can use this script to run and execute your own SCAMP workload on GKE as follows::

  cd src/distributed/config && ./create_gke_cluster.sh
  # Once cluster is up and running you can copy your desired input to the server
  kubectl cp <local SCAMP input file> <SCAMP server container name>:/
  # Now you can run SCAMP_distributed on the server and wait for the job to finish
  kubectl exec <SCAMP server container name> -c server -- /SCAMP/build/src/distributed/SCAMP_distributed <SCAMP arguments>
  # Copy the results back to a local storage location
  kubectl cp <SCAMP server container name>:/mp_columns_out .

**Note**: The configuration above runs SCAMP_distributed on the server, this is not required and is actually not the desired functionality. We would like to be able to run this remotely. While this is currently possible to do it is not reflected in our example.

The above example works on GKE but it should be simple to produce an example that works on Amazon EKS as well.

Limitations
***********

  * Server currently does not periodlically save state, so if it dies, all jobs are lost. This will eventually be fixed by adding sever checkpointing.
  * Server currently handles all work in memory and does not write intermediate files to disk. For this reason the server requires a lot of memory to operate on a large input. Eventually the server will operate mostly on files on disk rather than keep all intermediate data in memory.
  * All neighbors profiles and distance matrix summaries are not yet supported in distributed workloads.

Sharded implementation
**********************

The original distributed implementation used `AWS batch <https://aws.amazon.com/batch/>`_ and shards the time series to Amazon S3. This approach avoids the above limitations of our in-memory SCAMPserver, however our initial implementation was very limited in scope and was not extensible to other types of SCAMP workloads, so it is obsolete. The old scripts can be found `here <https://github.com/zpzim/SCAMP/tree/v2.1.0/aws>` for posterity. Though these would be strictly for inspiration, as there are AWS account side configurations required for operation that cannot be provided. 

