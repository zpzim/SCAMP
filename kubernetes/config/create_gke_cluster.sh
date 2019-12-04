#!/bin/bash
#Sample script which launches a GKE cluster in zone us-west1-a with preemptible V100 GPU nodes and a high-memory server node
#Configures the nodes to run SCAMPclient and SCAMPserver using the provided template configuration files
#Requires gcloud cli to be configured with your gcp info and the proper quotas for preemptible gpus, cpus, etc.
CLUSTER_NAME="scamp-cluster"
SERVER_MACHINE_TYPE="n1-highmem-8"
WORKER_MACHINE_TYPE="n1-standard-4"
ZONE="us-west1-a"
# Create cluser with preemptible GPU nodes to run workers, with autoscaling up to 40 GPUs
gcloud container clusters create $CLUSTER_NAME --zone $ZONE --num-nodes 1 --enable-autoscaling --min-nodes 0 --max-nodes 10 --machine-type=$WORKER_MACHINE_TYPE --accelerator type=nvidia-tesla-v100,count=4 --preemptible
# Create pool with a single non-preemptible cpu node for server
gcloud container node-pools create server-pool --zone $ZONE --cluster $CLUSTER_NAME --num-nodes 1 --machine-type $SERVER_MACHINE_TYPE
# Install GPU drivers on instances
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
# Start service and server/worker Deployments
kubectl apply -f service.yaml.template
kubectl apply -f server.yaml.template
kubectl apply -f client.yaml.template

