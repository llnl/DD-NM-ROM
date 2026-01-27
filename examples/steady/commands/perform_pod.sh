#!/bin/bash -i

### LSF syntax
### ---------------
#BSUB -nnodes 1            #number of nodes
#BSUB -W 12:00             #walltime in hours:minutes
#BSUB -e pod_err.txt       #stderr
#BSUB -o pod_out.txt       #stdout
#BSUB -J pod               #name of job
#BSUB -q pbatch            #queue to use
#BSUB -G sosu              #account

### Shell scripting
### ---------------
### Loading conda env thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_coral
### Launch program
python -u ./../scripts/perform_pod.py --inpfile ./../inputs/perform_pod.json
