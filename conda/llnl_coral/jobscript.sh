#!/bin/bash -i

### Launch command:
### bsub < jobscript.sh

### LSF syntax
### ---------------
#BSUB -nnodes 1              #number of nodes
#BSUB -W 12:00               #walltime in hours:minutes
#BSUB -e job_err.txt         #stderr
#BSUB -o job_out.txt         #stdout
#BSUB -J job                 #name of job
#BSUB -q pbatch              #queue to use
#BSUB -G sosu                #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_coral
### Launch program
python -u <script> --inpfile <inpfile>
