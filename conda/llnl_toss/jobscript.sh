#!/bin/bash -i

### Launch command:
### sbatch jobscript.sh

### Slurm syntax
### ---------------
#SBATCH -N 1                     #number of nodes
#SBATCH -t 24:00:00              #walltime in hours:minutes:seconds
#SBATCH -e job_err.txt           #stderr
#SBATCH -o job_out.txt           #stdout
#SBATCH -J job                   #name of job
#SBATCH -p pbatch                #queue to use
#SBATCH -A sosu                  #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_toss
### Launch program
python -u <script> --inpfile <inpfile>
