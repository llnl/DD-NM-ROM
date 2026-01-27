#!/bin/bash -i

### Slurm syntax
### ---------------
#SBATCH -N 1                     #number of nodes
#SBATCH -t 24:00:00              #walltime in hours:minutes:seconds
#SBATCH -e dd_nmrom_err.txt      #stderr
#SBATCH -o dd_nmrom_out.txt      #stdout
#SBATCH -J dd_nmrom              #name of job
#SBATCH -p pbatch                #queue to use
#SBATCH -A sosu                  #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_toss
### Launch program
python -u ./../scripts/test_dd_nmrom.py --inpfile ./../inputs/test_dd_nmrom.json
