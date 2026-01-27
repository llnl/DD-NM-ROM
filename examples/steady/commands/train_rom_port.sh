#!/bin/bash -i

### LSF syntax
### ---------------
#BSUB -nnodes 1                  #number of nodes
#BSUB -W 12:00                   #walltime in hours:minutes
#BSUB -e train_rom_port_err.txt  #stderr
#BSUB -o train_rom_port_out.txt  #stdout
#BSUB -J train_rom_port          #name of job
#BSUB -q pbatch                  #queue to use
#BSUB -G sosu                    #account

### Shell scripting
### ---------------
### Loading conda env thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_coral
### Launch program
python -u ./../scripts/train_rom_port.py --inpfile ./../inputs/train_rom_port.json
