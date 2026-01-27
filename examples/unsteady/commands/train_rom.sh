#!/bin/bash -i
#flux: -N 1
#flux: -q pbatch
#flux: -t 60
#flux: --exclusive
#flux: --setattr=thp=always
#flux: --error=train_rom_err.txt
#flux: --job-name=train_rom
#flux: --output=train_rom_out.txt

### LSF syntax
### ---------------
#BSUB -nnodes 1                   #number of nodes
#BSUB -W 12:00                    #walltime in hours:minutes
#BSUB -e train_rom_err.txt        #stderr
#BSUB -o train_rom_out.txt        #stdout
#BSUB -J train_rom                #name of job
#BSUB -q pbatch                   #queue to use
#BSUB -G sosu                     #account

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
machine="${SYS_TYPE:-toss_4_x86_64_ib}"

if [[ "${machine}" == "toss_4_x86_64_ib" ]] ;
then
    # Dane
    load_conda_env_toss
else
    # Tuolumne
    #source ddnmrom_env/bin/activate
    # todo; assumes this is launched from the commands/ folder
    venv_dir=$(cat ./../../../conda/llnl_toss/venv_path.txt)
    echo $venv_dir

    source $venv_dir/bin/activate

    export MPICH_GPU_SUPPORT_ENABLED=1
    export HSA_XNACK=1
fi

python -u ./../../steady/scripts/train_rom.py --inpfile ./../inputs/train_rom.json
