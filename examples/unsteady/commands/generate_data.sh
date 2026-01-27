#!/bin/bash -i
#flux: -N 1
#flux: -q pbatch
#flux: -t 60
#flux: --exclusive
#flux: --setattr=thp=always
#flux: --error=gen_data_err.txt
#flux: --job-name=gen_data
#flux: --output=gen_data_out.txt

### Slurm syntax
### ---------------
#SBATCH -N 1                     #number of nodes
#SBATCH -t 24:00:00              #walltime in hours:minutes:seconds
#SBATCH -e gen_data_err.txt      #stderr
#SBATCH -o gen_data_out.txt      #stdout
#SBATCH -J gen_data              #name of job
#SBATCH -p pbatch                #queue to use
#SBATCH -A sosu                  #account

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

### Launch program
python -u ./../scripts/generate_data.py --inpfile ./../inputs/generate_data.json
