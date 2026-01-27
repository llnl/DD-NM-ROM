#!/bin/bash -i
#flux: -N 1
#flux: -q pbatch
#flux: -t 60
#flux: --exclusive
#flux: --setattr=thp=always
#flux: --error=pod_err.txt
#flux: --job-name=pod
#flux: --output=pod_out.txt

### Slurm syntax
### ---------------
#SBATCH -N 1                     #number of nodes
#SBATCH -t 24:00:00              #walltime in hours:minutes:seconds
#SBATCH -e pod_err.txt      #stderr
#SBATCH -o pod_out.txt      #stdout
#SBATCH -J pod              #name of job
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

python -u ./../../steady/scripts/perform_pod.py --inpfile ./../inputs/perform_pod.json
