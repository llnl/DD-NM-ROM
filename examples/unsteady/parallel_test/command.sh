#!/bin/bash
#flux: -N 1
#flux: -q pdebug
#flux: -t 60
#flux: --exclusive
#flux: --setattr=thp=always
#flux: --error=test_rom_err.txt
#flux: --job-name=test_rom
#flux: --output=test_rom_out.txt

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file

machine="${SYS_TYPE:-toss_4_x86_64_ib}"

if [[ "${machine}" == "toss_4_x86_64_ib" ]] ;
then
    # Dane
    load_conda_env_toss

    python -u main.py --inpfile inputs.json
else
    # Tuolumne
    #source ddnmrom_env/bin/activate
    # todo; assumes this is launched from the commands/ folder
    venv_dir=$(cat ./../../../conda/llnl_toss/venv_path.txt)
    echo ${venv_dir}

    echo "Activating venv.."
    source ${venv_dir}/bin/activate
    echo "Done activating venv"

    export MPICH_GPU_SUPPORT_ENABLED=1
    export HSA_XNACK=1

    # start a spindle session for the batch jobs
    module load spindle
    spindle --start-session --level=high --python-prefix=${venv_dir}

    # show current queue inside instance
    flux queue list

    python -u main.py --inpfile inputs.json

    flux queue list
    flux jobs -R -u $USER

    # wait for the queue to finish
    flux queue drain
    flux job wait --all

    spindle --end-session
fi


