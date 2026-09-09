#!/bin/bash
#flux: --job-name=ddnmrom
#flux: --exclusive
#flux: --nodes=2
#flux: --queue=pdebug
#flux: --time-limit=30m
#flux: --bank=asccasc
#flux: --setattr=thp=always

module load python/3.11.5 rocm/6.3.1
module load rccl

export HSA_XNACK=1

# todo; assumes this is launched from the commands/ folder
venv_dir=$(cat ./../../../conda/llnl_toss/venv_path.txt)
echo $venv_dir
source $venv_dir/bin/activate

#source /path/to/virtualenv/bin/activate
#cd /path/to/DD-NM-ROM/repo/examples/unsteady/commands/

# optional: utility to kill job if it detects hang for 10 min after waiting 5
on_hang_stat_and_kill --thresh 10 --delay 5

# train with 2 nodes (8 ranks, 4 ranks/node)
flux run -N 2 -x -n 8 -c 1 -g 1 -vvv --setopt=mpibind=verbose:1 python -u ../../steady/scripts/train_rom.py --inpfile ./../inputs/train_rom_inputs.json
