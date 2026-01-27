import os

_SYSTEMS = {"toss_4_x86_64_ib":"toss", "toss_4_x86_64_ib_cray":"tuo", "blueos_3_ppc64le_ib_p9":"coral"}

# ----------------------------------------------------------------
# Functions to generate batch scripts for various job schedulers
# ----------------------------------------------------------------
def get_system() -> str:
    # Check for SYS_TYPE env var, default to TOSS if not found (which uses SLURM)
    sys_type = os.getenv("SYS_TYPE", default="toss_4_x86_64_ib")
    return _SYSTEMS[sys_type]


def generate_batch_script(tag, pyscript, inpfile, **kwargs):
    system = get_system()
    if system == "toss":
        return generate_batch_script_toss(tag, pyscript, inpfile, **kwargs)
    elif system == "tuo":
        return generate_batch_script_tuo(tag, pyscript, inpfile, **kwargs)
    elif system == "coral":
        return generate_batch_script_coral(tag, pyscript, inpfile, **kwargs)
    else:
        raise RuntimeError(f"Invalid system type {system} - failed to generate batch script")


def generate_batch_stub_slurm(jobname, queue='pbatch', nodes=1, walltime="24:00:00", account="sosu"):
    return f"""
### Slurm syntax
### ---------------
#SBATCH -N {nodes:<30} #number of nodes
#SBATCH -t {walltime:<30} #walltime in hours:minutes
#SBATCH -e {jobname+"_err.txt":<30} #stderr
#SBATCH -o {jobname+"_out.txt":<30} #stdout
#SBATCH -J {jobname:<30} #name of job
#SBATCH -p {queue:<30} #queue to use
{f"#SBATCH -A {account:<30} #account" if account is not "" else ""}
"""


def generate_batch_stub_flux(jobname, queue='pbatch', nodes=1, walltime="1d", account=""):
    return f"""
### Flux syntax
### ---------------
#flux: -N {nodes:<30} #number of nodes
{f"#flux: -t {walltime:<30} #walltime in minutes" if walltime is not "" else ""}
#flux: -n 1
#flux: -c 8
#flux: -o gpu-affinity=off
#flux: -o mpibind=verbose:1
#flux: -u
#flux: --setattr=thp=always
#flux: {"--job-name="+jobname:<30} #name of job
#flux: {"--error="+jobname+"_err.txt":<30} #stderr
#flux: {"--output="+jobname+"_out.txt":<30} #stdout
{f"#flux: -q {queue:<30} #queue" if queue is not "" else ''}
{f"#flux: -B {account:<30} #account" if account is not "" else ''}
"""


def generate_batch_stub_lsf(jobname, queue='pbatch', nodes=1, walltime="12:00", account="sosu"):
    return f"""
### LSF syntax
### ---------------
#BSUB -nnodes {nodes:<30} #number of nodes
#BSUB -W {walltime:<30} #walltime in hours:minutes
#BSUB -e {jobname+"_err.txt":<30} #stderr
#BSUB -o {jobname+"_out.txt":<30} #stdout
#BSUB -J {jobname:<30} #name of job
#BSUB -q {queue:<30} #queue to use
{f"#BSUB -G {account:<30} #account" if account is not "" else ""}
"""


def generate_batch_script_toss(tag, pyscript, inpfile, **kwargs):
    stub = generate_batch_stub_slurm(tag, **kwargs)
    return f"""#!/bin/bash -i
{stub}

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_toss
### Launch program
python -u {pyscript} --inpfile {inpfile}
"""


def generate_batch_script_tuo(tag, pyscript, inpfile, **kwargs):
    stub = generate_batch_stub_flux(tag, **kwargs)
    return f"""#!/bin/bash
{stub}

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
#source ddnmrom_env/bin/activate
# todo; assumes this is launched from the commands/ folder
venv_dir=$(cat ./../../../conda/llnl_toss/venv_path.txt)
echo $venv_dir

echo "Activating venv.."
source $venv_dir/bin/activate
echo "Done activating venv"

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1

### Launch program
python -u {pyscript} --inpfile {inpfile}
"""


def generate_batch_script_coral(tag, pyscript, inpfile, **kwargs):
    stub = generate_batch_stub_lsf(tag, **kwargs)
    return f"""#!/bin/bash -i
{stub}

### Shell scripting
### ---------------
### Loading conda environment thanks to interactive shell
### > See: 'dd-nm-rom/conda/README.md' file
load_conda_env_coral
### Launch program
python -u {pyscript} --inpfile {inpfile}
"""

