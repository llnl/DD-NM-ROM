# Conda enviroment setup @ LLNL
---

**Choose your system**

Before creating an environment, ensure you select the appropriate system (Toss or Coral) where you intend to install the library.

**Creating an environment**

1. Modify the parameters in the `llnl_<system>/create_env.sh` script to suit your requirements.
2. Run the following command in your terminal:
```
bash llnl_<system>/create_env.sh
```

**Loading an environment**

To load an existing environment:

1. Incorporate the `load_conda_env_<system>` function from `llnl_<system>/load_env.sh` into your `~/.bashrc` file.
2. After re-sourcing the bash environment, execute the function with `-h` argument to view all required inputs:
```
source ~/.bashrc
load_conda_env_<system> -h
```

**Launching a job**

To launch a job, refer to the example in `llnl_<system>/jobscript.sh`. Use the appropriate command based on your system:

- For Toss machines, run:
```
sbatch jobscript.sh
```
- For Coral machines, run:
```
bsub < jobscript.sh
```


# For Tuolumne:
---

For the most recent ROCm, the public torch package is now recommended:

```
module load python/3.13.2 rocm/7.2.1 rccl

python3 -m venv /path/to/venv/dir/

source /path/to/venv/dir/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/rocm7.2

pip install -r requirements.txt

```

For recent ROCm versions, compile a local sparselinear package instead of using
the workaround `create_env.sh` for torch_sparse and torch_scatter. The SparseLinear
package does not need them, but recent versions are not available through pip.

```
sh ./install_sparselinear.sh
```

Ensure MPI is installed matching the current environment, e.g,

```
pip install mpi4py==4.1.1+mpich.9.1.0
```

- *Optional* for direct sparse solves on ROCm, the strumpack backend is supported through torch_sla. Strumpack and its dependencies can be installed using the `./install_strumpack_rocm72.sh` script.


**Running with Flux**

- Request an allocation using your LC bank:

```
flux alloc -N 1 -q pdebug -B <BANK> -t 60 -x -S thp=always
```

- Load environment
```
module load python/3.13.2 rocm/7.2.1 rccl

export HSA_XNACK=1

source /path/to/venv/bin/activate
```

- For serial runs with CPU/numpy backend, the python or pytest executable can be used directly.

- For parallel runs, invoke using flux run. For example, a 4-rank run:

```
flux run -N 1 -x -n 4 -g 1 -c 1 -vvv --setopt=mpibind=verbose:1 python path/to/script.py
```

- For parallel unit tests:

```
flux run -N 1 -x -n 4 -g 1 -c 1 -vvv --setopt=mpibind=verbose:1 python -u -m pytest --verbose -s -x --with-mpi tests/
```

The `--backend=` flag can be used to select `numpy`, `torch_cpu` or `torch_gpu`.
