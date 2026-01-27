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
