#!/bin/bash -i

# =====================================
# Deep Learning @ LLNL
# =====================================
# The present script loads the desired conda environment
# on LC CORAL 1 systems (i.e., rzansel, lassen, sierra).
# Add the following function to your ~/.bashrc.

load_conda_env_coral() {
  # Print a help message
  usage() {
    echo -e "Usage: load_conda_env [-h] [-d ANACONDA_DIR] [-n ENV_NAME] [-c CUDA_VER]" 1>&2
  }

  # Default arguments
  cuda_ver=11.8
  env_name=opence-1.9.1-cuda-11.8.0
  anaconda_dir=/usr/workspace/${USER}/Applications/anaconda/${SYS_TYPE}

  # Parse arguments
  local OPTIND;
  while getopts ":hd:n:c:" opt; do
    case ${opt} in
      c)
        cuda_ver=$OPTARG
        ;;
      n)
        env_name=$OPTARG
        ;;
      d)
        anaconda_dir=$OPTARG
        ;;
      h) # Display help.
        usage
        return 0
        ;;
      \?)
        echo -e "Invalid option: $OPTARG" 1>&2
        usage
        return 0
        ;;
      :)
        echo -e "Invalid option: $OPTARG requires an argument" 1>&2
        usage
        return 0
        ;;
    esac
  done
  shift $((OPTIND -1))

  # Activate the environment
  module load cuda/${cuda_ver}
  source ${anaconda_dir}/bin/activate
  conda activate ${env_name}
}
