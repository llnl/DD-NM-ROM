"""
Launch multiple training jobs for DD-NM-ROM.
"""

import os
import sys
import json
import argparse

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
parser.add_argument("--dry-run", action="store_true", help="Do not submit jobs, generate batch script and inputs only.")
args = parser.parse_args()

with open(args.inpfile) as file:
  inputs = json.load(file)

# Import 'dd_nm_rom' package
# =====================================
with open(inputs["pathfile"]) as file:
  paths = file.read().splitlines()
sys.path.extend(paths)

# Libraries
# =====================================
import copy
import subprocess
import numpy as np

from dd_nm_rom import ops, jobs

# Initialization
# =====================================
# Directories
for path in (
  inputs["paths"]["inp_dir"],
  inputs["paths"]["cmd_dir"]
):
  os.makedirs(path, exist_ok=True)

# Running
# =====================================
# Batch script function
# -------------------------------------
batch_opts = {"nodes": 1,
              "queue": "pbatch"}

system = jobs.get_system()
if (system == "coral"):
  batch_cmd = lambda cmdfile: f"bsub < {cmdfile}"
elif (system == "toss"):
  batch_cmd = lambda cmdfile: f"sbatch {cmdfile}"
elif (system == "tuo"):
  batch_cmd = lambda cmdfile: f"flux batch --flags waitable {cmdfile}"
  # tuolumne-specific batch options:
  batch_opts["queue"] = "" # remove queue, these jobs are wrapped in a flux instance
  batch_opts["walltime"] = "1h"
else:
  raise ValueError("System not valid.")


# Looping over trainable elements
# -------------------------------------
n_jobs = 0
for element in inputs["elements"]:
  edim = inputs["dim"]["ranges"][element]
  dims = ops.generate_combs([
    np.arange(**edim[k]) for k in ("latent_dim", "row_nonzero")
  ])
  for (ld, rnz) in dims:
    text = "Launching training for element "
    text += f"'{element}' with (ld, rnz) = ({ld}, {rnz}) ..."
    print(text)
    # Input file
    # -------------
    # > Set tag
    tag_i = f"{element}_ld_{ld}_rnz_{rnz}"
    # > Set dimensions
    dim_i = copy.deepcopy(inputs["dim"]["default"][element])
    dim_i["latent_dim"] = int(ld)
    dim_i["row_nonzero"] = int(rnz)
    # > Update file
    with open(inputs["paths"]["inpfile"][element]) as file:
      inp_i = json.load(file)
    inp_i["model"]["path"] = inputs["paths"]["save_dir"]+f"/{tag_i}/"
    inp_i["trainable"]["elements"] = [element]
    inp_i["autoencoder"]["dim"] = {element: dim_i}
    if inputs["refine"]["active"]:
      inp_i["autoencoder"]["refine"] = True
      inp_i["model"]["compile"]["lr"] = inputs["refine"]["lr"]
      tag_i += "_ref"

    # assign different devices for each job
    if inp_i["env"]["device"]:
      if inp_i["env"]["device"] == "cuda":
        inp_i["env"]["device_idx"] = n_jobs % 4
        print(" job {} using device {}".format(n_jobs,inp_i["env"]["device_idx"]))

    # > Save file
    inpfile_i = inputs["paths"]["inp_dir"] + f'/train_rom_{tag_i}.json'
    with open(inpfile_i, 'w') as file:
      json.dump(inp_i, file, indent=2)
    # Python script
    # -------------
    pyscript_i = "train_rom"
    if (element == "port"):
      pyscript_i += "_port"
    # Batch script
    # -------------
    cmdfile_i = inputs["paths"]["cmd_dir"] + f'/train_rom_{tag_i}.sh'
    with open(cmdfile_i, 'w') as file:
      file.write(jobs.generate_batch_script("train_rom_"+tag_i,
                                            f"./../../steady/scripts/{pyscript_i}.py", inpfile_i, **batch_opts))
    # Launch program
    # -------------
    if not args.dry_run:
        subprocess.run(
            batch_cmd(cmdfile_i),
            shell=True,
            timeout=1e2,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
    n_jobs += 1

print(f"\nTotal number of jobs: {n_jobs}\n")
