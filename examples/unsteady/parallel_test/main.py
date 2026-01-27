"""
Launch multiple testing jobs for DD-NM-ROM.
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
  batch_opts["walltime"] = ""
else:
  raise ValueError("System not valid.")

# Generate all configurations
# -------------------------------------
dims, cfgs = {}, []
for element in inputs["elements"]:
  dims[element] = ops.generate_combs([
    np.arange(**inputs["dim"]["ranges"][element]["latent_dim"]),
    np.arange(**inputs["dim"]["ranges"][element]["row_nonzero"])
  ])
  cfgs.append(np.arange(len(dims[element])))
cfgs = ops.generate_combs(cfgs)

# Loop over configurations
# -------------------------------------
n_jobs = 0
for cfg in cfgs:
  # > Set tag
  tag_i = {}
  text = "\nLaunching testing for elements:"
  for (e, element) in enumerate(inputs["elements"]):
    ld, rnz = dims[element][cfg[e]]
    tag_i[element] = f"{element}_ld_{ld}_rnz_{rnz}"
    text += f"\n- '{element}' with (ld, rnz) = ({ld}, {rnz})"
  fulltag_i = "_".join(tag_i.values())
  print(text)
  # > Update input file
  with open(inputs["paths"]["inpfile"]) as file:
    inp_i = json.load(file)
  for (element, etag) in tag_i.items():
    inp_i["paths"]["nets_tag"][element] = etag

  # assign different devices for each job
  if inp_i["env"]["device"]:
    if inp_i["env"]["device"] == "cuda":
        inp_i["env"]["device_idx"] = n_jobs % 4
        print(" job {} using device {}".format(n_jobs,inp_i["env"]["device_idx"]))

  # > Save input file
  inpfile_i = inputs["paths"]["inp_dir"] + f'/test_dd_nmrom_{fulltag_i}.json'
  with open(inpfile_i, 'w') as file:
    json.dump(inp_i, file, indent=2)
  # Batch script
  # -------------
  cmdfile_i = inputs["paths"]["cmd_dir"] + f'/test_dd_nmrom_{fulltag_i}.sh'
  with open(cmdfile_i, 'w') as file:
    file.write(jobs.generate_batch_script("test_dd_nmrom_" + fulltag_i,
                                          "./../scripts/test_dd_nmrom.py",
                                          inpfile_i, **batch_opts))
  # Launch program
  # -------------
  if not args.dry_run:
    print(" Launching job with '{}'".format(batch_cmd(cmdfile_i)))
    subprocess.run(
        batch_cmd(cmdfile_i),
        shell=True,
        timeout=1e2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
  n_jobs += 1

print(f"\nTotal number of jobs: {n_jobs}\n")
