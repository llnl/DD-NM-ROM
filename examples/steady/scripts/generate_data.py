"""
Generate steady-state solutions for the 2D Burgers' equation.
"""

import os
import sys
import json
import argparse

# Inputs
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument("--inpfile", type=str, help="path to JSON input file")
args = parser.parse_args()

with open(args.inpfile) as file:
  inputs = json.load(file)

# Import 'dd_nm_rom' package
# =====================================
with open(inputs["pathfile"]) as file:
  paths = file.read().splitlines()
sys.path.extend(paths)

# Environment
# =====================================
from dd_nm_rom import env
env.set(**inputs["env"])

# Libraries
# =====================================
import shutil
import numpy as np
import dill as pickle

from dd_nm_rom import utils
from dd_nm_rom import fom as fom_mod
from dd_nm_rom import field as field_mod
from dd_nm_rom.elements import mesh as mesh_mod

# Initialization
# =====================================
print("\nInitialization ...")
# Mesh
mesh = utils.get_class(modules=[mesh_mod], **inputs["mesh"])
mesh.build()
# Field
field = utils.get_class(
  modules=[field_mod],
  name=inputs["field"]["name"]
)(mesh=mesh, **inputs["field"]["kwargs"])
# FOM
fom = utils.get_class(
  modules=[fom_mod],
  name="Burgers2D"
)(mesh=mesh, **inputs["fom"]["kwargs"])
# Construct design matrix
mu = field.construct_design_mat(n_samples=inputs["data_gen"]["n_samples"])
# Saving path
path_to_save = inputs["save_dir"]
os.makedirs(path_to_save, exist_ok=True)

# Data generation
# =====================================
# Solution function
# -------------------------------------
def compute_sol(index):
  mu_i = mu[index]
  # Set current parameters
  field.set_params(mu_i)
  # Build FOM
  fom.build(field)
  # Solve PDE
  uv, res, converged = fom.solve(**inputs["solver"])
  # Save solution
  if converged:
    case_i = {
      "index": index,
      "mu": mu_i,
      "snapshots": np.concatenate([uv["u"], uv["v"]]),
      "residuals": res,
      "runtime": fom.runtime
    }
    utils.save_case(path=path_to_save, index=index, data=case_i)
  return int(converged)

# Parallel data generation
# -------------------------------------
print("\nData generation ...")
utils.generate_case_parallel(compute_sol, **inputs["data_gen"])
# Save parameters
filename = path_to_save + "/mu.p"
pickle.dump(mu, open(filename, "wb"))
# Copy input file
filename = path_to_save + "/inputs.json"
shutil.copyfile(args.inpfile, filename)

print("\nDone!\n")
